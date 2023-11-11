#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import os
from typing import List, Tuple, Dict
from conversion_utils.preprocessing_file_saver import generate_crossvalidation_folds, save_dataset_splits
from collections import Counter
import pprint
import argparse
import sys

from conversion_utils.tempeval_datastructures import (
    Document,
    TagTimex3,
    TagFullEvent,
    extract_makeinstance_objects,
    extract_tlink_objects,
    extract_signal_objects,
    extract_event_objects,
    extract_timex3_objects,
    extract_full_event_objects,
    extract_meta_document_time,
    extract_article_content_timebank_extra,
    generate_sentence_objects,
    strip_article_content_timebank_extra,
    create_tempeval_document,
    split_sentences_timebank_extra,
    extract_article_content_timebank_timeml,
    strip_article_content_timebank_timeml,
    split_sentences_timebank_timeml,
    extract_article_content_aquaint,
    strip_article_content_aquaint,
    split_sentences_aquaint,
)


class TempevalDatasetConverter:
    """
    This class converts the TempEval dataset (consisting of TimeBank and AQUAINT) to the jsonline format. The class takes into account the details of the TimeBank/Aquaint datasets.

    This converter also converts temporal relations, which consist of EVENTS, TIMEX3, MAKEINSTANCE and TLINK tags.
    Temporal relations may occur between two entities (E-E), between an entity and a temporal expression (E-T) or between two temporal expressions (T-T).
    All those cases are inside the TLINK tags.

    It also implements crossvalidation and saves the dataset in multiple copies.

    The dataset is split into two distinct directories with TML files that have different annotation styles. Therefore these
    files have to be handeled seperately.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(
            self, 
            input_filepaths_timebank_extra: List[str], 
            input_filepaths_timebank_timeml: List[str], 
            input_filepaths_aquaint: List[str],
            extract_tempeval: bool,
            extract_timebank: bool,
            extract_aquaint: bool,
            output_base_directory_tempeval_path: str,
            output_base_directory_timebank_path: str,
            output_base_directory_aquaint_path: str,
            single_entity_class: bool = False,
            extract_temporal_expressions: bool = True,
            extract_events: bool = True,
            extract_temporal_relations: bool = True,
            crossvalidation_enabled: bool = False, 
            folds: int = 10
        ) -> None:
        """
        Extraction targets.
        """
        self.extract_temporal_expressions = extract_temporal_expressions
        self.extract_events = extract_events
        self.extract_temporal_relations = extract_temporal_relations

        if extract_temporal_relations and not (extract_temporal_expressions and extract_events):
            raise Exception("Cannot extract temporal relations without extracting temporal expressions and events!")
        
        if not extract_temporal_relations and not extract_temporal_expressions and not extract_events:
            raise Exception("Cannot extract nothing! Please specify at least one extraction target!")


        """
        I/O Filepath variables.
        """
        self.timebank_tml_filepaths_extra = input_filepaths_timebank_extra
        self.timebank_tml_filepaths_timeml = input_filepaths_timebank_timeml
        self.aquaint_tml_filepaths = input_filepaths_aquaint

        #Which parts of the dataset to extract into a seperate jsonlines file
        self.extract_tempeval = extract_tempeval
        self.extract_timebank = extract_timebank
        self.extract_aquaint = extract_aquaint

        #Where to save the converted datasets
        self.output_base_directory_tempeval_path = output_base_directory_tempeval_path
        self.output_base_directory_timebank_path = output_base_directory_timebank_path
        self.output_base_directory_aquaint_path = output_base_directory_aquaint_path


        """
        Load tml files into memory.
        """
        self.tml_file_contents_extra = self.load_tml_files(self.timebank_tml_filepaths_extra) 
        self.tml_file_contents_timeml = self.load_tml_files(self.timebank_tml_filepaths_timeml)
        self.tml_file_contents_aquaint = self.load_tml_files(self.aquaint_tml_filepaths)

        assert len(self.timebank_tml_filepaths_extra) == len(self.tml_file_contents_extra)
        assert len(self.timebank_tml_filepaths_timeml) == len(self.tml_file_contents_timeml)
        assert len(self.aquaint_tml_filepaths) == len(self.tml_file_contents_aquaint)
        

        """
        Define temporal types mapping.
        """
        #Mapping of timex3 types to temporal classes
        if single_entity_class:
            self.temporal_classes_dictionary = {
                "date": "tempexp",
                "time": "tempexp",
                "duration": "tempexp",
                "set": "tempexp"
            }
        else:
            self.temporal_classes_dictionary = {
                "date": "date",
                "time": "time",
                "duration": "duration",
                "set": "set"
            }

        self.event_classes_dictionary = {
            "occurrence": "occurrence",
            "reporting": "reporting",
            "state": "state",
            "i_action": "i-action",
            "i_state": "i-state",
            "aspectual": "aspectual",
            "perception": "perception",
            "empty": "empty"
        }


        """
        Statistics
        """
        self.event_classes = Counter() #Check all event classes when converting to json
        self.relation_classes = Counter() #Check all relation classes when converting to json


        """
        Define output files information.
        """
        self.timebank_output_file_prefix = "timebank"
        self.tempeval_output_file_prefix = "tempeval"
        self.aquaint_output_file_prefix = "aquaint"

        self.output_file_ending = ".jsonlines"
        self.output_file_train_suffix = "-train" + self.output_file_ending
        self.output_file_test_suffix = "-test" + self.output_file_ending
        self.output_file_val_suffix = "-val" + self.output_file_ending        
        self.output_file_full_suffix = "-full" + self.output_file_ending


        """
        Split percentages of the dataset
        """
        self.train_percent = 0.8
        self.test_percent = 0.1
        self.val_percent = 0.1


        """
        Crossvalidation.
        """
        self.folds = folds #If crossvalidation is enabled, how many folds to create
        self.crossvalidation_enabled = crossvalidation_enabled #Whether to furhter split the dataset into folds or not
        

        """
        Filtering.
        """
        self.min_tokens_per_sentence_treshold = 5 #Minimum number of tokens per sentence. Sentences with less tokens are dropped.
        self.single_entity_class = single_entity_class #Whether to use a single entity class or not
        

    def load_tml_files(self, input_filepaths: List[str]) -> List[str]:
        """
        Loads the tml files from the specified filepaths.

        Args:
            input_filepaths: List of filepath to the tml files.
        
        Returns:
            A list of strings. Each string represents the contents of a tml file.
        """
        tml_file_contents = list()
        for filepath in input_filepaths:
            if os.path.isfile(filepath) and filepath.endswith(".tml"):
                with open(filepath, "r", encoding="utf-8") as tml_file:
                    original_contents = tml_file.read()
                tml_file_contents += [original_contents.strip()]
        return tml_file_contents
    

    def convert_dataset(self) -> None:
        #List of json lists. Each json list contains the json representation of a document for all documents in the respective dataset.
        json_list_tempelval = []
        json_list_only_timebank = []
        json_list_aquaint = []
        if self.extract_tempeval or (self.extract_timebank and self.extract_aquaint):
            documents_timeml: List[Document] = self.parse_timeml_directory()
            documents_extra: List[Document] = self.parse_extra_directory()
            documents_aquaint: List[Document] = self.parse_aquaint_files()

            json_list_timeml: List[dict] = self.documents_to_json_list(documents_timeml)
            json_list_extra: List[dict] = self.documents_to_json_list(documents_extra)
            json_list_aquaint: List[dict] = self.documents_to_json_list(documents_aquaint)
        elif self.extract_timebank:
            documents_timeml: List[Document] = self.parse_timeml_directory()
            documents_extra: List[Document] = self.parse_extra_directory()

            json_list_timeml: List[dict] = self.documents_to_json_list(documents_timeml)
            json_list_extra: List[dict] = self.documents_to_json_list(documents_extra)
        elif self.extract_aquaint:
            documents_aquaint: List[Document] = self.parse_aquaint_files()
            json_list_aquaint: List[dict] = self.documents_to_json_list(documents_aquaint)
        else:
            raise Exception("No dataset to convert specified!")


        #List of tuples. Each tuple contains the json list, the output directory path and the output file prefix.
        generate_files_list = []

        if self.extract_tempeval:
            output_directory_path_tempeval =  self.output_base_directory_tempeval_path
            json_list_tempelval = json_list_timeml + json_list_extra + json_list_aquaint
            random.shuffle(json_list_tempelval)
            generate_files_list += [(json_list_tempelval, output_directory_path_tempeval, self.tempeval_output_file_prefix)]
        
        if self.extract_timebank:
            output_directory_path_timebank = self.output_base_directory_timebank_path
            json_list_only_timebank = json_list_timeml + json_list_extra
            random.shuffle(json_list_only_timebank)
            generate_files_list += [(json_list_only_timebank, output_directory_path_timebank, self.timebank_output_file_prefix)]

        if self.extract_aquaint:
            output_directory_path_aquaint = self.output_base_directory_aquaint_path
            json_list_only_aquaint = json_list_aquaint + list()
            random.shuffle(json_list_only_aquaint)
            generate_files_list += [(json_list_only_aquaint, output_directory_path_aquaint, self.aquaint_output_file_prefix)]

        for json_list, output_directory_path, output_file_prefix in generate_files_list:
            print(f"Generating files for {output_directory_path}...")
            save_dataset_splits(
                json_list=json_list,
                output_directory_path=output_directory_path,
                train_percent=self.train_percent,
                test_percent=self.test_percent,
                val_percent=self.val_percent,
                output_file_train_suffix=self.output_file_train_suffix,
                output_file_test_suffix=self.output_file_test_suffix,
                output_file_val_suffix=self.output_file_val_suffix,
                output_file_full_suffix=self.output_file_full_suffix,
                output_file_prefix=output_file_prefix
            )

            if self.crossvalidation_enabled:
                generate_crossvalidation_folds(
                    json_list=json_list,
                    output_dirname=output_directory_path,
                    folds=self.folds,
                    output_file_train_suffix=self.output_file_train_suffix,
                    output_file_val_suffix=self.output_file_val_suffix,
                    output_file_test_suffix=self.output_file_test_suffix,
                    output_file_prefix=output_file_prefix
                )
            print(f"Completed generating files for {output_directory_path}!\n")
        
        print("\nConversion complete! Have a great day!\n")
        print("Final event classes used for labeling:")
        pprint.pprint(self.event_classes)
        print()
        print("Final relation classes used for labeling:")
        pprint.pprint(self.relation_classes)


    def documents_to_json_list(self, documents: List[Document]) -> List[dict]: 
        sentence_json_list = []
        error_filepaths = Counter()
        for document in documents:
            for sentence in document.sentences:
                drop_sentence_incomplete_information = False
                drop_sentence_incomplete_information = drop_sentence_incomplete_information or "<TIMEX" in sentence.text
                drop_sentence_too_short = False

                text = sentence.text
                tokens = sentence.tokens
                intra_sentence_timex3 = sentence.timex3_in_sentence
                intra_sentence_events = sentence.events_in_sentence
                intra_sentence_relations = sentence.entity_relations

                if len(tokens) < self.min_tokens_per_sentence_treshold:
                    drop_sentence_too_short = True
                    continue

                temporal_entities = []
                for intra_sentence_timex3_object in intra_sentence_timex3:
                    temporal_type = intra_sentence_timex3_object.timex3_type.lower()
                    temporal_span_start = intra_sentence_timex3_object.position_in_sentence_start
                    temporal_span_end = intra_sentence_timex3_object.position_in_sentence_end
                    temporal_text = intra_sentence_timex3_object.text
                    temporal_offset = [i for i in range(temporal_span_start, temporal_span_end + 1)] if temporal_span_start != None and temporal_span_start >= 0 else []
                    drop_sentence_incomplete_information = drop_sentence_incomplete_information or temporal_offset == []

                    entity = {
                        "type": self.temporal_classes_dictionary[temporal_type],
                        "offset": temporal_offset,
                        "text": temporal_text
                    }
                    temporal_entities += [entity]

                events = []
                for intra_sentence_event_object in intra_sentence_events:
                    event_type = intra_sentence_event_object.event_class.lower() if intra_sentence_event_object.event_class else "empty" 
                    self.event_classes[event_type] += 1
                    event_span_start = intra_sentence_event_object.position_in_sentence_start
                    event_span_end = intra_sentence_event_object.position_in_sentence_end
                    event_text = intra_sentence_event_object.text
                    event_offset = [i for i in range(event_span_start, event_span_end + 1)] if event_span_start != None and event_span_start >= 0 else []
                    drop_sentence_incomplete_information = drop_sentence_incomplete_information or event_offset == []

                    event = {
                        "type": self.event_classes_dictionary[event_type],
                        "offset": event_offset,
                        "text": event_text,
                    }
                    events += [event]

                relations = []
                for intra_sentence_relation_object in intra_sentence_relations:
                    relation_type = intra_sentence_relation_object.relation_type.lower().replace("_", "-")
                    self.relation_classes[relation_type] += 1

                    source = intra_sentence_relation_object.source
                    if isinstance(source, TagTimex3):
                        source_text = source.text
                        source_type = self.temporal_classes_dictionary[source.timex3_type.lower().strip()]
                        source_start = source.position_in_sentence_start
                        source_end = source.position_in_sentence_end
                        source_offset = [i for i in range(source_start, source_end + 1)] if source_start != None and source_start >= 0 else []
                        drop_sentence_incomplete_information = drop_sentence_incomplete_information or source_offset == []
                    elif isinstance(source, TagFullEvent):
                        source_text = source.text
                        source_type = source.event_class.lower().strip() if source.event_class else "empty"  #TODO: Maybe change
                        source_type = self.event_classes_dictionary[source_type]
                        source_start = source.position_in_sentence_start
                        source_end = source.position_in_sentence_end
                        source_offset = [i for i in range(source_start, source_end + 1)] if source_start != None and source_start >= 0 else []
                        drop_sentence_incomplete_information = drop_sentence_incomplete_information or source_offset == []
                    else:
                        raise Exception(f"Unknown entity type for source relation.")
                    
                    target = intra_sentence_relation_object.target
                    if isinstance(target, TagTimex3):
                        target_text = target.text
                        target_type = self.temporal_classes_dictionary[target.timex3_type.lower().strip()]
                        target_start = target.position_in_sentence_start
                        target_end = target.position_in_sentence_end
                        target_offset = [i for i in range(target_start, target_end + 1)] if target_start != None and target_start >= 0 else []
                        drop_sentence_incomplete_information = drop_sentence_incomplete_information or target_offset == []
                    elif isinstance(target, TagFullEvent):
                        target_text = target.text
                        target_type = target.event_class.lower() if target.event_class else "empty"
                        target_type = self.event_classes_dictionary[target_type]
                        target_start = target.position_in_sentence_start
                        target_end = target.position_in_sentence_end
                        target_offset = [i for i in range(target_start, target_end + 1)] if target_start != None and target_start >= 0 else []
                        drop_sentence_incomplete_information = drop_sentence_incomplete_information or target_offset == []
                    else:
                        raise Exception(f"Unknown entity type for target relation.")

                    relation = {
                        "type": relation_type,
                        "args": [
                            {
                                "type": source_type,
                                "offset": source_offset,
                                "text": source_text
                            },
                            {
                                "type": target_type,
                                "offset": target_offset,
                                "text": target_text
                            }
                        ]
                    }
                    relations += [relation]

                self.extract_temporal_expressions
                self.extract_events
                
                if self.extract_temporal_relations:
                    sentence_json = {
                        "text": text,
                        "tokens": tokens,
                        "entity": temporal_entities + events,
                        "relation": relations
                    }
                elif self.extract_temporal_expressions and self.extract_events:
                    sentence_json = {
                        "text": text,
                        "tokens": tokens,
                        "entity": temporal_entities + events,
                    }
                elif self.extract_temporal_expressions and not self.extract_events:
                    sentence_json = {
                        "text": text,
                        "tokens": tokens,
                        "entity": temporal_entities,
                    }
                elif self.extract_events and not self.extract_temporal_expressions:
                    sentence_json = {
                        "text": text,
                        "tokens": tokens,
                        "entity": events,
                    }
                else:
                    raise Exception("Unknown error while converting document to json.")


                if not drop_sentence_incomplete_information and not drop_sentence_too_short:
                    sentence_json_list += [sentence_json]
                elif drop_sentence_incomplete_information:
                    error_filepaths["incomplete_" + document.filepath] += 1
                elif drop_sentence_too_short:
                    error_filepaths["too_short_" + document.filepath] += 1
                else:
                    raise Exception("Unknown error while converting document to json.")
                
        print("The following filepaths contain sentences that were dropped due to missing offsets:")
        pprint.pprint(error_filepaths)
        return sentence_json_list


    def _generate_document_from_file(self, tml_filepath: str, tml_full_file_contents: str, article_content_with_tags: str, article_content_without_tags: str, split_sentences: List[str]) -> Document:
        makeinstance_objects = extract_makeinstance_objects(tml_full_file_contents)
        tlink_objects = extract_tlink_objects(tml_full_file_contents)
        event_objects = extract_event_objects(tml_full_file_contents)
        timex3_objects = extract_timex3_objects(tml_full_file_contents)
        meta_document_time = extract_meta_document_time(timex3_objects)
        signal_objects = extract_signal_objects(tml_full_file_contents)
        full_event_objects = extract_full_event_objects(tml_full_file_contents)

        if len(makeinstance_objects) != len(event_objects) != len(full_event_objects):
            print(f"Warning: Number of makeinstance objects ({len(makeinstance_objects)}), event objects ({len(event_objects)}) and full event objects ({len(full_event_objects)}) are not the same size in file:\n{tml_filepath}.")

        document_sentences = generate_sentence_objects(
            split_sentences, 
            makeinstance_objects,
            tlink_objects,
            meta_document_time
        )

        document = create_tempeval_document(
            filepath=tml_filepath,
            filecontents=tml_full_file_contents,
            article_contents=article_content_with_tags,
            clean_text=article_content_without_tags,
            sentences=document_sentences,
            meta_time=meta_document_time,
            full_event_objects=full_event_objects,
            timex3_objects=timex3_objects,
            tlink_objects=tlink_objects,
            makeinstance_objects=makeinstance_objects,
            signal_objects=signal_objects
        )

        return document


    def parse_extra_directory(self) -> List[Document]:
        """
        The 'extra' directory is part of the TimeBank dataset. Its files are inconsistently 
        tagged and therefore require extraction rules, that take the differences into account.
        Nevertheless, the files still have a common structure, which allows for common 
        extraction rules.

        Method parses the 'extra' directory and returns a document per file.

        Args:
            tml_filenames: List of tml filenames.
            tml_filecontents: List of tml file contents.

        Returns:
            A list of documents. Each document represents a tml file.
        """
        print("Parsing TimeBank Extra directory...")
        assert len(self.timebank_tml_filepaths_extra) == len(self.tml_file_contents_extra)
        documents: List[Document] = []
        for tml_filepath, tml_full_file_contents in zip(self.timebank_tml_filepaths_extra ,self.tml_file_contents_extra):
            article_content_with_tags = extract_article_content_timebank_extra(tml_full_file_contents)
            article_content_without_tags = strip_article_content_timebank_extra(article_content_with_tags)
            split_sentences = split_sentences_timebank_extra(article_content_with_tags)

            document = self._generate_document_from_file(tml_filepath, tml_full_file_contents, article_content_with_tags, article_content_without_tags, split_sentences)
            documents += [document]
        print("Completed parsing TimeBank Extra directory!")

        return documents


    def parse_timeml_directory(self) -> List[Document]:
        """
        The 'timeml' directory is part of the TimeBank dataset. The files have a common structure.
        A challenge is that the metadata isn't properly split from the main contents of the file.
        The extraction rules need to take this into account.

        Method parses the 'timeml' directory and returns a document per file.

        Args:
            tml_filenames: List of tml filenames.
            tml_filecontents: List of tml file contents.

        Returns:
            A list of documents. Each document represents a tml file.
        """
        print("Parsing TimeBank TimeMl directory...")
        assert len(self.timebank_tml_filepaths_timeml) == len(self.tml_file_contents_timeml)
        documents: List[Document] = []
        for tml_filepath, tml_full_file_contents in zip(self.timebank_tml_filepaths_timeml ,self.tml_file_contents_timeml):
            article_content_with_tags = extract_article_content_timebank_timeml(tml_full_file_contents)
            article_content_without_tags = strip_article_content_timebank_timeml(article_content_with_tags)
            split_sentences = split_sentences_timebank_timeml(article_content_with_tags)

            document = self._generate_document_from_file(tml_filepath, tml_full_file_contents, article_content_with_tags, article_content_without_tags, split_sentences)
            documents += [document]
        print("Completed parsing TimeBank TimeMl directory!")

        return documents


    def parse_aquaint_files(self) -> List[dict]:
        print("Parsing Aquaint directory...")
        assert len(self.aquaint_tml_filepaths) == len(self.tml_file_contents_aquaint)
        documents: List[Document] = []
        for tml_filepath, tml_full_file_contents in zip(self.aquaint_tml_filepaths ,self.tml_file_contents_aquaint):
            article_content_with_tags = extract_article_content_aquaint(tml_full_file_contents)
            article_content_without_tags = strip_article_content_aquaint(article_content_with_tags)
            split_sentences = split_sentences_aquaint(article_content_with_tags)

            document = self._generate_document_from_file(tml_filepath, tml_full_file_contents, article_content_with_tags, article_content_without_tags, split_sentences)
            documents += [document]
        print("Completed parsing Aquaint directory!")

        return documents



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filepath_timebank_timeml",
        "-itt",
        type = str,
        default = "../../original_datasets/timebank/data/timeml",
        help = "Path to the 'timeml' directory of the original TimeBank dataset. This directory contains multiple TML files.",
    )

    parser.add_argument(
        "--input_filepath_timebank_extra",
        "-ite",
        type = str,
        default = "../../original_datasets/timebank/data/extra",
        help = "Path to the 'extra' directory of the original TimeBank dataset. This directory contains multiple TML files.",
    )

    parser.add_argument(
        "--input_filepath_aquaint",
        "-ia",
        type = str,
        default = "../../original_datasets/aquaint",
        help = "Path to the original AQUAINT dataset. This directory contains multiple TML files.",
    )

    parser.add_argument(
        "--output_directory_tempeval",
        "-ote",
        type = str,
        default = "../../relation/my_converted_datasets/tempeval",
        help = "The directory for the newly converted dataset files."
    )

    parser.add_argument(
        "--output_directory_timebank",
        "-oti",
        type = str,
        default = "../../relation/my_converted_datasets/timebank",
        help = "The directory for the newly converted dataset files."
    )

    parser.add_argument(
        "--output_directory_aquaint",
        "-oa",
        type = str,
        default = "../../relation/my_converted_datasets/aquaint",
        help = "The directory for the newly converted dataset files."
    )

    parser.add_argument(
        "--extract_tempeval",
        "-ete",
        action = "store_true",
        help = "Wether to extract TempEval dataset or not."
    )

    parser.add_argument(
        "--extract_timebank",
        "-eti",
        action = "store_true",
        help = "Wether to extract TimeBank dataset or not."
    )

    parser.add_argument(
        "--extract_aquaint",
        "-ea",
        action = "store_true",
        help = "Wether to extract AQUAINT dataset or not."
    )

    parser.add_argument(
        "--single_class",
        "-s",
        action = "store_true",
        help = "Wether to have the four timex3 temporal classes or only a single generic one."
    )

    parser.add_argument(
        "--extract_tempexp",
        "-ext",
        action = "store_true",
        help = "Wether to extract temporal expressions or not."
    )

    parser.add_argument(
        "--extract_events",
        "-exe",
        action = "store_true",
        help = "Wether to extract events or not."
    )

    parser.add_argument(
        "--extract_relations",
        "-exr",
        action = "store_true",
        help = "Wether to extract relations or not."
    )

    parser.add_argument(
        "--crossvalidation",
        "-c",
        action = "store_true",
        help = "Wether to generate crossvalidation folds or not."
    )

    parser.add_argument(
        "--folds",
        "-f",
        type = int,
        default = 10,
        help = "Number of crossvalidation folds."
    )
    args = parser.parse_args()


    timebank_directory_path_extra = args.input_filepath_timebank_extra
    timebank_filepaths_extra = [os.path.join(timebank_directory_path_extra, f) for f in os.listdir(timebank_directory_path_extra) if os.path.isfile(os.path.join(timebank_directory_path_extra, f)) and f.endswith(".tml")]
    timebank_filepaths_extra.sort()

    timebank_directory_path_timeml = args.input_filepath_timebank_timeml
    timebank_filepaths_timeml = [os.path.join(timebank_directory_path_timeml, f) for f in os.listdir(timebank_directory_path_timeml) if os.path.isfile(os.path.join(timebank_directory_path_timeml, f)) and f.endswith(".tml")]
    timebank_filepaths_timeml.sort()

    aquaint_directory_path = args.input_filepath_aquaint
    aquaint_filepaths = [os.path.join(aquaint_directory_path, f) for f in os.listdir(aquaint_directory_path) if os.path.isfile(os.path.join(aquaint_directory_path, f)) and f.endswith(".tml")]
    aquaint_filepaths.sort()


    converter = TempevalDatasetConverter(
        input_filepaths_timebank_extra=timebank_filepaths_extra,
        input_filepaths_timebank_timeml=timebank_filepaths_timeml,
        input_filepaths_aquaint=aquaint_filepaths,
        extract_tempeval=args.extract_tempeval,
        extract_timebank=args.extract_timebank,
        extract_aquaint=args.extract_aquaint,
        output_base_directory_tempeval_path=args.output_directory_tempeval,
        output_base_directory_timebank_path=args.output_directory_timebank,
        output_base_directory_aquaint_path=args.output_directory_aquaint,
        single_entity_class=args.single_class,
        extract_temporal_expressions=args.extract_tempexp,
        extract_events=args.extract_events,
        extract_temporal_relations=args.extract_relations,
        crossvalidation_enabled=args.crossvalidation,
        folds=args.folds
    )
    converter.convert_dataset()