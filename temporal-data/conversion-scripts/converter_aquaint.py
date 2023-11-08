#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import random
import nltk
import os
from typing import List, Tuple
from preprocessing_utils import find_sublist_in_list
from preprocessing_file_saver import generate_crossvalidation_folds, save_dataset_splits
from preprocessing_utils import DatasetNltkTokenizer

class AquaintDatasetConverter:
    """
    This class converts the AQUAINT dataset to the jsonline format. The class takes into account the details of the AQUAINT dataset.
    It also implements crossvalidation and saves the dataset in multiple copies.

    The dataset is split into two distinct directories with TML files that have different annotation styles. Therefore these
    files have to be handeled seperately.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(self, input_filepaths_extra: List[str], input_filepaths_timeml: List[str], output_directory_path: str, single_entity_class: bool, crossvalidation_enabled: bool = False, folds: int = 10) -> None:
        #Regex patterns to extract the contents of the tml files
        self.contents_regex_pattern_s = "<s>.*?</s>"
        self.contents_regex_pattern_timex3 = r"<TIMEX3[^>]*>.*?</TIMEX3>"
        self.contents_regex_pattern_timeml = r"<TimeML[^>]*>.*?</TimeML>"
        self.contents_regex_pattern_text = r"<TEXT[^>]*>.*?</TEXT>"

        self.tag_regex_pattern_anyxml = "<[^>]+>"
        self.tag_regex_pattern_makeinstance = "<MAKEINSTANCE[^>]+>"
        self.tag_regex_pattern_tlink = "<TLINK[^>]+>"
        self.tag_regex_pattern_slink = "<SLINK[^>]+>"
        self.tag_regex_pattern_alink = "<ALINK[^>]+>"
        self.tag_regex_pattern_timeml_open = "<TimeML[^>]*>"
        self.tag_regex_pattern_timeml_close = "</TimeML[^>]*>"
        self.tag_regex_pattern_xml = "<\?xml[^>]+>"

        #Sentences that contain one of those get dropped
        self.forbidden_strings = [
            "--",
            "@",
            " = ",
            "....."
        ]

        #Drop sentences with less tokens than this
        self.MIN_SIZE_TOKENS_IN_INPUT = 4
        
        self.tml_files_extra_filepaths = input_filepaths_extra #Filepaths of the tml files in the extra folder
        self.tml_files_timeml_filepaths = input_filepaths_timeml #Filepaths of the tml files in the timeml folder

        self.output_directory_path = output_directory_path #Where to save the converted dataset
        self.crossvalidation_enabled = crossvalidation_enabled #Whether to split the dataset into folds or not
        self.folds = folds #If crossvalidation is enabled, how many folds to create
        self.single_entity_class = single_entity_class #Whether to use a single entity class or not

        #Split percentages of the dataset
        self.train_percent = 0.8
        self.test_percent = 0.1
        self.val_percent = 0.1

        #Output file names
        self.output_file_prefix = "aquaint-old"
        self.output_file_ending = ".jsonlines"
        self.output_file_train_suffix = "-train" + self.output_file_ending
        self.output_file_test_suffix = "-test" + self.output_file_ending
        self.output_file_val_suffix = "-val" + self.output_file_ending        
        self.output_file_full_suffix = "-full" + self.output_file_ending

        #Represents all timex3 types: DATE, TIME, DURATION, SET
        if single_entity_class:
            self.classes_dictionary = {
                "date": "tempexp",
                "time": "tempexp",
                "duration": "tempexp",
                "set": "tempexp"
            }
        else:
            self.classes_dictionary = {
                "date": "date",
                "time": "time",
                "duration": "duration",
                "set": "set"
            }
        
        #Load tokenizers
        self.word_tokenizer, self.sentence_tokenizer = self.initiate_tokenizers()

        #Load tml input files
        self.tml_files = self.load_tml_files(self.tml_files_timeml_filepaths)

        self.dataset_aquaint = list()

    def initiate_tokenizers(self) -> Tuple[DatasetNltkTokenizer, nltk.data.load]:
        """
        Initiates the tokenizers.

        Returns:
            A tuple containing the word tokenizer and sentence tokenizer.
        """
        sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        word_tokenizer = DatasetNltkTokenizer()
        return word_tokenizer, sentence_tokenizer

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
                contents = original_contents.replace("\n", " ").strip()
                tml_file_contents += [contents]
        return tml_file_contents
    
    def convert_dataset(self) -> None:
        """
        Converts the dataset into json format and writes it to the filesystem.
        The dataset is saved in multiple copies:
            (1) Full dataset
            (2) Train dataset / Test dataset
            (3) Train dataset / Test dataset for each crossvalidation fold

        Prior to conversion the dataset is shuffled.
        """
        self.dataset_aquaint = self.aquaint_folder_to_jsonlist(
            tml_files=self.tml_files,
            word_tokenizer=self.word_tokenizer,
            sentence_tokenizer=self.sentence_tokenizer
        )

        json_list = self.dataset_aquaint
        print(f"Loaded input dataset in memory and created json structure.\n")
        
        random.shuffle(json_list)

        save_dataset_splits(
            json_list=json_list,
            output_directory_path=self.output_directory_path,
            train_percent=self.train_percent,
            test_percent=self.test_percent,
            val_percent=self.val_percent,
            output_file_train_suffix=self.output_file_train_suffix,
            output_file_test_suffix=self.output_file_test_suffix,
            output_file_val_suffix=self.output_file_val_suffix,
            output_file_full_suffix=self.output_file_full_suffix,
            output_file_prefix=self.output_file_prefix,
        )

        if self.crossvalidation_enabled:
            generate_crossvalidation_folds(
                json_list=json_list,
                output_dirname=self.output_directory_path, 
                folds=self.folds,
                output_file_train_suffix=self.output_file_train_suffix,
                output_file_val_suffix=self.output_file_val_suffix,
                output_file_test_suffix=self.output_file_test_suffix,
                output_file_prefix=self.output_file_prefix,
            )
        
        print("\nConversion complete!")

    def aquaint_folder_to_jsonlist(self, tml_files: List[str], word_tokenizer: DatasetNltkTokenizer, sentence_tokenizer: nltk.data.load) -> List[dict]:
        """
        Converts the tml files into the json format. It extracts the sentences and the temporal information.

        Args:
            tml_files: List of tml files.
            word_tokenizer: The word tokenizer.
            sentence_tokenizer: The sentence tokenizer.

        Returns:
            A list of dictionaries. Each dictionary represents a sentence with its temporal information.
        """
        patterns = [
            ("\s{2,}", " "), #Changes whitespace gaps larger than 1 into exactly 1
            (self.tag_regex_pattern_xml, ""),
            (self.tag_regex_pattern_timeml_open, ""),
            (self.tag_regex_pattern_timeml_close, ""),
            (self.tag_regex_pattern_makeinstance, ""),
            (self.tag_regex_pattern_tlink, ""),
            (self.tag_regex_pattern_slink, ""),
            (self.tag_regex_pattern_alink, ""),
            ("-", " - ")
        ]

        json_dictionaries = []
        for tml_file in tml_files:
            contents = tml_file
            contents = contents.replace("\n", "")
            
            for pattern, replacement in patterns:
                contents = re.sub(pattern, replacement, contents)

            #Get contents of <TEXT> tag
            text_contents = re.findall(self.contents_regex_pattern_text, contents)
            if len(text_contents) > 0:
                contents = text_contents[0] 
                contents = contents.replace("<TEXT>", "").replace("</TEXT>", "").strip()
            else:
                contents = ""
            
            #Split sentences and drop bad ones
            sentences = sentence_tokenizer.tokenize(contents)
            sents_to_remove = []
            for sentence in sentences:
                do_cont = True
                for forbidden_string in self.forbidden_strings:
                    if do_cont and forbidden_string in sentence:
                        sents_to_remove += [sentence]
                        do_cont = False
            for sentence in sents_to_remove:
                sentences.remove(sentence)
            
            # Structure information in a sentence
            timex3_tags = []
            for sentence in sentences:
                xml_tags_removed = re.sub(self.tag_regex_pattern_anyxml, "", sentence)
                tokens = word_tokenizer.tokenize(xml_tags_removed)
                if len(tokens) < self.MIN_SIZE_TOKENS_IN_INPUT:
                    continue

                # Extract entities from time tags
                entities = []
                timex3_tags = re.findall(self.contents_regex_pattern_timex3, sentence)
                for timex3_tag in timex3_tags:
                    time_content = (re.sub(self.tag_regex_pattern_anyxml, "", timex3_tag)).strip()
                    entity_tokens = []
                    if time_content != "":
                        entity_tokens = word_tokenizer.tokenize(time_content)
                    entity_start_index, entity_end_index = find_sublist_in_list(tokens, entity_tokens)

                    timex3_type = ""
                    timex3_type_match = re.search(r'type="[^"]+"', timex3_tag)
                    if timex3_type_match:
                        timex3_type = timex3_type_match.group()
                        timex3_type = timex3_type.replace('type="', "").replace('"', "")
                        timex3_type = timex3_type.strip().lower()
                        timex3_type = self.classes_dictionary[timex3_type]

                    entity_dict = {
                        "text": time_content,
                        "type": timex3_type,
                        "start": entity_start_index,
                        "end": entity_end_index
                    }
                    entities += [entity_dict]

                #Check if entity start or end is -1
                is_corrupted = False
                for entity in entities:
                    if entity["start"] == -1 or entity["end"] == -1:
                        is_corrupted = True

                if not is_corrupted:
                    json_dictionaries += [{
                        "text": xml_tags_removed, 
                        "tokens": tokens, 
                        "entity": entities
                    }]
        return json_dictionaries

if __name__ == "__main__":
    aquaint_directory_extra = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/aquaint"
    aquaint_files_extra = [os.path.join(aquaint_directory_extra, f) for f in os.listdir(aquaint_directory_extra) if os.path.isfile(os.path.join(aquaint_directory_extra, f)) and f.endswith(".tml")]
    aquaint_files_extra.sort()

    aquaint_directory_timeml = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/aquaint"
    aquaint_files_timeml = [os.path.join(aquaint_directory_timeml, f) for f in os.listdir(aquaint_directory_timeml) if os.path.isfile(os.path.join(aquaint_directory_timeml, f)) and f.endswith(".tml")]
    aquaint_files_timeml.sort()

    converter_inputs = [
        {
            "input_filepaths_extra": aquaint_files_extra,
            "input_filepaths_timeml": aquaint_files_timeml,
            "output_filepath": "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted/aquaint-old_single",
            "single_entity_class": True,
            "crossvalidation_enabled": True,
            "folds": 10,
            "printmessage": "Converting dataset:\nSingle=True"
        },
        {
            "input_filepaths_extra": aquaint_files_extra,
            "input_filepaths_timeml": aquaint_files_timeml,
            "output_filepath": "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted/aquaint-old_multi",
            "single_entity_class": False,
            "crossvalidation_enabled": True,
            "folds": 10,
            "printmessage": "Converting dataset:\nSingle=False"
        }
    ]

    for converter_input in converter_inputs:
        input_filepaths_extra: List[str] = converter_input["input_filepaths_extra"]
        input_filepaths_timeml: List[str] = converter_input["input_filepaths_timeml"]
        output_filepath: str = converter_input["output_filepath"]
        single_entity_class: bool = converter_input["single_entity_class"]
        crossvalidation_enabled: bool = converter_input["crossvalidation_enabled"]
        folds: int = converter_input["folds"]
        printmessage: str = converter_input["printmessage"]
        print(printmessage)
        converter = AquaintDatasetConverter(input_filepaths_extra, input_filepaths_timeml, output_filepath, single_entity_class, crossvalidation_enabled, folds)
        converter.convert_dataset()
        print("\n" + "-" * 100 + "\n")