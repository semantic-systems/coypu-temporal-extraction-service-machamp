#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import random
import os
from typing import List, Tuple
import nltk
from preprocessing_utils import find_sublist_in_list
from preprocessing_file_saver import generate_crossvalidation_folds, save_dataset_splits
from preprocessing_utils import DatasetNltkTokenizer

class WikiwarsTaggedDatasetConverter:
    """
    This class converts the WikiwarsTagged dataset to the jsonline format. The class takes into account the details of the WikiwarsTagged dataset.
    It also implements crossvalidation and saves the dataset in multiple copies.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(self, input_filepaths: List[str], output_directory_path: str, single_entity_class: bool, crossvalidation_enabled: bool = False, folds: int = 10) -> None:
        self.contents_regex_pattern_text = r"<TEXT>.*?</TEXT>"
        self.contents_regex_pattern_timex3 = r"<TIMEX3[^>]*>.*?</TIMEX3>"

        self.tag_regex_pattern_anytml = "<[^>]+>"
        self.tag_regex_pattern_text_open = "<TEXT>"
        self.tag_regex_pattern_text_close = "</TEXT>"

        #Sentences that contain one of those get dropped
        self.forbidden_strings = [
            "--",
            "@",
            " = ",
            "....."
        ]
        
        self.MIN_TOKENS_IN_SENTENCE_SIZE = 4 #Drop sentences with less tokens than this
        
        self.input_filepaths = input_filepaths #List of input filepaths
        self.output_directory_path = output_directory_path #Where to save the converted dataset
        self.crossvalidation_enabled = crossvalidation_enabled #Whether to split the dataset into folds or not
        self.folds = folds #If crossvalidation is enabled, how many folds to create
        self.single_entity_class = single_entity_class #Whether to use a single entity class or not

        #Split percentages of the dataset
        self.train_percent = 0.8
        self.test_percent = 0.1
        self.val_percent = 0.1

        #Output file names
        self.output_file_prefix = "wikiwars-tagged"
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
        
        #Load word tokenizer
        self.word_tokenizer, self.sentence_tokenizer = self.initiate_tokenizers()

        #Load texts from tml files
        self.tml_files_texts = self.load_tml_files(self.input_filepaths) 
        
        #Load dataset
        self.dataset = self.create_jsonlist_dataset(self.tml_files_texts, self.word_tokenizer, self.sentence_tokenizer)

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
                contents = original_contents.strip()
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
        
        json_list = self.dataset
        random.shuffle(json_list)
        print(f"Loaded input dataset in memory and created json structure.\n")

        #Saves copies that are more human readable
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

    def create_jsonlist_dataset(self, tml_files, word_tokenizer, sentence_tokenizer):
        patterns = [
            (self.tag_regex_pattern_text_open, ""),
            (self.tag_regex_pattern_text_close, ""),
            ("%", " % "),
            ("/", " / "),
            ("-", " - ")
        ]

        json_dictionaries = []
        for tml_file in tml_files:
            contents = tml_file

            #Extract relevant text
            match = re.search(self.contents_regex_pattern_text, contents, re.DOTALL)

            if match:
                contents = match.group()
            else:
                raise Exception("Couldn't find text tags! Breaking method...")
            
            #Extract paragraphs
            contents = contents.replace("\r", "")
            paragraphs = contents.split("\n\n")

            sentences = []
            for paragraph in paragraphs:
                paragraph = re.sub("\s{2,}", " ", paragraph)
                if len(paragraph) == 0:
                    continue
                paragraph = paragraph.replace("\n", " ")
                sentences += sentence_tokenizer.tokenize(paragraph)
            
            #Drop bad sentences
            sents_to_remove = []
            for sentence in sentences:
                do_cont = True
                for forbidden_string in self.forbidden_strings:
                    if do_cont and forbidden_string in sentence:
                        sents_to_remove += [sentence]
                        do_cont = False
            for sentence in sents_to_remove:
                sentences.remove(sentence)

            #Structure information in a sentence
            timex3_tags = []
            for sentence in sentences:
                #Extract text and tokenize
                tml_tags_removed = re.sub(self.tag_regex_pattern_anytml, "", sentence)
                for pattern, replacement in patterns:
                    tml_tags_removed = re.sub(pattern, replacement, tml_tags_removed, re.DOTALL)
                tokens = word_tokenizer.tokenize(tml_tags_removed)
                if len(tokens) < self.MIN_TOKENS_IN_SENTENCE_SIZE:
                    continue
                tml_tags_removed = re.sub("\s{2,}", " ", tml_tags_removed)

                #Extract entities from time tags
                entities = []
                timex3_tags = re.findall(self.contents_regex_pattern_timex3, sentence)
                contains_broken_index = False
                for timex3_tag in timex3_tags:
                    time_content = (re.sub(self.tag_regex_pattern_anytml, "", timex3_tag))
                    for pattern, replacement in patterns:
                        time_content = re.sub(pattern, replacement, time_content, re.DOTALL)
                    entity_tokens = []
                    entity_tokens = word_tokenizer.tokenize(time_content)
                    entity_start_index, entity_end_index = find_sublist_in_list(tokens, entity_tokens)
                    if entity_start_index == -1 or entity_end_index == -1: contains_broken_index = True
                    timex3_type = (re.search("type=\"(.*?)\"", timex3_tag).group(1)).lower()
                    entity_dict = {
                        "text": time_content,
                        "type": self.classes_dictionary[timex3_type],
                        "start": entity_start_index,
                        "end": entity_end_index
                    }
                    entities += [entity_dict]
                if not contains_broken_index:
                    json_dictionaries += [{
                        "text": tml_tags_removed, 
                        "tokens": tokens, 
                        "entity": entities
                    }]
        return json_dictionaries

if __name__ == "__main__":
    wikiwars_directory = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/wikiwars-tagged"
    wikiwars_tml_filepaths = [os.path.join(wikiwars_directory, f) for f in os.listdir(wikiwars_directory) if os.path.isfile(os.path.join(wikiwars_directory, f)) and f.endswith(".tml")]
    wikiwars_tml_filepaths.sort()

    converter_inputs = [
        {
            "input_filepaths": wikiwars_tml_filepaths,
            "output_filepath": "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted/wikiwars-tagged_single",
            "single_entity_class": True,
            "crossvalidation_enabled": True,
            "folds": 10,
            "printmessage": "Converting dataset:\nSingle=True"
        },

        {
            "input_filepaths": wikiwars_tml_filepaths,
            "output_filepath": "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted/wikiwars-tagged_multi",
            "single_entity_class": False,
            "crossvalidation_enabled": True,
            "folds": 10,
            "printmessage": "Converting dataset:\nSingle=False"
        }
    ]

    for converter_input in converter_inputs:
        input_filepaths: List[str] = converter_input["input_filepaths"]
        output_filepath: str = converter_input["output_filepath"]
        single_entity_class: bool = converter_input["single_entity_class"]
        crossvalidation_enabled: bool = converter_input["crossvalidation_enabled"]
        folds: int = converter_input["folds"]
        printmessage: str = converter_input["printmessage"]
        print(printmessage)
        converter = WikiwarsTaggedDatasetConverter(input_filepaths, output_filepath, single_entity_class, crossvalidation_enabled, folds)
        converter.convert_dataset()
        print("\n" + "-" * 100 + "\n")