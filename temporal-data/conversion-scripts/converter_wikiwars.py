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
import argparse
import sys
import pprint

class WikiwarsDatasetConverter:
    """
    This class converts the Wikiwars dataset to the jsonline format. The class takes into account the details of the Wikiwars dataset.
    It also implements crossvalidation and saves the dataset in multiple copies.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(self, input_filepaths: List[str], output_directory_path: str, single_entity_class: bool, crossvalidation_enabled: bool = False, folds: int = 10) -> None:
        self.contents_regex_pattern_text = r"<TEXT>.*?</TEXT>"
        self.contents_regex_pattern_timex2 = r"<TIMEX2[^>]*>.*?</TIMEX2>"

        self.tag_regex_pattern_anyxml = "<[^>]+>"
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
        self.output_file_prefix = "wikiwars"
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

        #Load texts from xml files
        self.xml_files_texts = self.load_xml_files(self.input_filepaths) 
        
        #Load dataset
        self.dataset = self.create_jsonlist_dataset(self.xml_files_texts, self.word_tokenizer, self.sentence_tokenizer)

    def initiate_tokenizers(self) -> Tuple[DatasetNltkTokenizer, nltk.data.load]:
        """
        Initiates the tokenizers.

        Returns:
            A tuple containing the word tokenizer and sentence tokenizer.
        """
        sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        word_tokenizer = DatasetNltkTokenizer()
        return word_tokenizer, sentence_tokenizer
    
    def load_xml_files(self, input_filepaths: List[str]) -> List[str]:
        """
        Loads the xml files from the specified filepaths.

        Args:
            input_filepaths: List of filepath to the xml files.
        
        Returns:
            A list of strings. Each string represents the contents of a xml file.
        """
        xml_file_contents = list()
        for filepath in input_filepaths:
            if os.path.isfile(filepath) and filepath.endswith(".xml"):
                with open(filepath, "r", encoding="utf-8") as xml_file:
                    original_contents = xml_file.read()
                contents = original_contents.strip()
                xml_file_contents += [contents]
        return xml_file_contents
    
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

    def create_jsonlist_dataset(self, xml_files, word_tokenizer, sentence_tokenizer):
        patterns = [
            (self.tag_regex_pattern_text_open, ""),
            (self.tag_regex_pattern_text_close, ""),
            ("%", " % "),
            ("/", " / "),
            ("-", " - ")
        ]

        json_dictionaries = []
        for xml_file in xml_files:
            contents = xml_file

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
            timex2_tags = []
            for sentence in sentences:
                #Extract text and tokenize
                xml_tags_removed = re.sub(self.tag_regex_pattern_anyxml, "", sentence)
                for pattern, replacement in patterns:
                    xml_tags_removed = re.sub(pattern, replacement, xml_tags_removed, re.DOTALL)
                tokens = word_tokenizer.tokenize(xml_tags_removed)
                if len(tokens) < self.MIN_TOKENS_IN_SENTENCE_SIZE:
                    continue
                xml_tags_removed = re.sub("\s{2,}", " ", xml_tags_removed)

                #Extract entities from time tags
                entities = []
                timex2_tags = re.findall(self.contents_regex_pattern_timex2, sentence)
                contains_broken_index = False
                for timex2_tag in timex2_tags:
                    time_content = (re.sub(self.tag_regex_pattern_anyxml, "", timex2_tag))
                    for pattern, replacement in patterns:
                        time_content = re.sub(pattern, replacement, time_content, re.DOTALL)
                    entity_tokens = []
                    entity_tokens = word_tokenizer.tokenize(time_content)
                    entity_start_index, entity_end_index = find_sublist_in_list(tokens, entity_tokens)
                    if entity_start_index == -1 or entity_end_index == -1: contains_broken_index = True
                    entity_dict = {
                        "text": time_content,
                        "type": "tempexp",
                        "start": entity_start_index,
                        "end": entity_end_index
                    }
                    entities += [entity_dict]
                if not contains_broken_index:
                    json_dictionaries += [{
                        "text": xml_tags_removed, 
                        "tokens": tokens, 
                        "entity": entities
                    }]
        return json_dictionaries



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_parent_filepath",
        "-i", 
        type = str,
        default = "../original_datasets/wikiwars",
        help = "The original WikiWars dataset consists of many XML files in one directory. It is assumed that all XML files are in the same directory. The filepath is expected as the value for the parameter."
    )

    parser.add_argument(
        "--output_directory",
        "-o",
        type = str,
        default = "../entity/my_datasets/jsonlines/wikiwars_single",
        help = "The directory for the newly converted dataset files."
    )

    parser.add_argument(
        "--single_class",
        "-s",
        action = "store_true",
        help = "Wether to have the four timex3 temporal classes or only a single generic one."
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

    wikiwars_directory = args.input_parent_filepath
    wikiwars_xml_filepaths = [os.path.join(wikiwars_directory, f) for f in os.listdir(wikiwars_directory) if os.path.isfile(os.path.join(wikiwars_directory, f)) and f.endswith(".xml")]
    wikiwars_xml_filepaths.sort()


    #Validate input
    is_error: bool = False
    if (len(wikiwars_xml_filepaths) == 0) or not isinstance(wikiwars_xml_filepaths[0], str) or not wikiwars_xml_filepaths[0].endswith(".xml"):
        is_error = True

    if args.input_parent_filepath is None or args.input_parent_filepath == []:
        is_error = True

    if args.output_directory is None:
        is_error = True

    if is_error:
        print("Problem with input arguments.")
        sys.exit()

    if not args.single_class:
        print("Multi-class conversion is not possible. Use the WikiWars-tagged script on the tagged WikiWars dataset or do a single class conversion.")
        sys.exit()

    
    print(f"Loading WikiWars (original) conversion script...")
    print(f"Note that multi-class conversion is impossible, because the original WikiWars dataset is based on TIMEX2 tags, which do not contain temporal class information.")
    print(f"Following arguments were passed:")
    print(f"WikiWars dataset input filepath:    {args.input_parent_filepath} => {type(args.input_parent_filepath)}")
    print(f"Output directory:                   {args.output_directory} => {type(args.output_directory)}")
    print(f"Crossvalidation enabled:            {args.crossvalidation} => {type(args.crossvalidation)}")
    print(f"Number of folds:                    {args.folds} => {type(args.folds)}")

    print(f"The parent directory contains the following XML files:")
    pprint.pprint(wikiwars_xml_filepaths)


    print()
    if not os.path.exists(args.output_directory):
        print(f"Output directory does not exist. Creating directory '{os.path.abspath(args.output_directory)}'.\n")
        os.makedirs(os.path.abspath(args.output_directory))

    
    converter = WikiwarsDatasetConverter(
        input_filepaths=wikiwars_xml_filepaths,
        output_directory_path=args.output_directory,
        single_entity_class=args.single_class,
        crossvalidation_enabled=args.crossvalidation,
        folds=args.folds
    )
    converter.convert_dataset()