#!/usr/bin/env python
# -*- coding:utf-8 -*-
import jsonlines
import re
import random
from typing import List
from preprocessing_utils import find_sublist_in_list
from preprocessing_file_saver import generate_crossvalidation_folds, save_dataset_splits
from preprocessing_utils import DatasetNltkTokenizer
import argparse
import sys
import os
import pprint

class FullPateDatasetConverter:
    """
    This class converts the FULL PATE dataset to the jsonline format. It is assumed that each of the original datasets Snips and Pate have
    already been converter by their respective converter classes. 
    The class takes into account all the details of the PATE dataset. It also implements crossvalidation and saves the dataset in multiple copies.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(self, input_filepaths: List[str], output_directory_path: str, single_entity_class: bool, crossvalidation_enabled: bool = False, folds: int = 10, only_temporal_entities: bool = False) -> None:
        self.input_filepaths = input_filepaths #List of input filepaths
        self.output_directory_path = output_directory_path #Where to save the converted dataset
        self.crossvalidation_enabled = crossvalidation_enabled #Whether to split the dataset into folds or not
        self.folds = folds #If crossvalidation is enabled, how many folds to create
        self.single_entity_class = single_entity_class #Whether to use a single entity class or not
        self.only_temporal_entities = only_temporal_entities #Whether to only keep temporal entities or not

        #Split percentages of the dataset
        self.train_percent = 0.8
        self.test_percent = 0.1
        self.val_percent = 0.1

        self.crossvalidation_test_percent = 0.1 #Full dataset is permanently split into 90-10, crossvalidation is applied to the 90% part

        #Output file names
        self.output_file_prefix = "fullpate"
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
        self.word_tokenizer = DatasetNltkTokenizer()
        
        #Load dataset
        self.dataset = self.load_dataset(self.input_filepaths)
        

    def load_dataset(self, dataset_filepaths: List[str]) -> List[dict]:
        """
        Loads the dataset from the specified filepaths.

        Args:
            dataset_filepaths: List of filepath to the dataset files.

        Returns:
            A list of dictionaries. Each dictionary a dataset entry.
        """
        dataset = list()
        for dataset_filepath in dataset_filepaths:
            with jsonlines.open(dataset_filepath, 'r') as file:
                dataset += [obj for obj in file]
        return dataset
    

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
        if self.only_temporal_entities:
            for json_element in json_list:
                entities = json_element["entity"]
                filtered_entities = entities.copy()
                for entity in entities:
                    if entity["type"] not in self.classes_dictionary.values():
                        filtered_entities.remove(entity)
                json_element["entity"] = filtered_entities
        print(f"Loaded input dataset in memory and created json structure.\n")

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


    def extract_entities(self, entities, original_text, original_text_tokens) -> List[dict]:
        """
        Extracts the timex3 entities from the dataset entry.

        Args:
            entities: The entities of the dataset entry.
            original_text: The original text of the dataset entry.
            original_text_tokens: The tokenized original text of the dataset entry.

        Returns:
            A list of dictionaries. Each dictionary represents a timex3 entity.
        """
        extracted_entities = list()
        for entity in entities:
            timex_entities = entity["TIMEX3"]
            for timex3 in timex_entities:
                if timex3["expression"] == "" and timex3["beginPoint"] == "" and timex3["endPoint"] == "":
                    continue
                timex3_type = timex3["type"].strip().lower()
                timex3_type = self.classes_dictionary[timex3_type]
                duration_text = ""
                if timex3["type"] == "DURATION" and timex3["expression"].strip() == "":
                    duration_regex = self.pate_dataset_duration_span_regex_extractor(timex3["beginPoint"], timex3["endPoint"], entities)
                    pattern_matches = re.findall(duration_regex, original_text)
                    duration_text = pattern_matches[0] if len(pattern_matches) > 0 else ""
                    duration_text = duration_text.strip()
                    if duration_text == "": raise Exception("Duration text is empty.")
                
                text = timex3["expression"].strip() if duration_text == "" else duration_text
                tokens = self.word_tokenizer.tokenize(text)
                entity_start_index, entity_end_index = find_sublist_in_list(original_text_tokens, tokens)
                extracted_entities += [{
                    "text": text,
                    "type": timex3_type,
                    "start": entity_start_index,
                    "end": entity_end_index
                }]
        return extracted_entities


    def pate_dataset_duration_span_regex_extractor(self, beginPoint: str, endPoint: str, entities: List[dict]) -> str:
        """
        Extracts the span regex for a duration timex3 tag. The regex is used to find the duration timex3 tag in the sentence.

        Args:
            beginPoint: The begin point of the duration timex3 tag.
            endPoint: The end point of the duration timex3 tag.
            entities: The entities of the dataset entry.

        Returns:
            A string containing the span regex for the duration timex3 tag.
        """
        span_regex = ".*"
        begin_regex = ""
        end_regex = ""

        beginFound = False
        endFound = False
        for entity in entities:
            timex_entities = entity["TIMEX3"]
            for timex3 in timex_entities:
                if timex3["beginPoint"] == beginPoint:
                    for entity_compare in entities:
                        for timex3_compare in entity_compare["TIMEX3"]:
                            if timex3_compare["tid"] == beginPoint:
                                begin_regex = timex3_compare["expression"]
                                beginFound = True
                if timex3["endPoint"] == endPoint:
                    for entity_compare in entities:
                        for timex3_compare in entity_compare["TIMEX3"]:
                            if timex3_compare["tid"] == endPoint:
                                end_regex = timex3_compare["expression"]
                                endFound = True
        return (begin_regex + span_regex + end_regex) if beginFound and endFound else "NO DURATION FOUND"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filepath_snips",
        "-ifs",
        type = str,
        default = "../entity/jsonlines/snips_multi/snips-full.jsonlines",
        help = "Relative path to converted Snips dataset in jsonlines format.",
    )

    parser.add_argument(
        "--input_filepath_pate",
        "-ifp",
        type = str,
        default = "../entity/jsonlines/pate_multi/pate-full.jsonlines",
        help = "Relative path to converted PATE dataset in jsonlines format.",
    )

    parser.add_argument(
        "--output_directory",
        "-o",
        type = str,
        default = "../entity/my_datasets/jsonlines/fullpate_multi",
        help = "The directory for the newly converted dataset files."
    )

    parser.add_argument(
        "--single_class",
        "-s",
        action = "store_true",
        help = "Wether to have the four timex3 temporal classes or only a single generic one."
    )

    parser.add_argument(
        "--multi_class",
        "-m",
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

    parser.add_argument(
        "--only_temporal",
        "-ot",
        action = "store_true",
        default = True,
        help = "Wether to contain only temporal classes or not. The Snips dataset contains other entitiy classes than the four temporal timex3 classes."
    )
    args = parser.parse_args()


    #Validate input
    is_error: bool = False
    if args.multi_class and args.single_class:
        is_error = True

    if not args.multi_class and not args.single_class:
        is_error = True

    if args.input_filepath_snips is None:
        is_error = True

    if args.input_filepath_pate is None:
        is_error = True

    if isinstance(args.input_filepath_snips, str) and not args.input_filepath_snips.lower().endswith(".jsonlines"):
        is_error = True

    if isinstance(args.input_filepath_pate, str) and not args.input_filepath_pate.lower().endswith(".jsonlines"):
        is_error = True

    if args.output_directory is None:
        is_error = True

    if is_error:
        print("Problem with input arguments.")
        sys.exit()


    print(f"Loading Fullpate conversion script...")
    print(f"Following arguments were passed:")
    print(f"Snips dataset input filepath:   {args.input_filepath_snips} => {type(args.input_filepath_snips)}")
    print(f"Pate dataset input filepath:    {args.input_filepath_pate} => {type(args.input_filepath_pate)}")
    print(f"Output directory:               {args.output_directory} => {type(args.output_directory)}")
    print(f"Single class only:              {args.single_class} => {type(args.single_class)}")
    print(f"Multi class only:               {args.multi_class} => {type(args.single_class)}")
    print(f"Crossvalidation enabled:        {args.crossvalidation} => {type(args.crossvalidation)}")
    print(f"Number of folds:                {args.folds} => {type(args.folds)}")
    print(f"Temporal classes only:          {args.only_temporal} => {type(args.only_temporal)}")

    input_filepaths: List[str] = list()
    input_filepaths += [os.path.abspath(args.input_filepath_snips)]
    input_filepaths += [os.path.abspath(args.input_filepath_pate)]
    pprint.pprint(f"Input directory paths:          {input_filepaths}")


    output_dirpath: str = args.output_directory
    single_entity_class: bool = args.single_class
    crossvalidation_enabled: bool = args.crossvalidation
    folds: int = args.folds
    only_temporal_entities: bool = args.only_temporal


    print()
    if not os.path.exists(output_dirpath):
        print(f"Output directory does not exist. Creating directory '{os.path.abspath(output_dirpath)}'.\n")
        os.makedirs(os.path.abspath(output_dirpath))

    print("Running converter...")
    converter = FullPateDatasetConverter(
        input_filepaths = input_filepaths,
        output_directory_path = output_dirpath,
        single_entity_class = single_entity_class,
        crossvalidation_enabled = crossvalidation_enabled,
        folds = folds,
        only_temporal_entities = only_temporal_entities
    )
    converter.convert_dataset()