#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re
import random
import os
from typing import List
from preprocessing_utils import find_sublist_in_list
from preprocessing_file_saver import save_dataset_splits, generate_crossvalidation_folds
from preprocessing_utils import DatasetNltkTokenizer

class SnipsDatasetConverter:
    """
    This class converts the Snips dataset (part of PATE) to the jsonline format. The class takes into account the details of the Snips dataset.
    It also implements crossvalidation and saves the dataset in multiple copies.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(self, input_filepaths: List[str], output_directory_path: str, single_entity_class: bool, crossvalidation_enabled: bool = False, folds: int = 10, only_temporal_entities: bool = False) -> None:
        self.SNIPS_TIME_ENTITY_TAG = "timerange"
        
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

        #Output file names
        self.output_file_prefix = "snips"
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
        dataset_instances = list()
        for dataset_filepath in dataset_filepaths:
            with open(dataset_filepath, 'r') as file:
                dataset_instances += [json.load(file)]

        dataset = list()
        for dataset_instance in dataset_instances:
            dataset += dataset_instance
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
        json_list = self.create_json_list(self.dataset)
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

    def create_json_list(self, original_dataset: List[dict]) -> List[dict]:
        """
        Creates a list of jsons from the dataset. The json files have the desired format.

        Args:
            original_dataset: The dataset in the original format.
        """
        json_list = list()
        for dataset_entry in original_dataset:
            #Construct the sentence
            sentence = ""
            dataset_entry_data = dataset_entry["data"]
            for i, text_piece in enumerate(dataset_entry_data):
                sentence_piece = text_piece["text"]
                #Adding additional whitespace, it will get removed after tokenization
                sentence += (sentence_piece + " ") if i < len(dataset_entry_data) - 1 else sentence_piece
            sentence = re.sub(r" {2,}", " ", sentence)
            tokens = self.word_tokenizer.tokenize(sentence)

            #Extract the entities
            entities = []
            for text_piece in dataset_entry["data"]:
                if not self.only_temporal_entities and "entity" in text_piece and text_piece["entity"].lower().strip() != self.SNIPS_TIME_ENTITY_TAG:
                    entity_text = text_piece["text"]
                    entity_tokens = self.word_tokenizer.tokenize(entity_text)
                    entity_tag = text_piece["entity"].strip().lower()
                    if entity_tag != "":
                        entity_start_index, entity_end_index = find_sublist_in_list(tokens, entity_tokens)
                        entities += [{
                            "text": entity_text,
                            "type": entity_tag,
                            "start": entity_start_index,
                            "end": entity_end_index
                        }]
                elif "entity" in text_piece and text_piece["entity"].lower().strip() == self.SNIPS_TIME_ENTITY_TAG:
                    assert("TIMEX3" in text_piece)
                    for timex3 in text_piece["TIMEX3"]:
                        timex3_type = timex3["type"].strip().lower()
                        timex3_expression = timex3["expression"]
                        if timex3_type != "" and timex3_expression != "":
                            timex3_tokens = self.word_tokenizer.tokenize(timex3["expression"])
                            timex3_start_index, timex3_end_index = find_sublist_in_list(tokens, timex3_tokens)
                            timex3_tag = self.classes_dictionary[timex3_type]
                            entities += [{
                                "text": timex3_expression,
                                "type": timex3_tag,
                                "start": timex3_start_index,
                                "end": timex3_end_index
                            }]
            json_element = {
                "text": sentence,
                "tokens": tokens,
                "entity": entities
            }
            json_list += [json_element]
        return json_list

if __name__ == "__main__":
    converter_inputs = [
        {
            "input_filepaths": [
                "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/pate_and_snips/snips_train.json",
                "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/pate_and_snips/snips_valid.json"
            ],
            "output_filepath": "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted/snips_single",
            "single_entity_class": True,
            "crossvalidation_enabled": True,
            "folds": 10,
            "only_temporal_entities": True,
            "printmessage": "Converting dataset:\nSingle=True"
        },
        {
            "input_filepaths": [
                "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/pate_and_snips/snips_train.json",
                "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/pate_and_snips/snips_valid.json"
            ],
            "output_filepath": "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted/snips_multi",
            "single_entity_class": False,
            "crossvalidation_enabled": True,
            "folds": 10,
            "only_temporal_entities": True,
            "printmessage": "Converting dataset:\nSingle=False"
        }
    ]

    for converter_input in converter_inputs:
        input_filepaths: List[str] = converter_input["input_filepaths"]
        output_filepath: str = converter_input["output_filepath"]
        single_entity_class: bool = converter_input["single_entity_class"]
        crossvalidation_enabled: bool = converter_input["crossvalidation_enabled"]
        folds: int = converter_input["folds"]
        only_temporal_entities: bool = converter_input["only_temporal_entities"]
        printmessage: str = converter_input["printmessage"]
        print(printmessage)
        converter = SnipsDatasetConverter(input_filepaths, output_filepath, single_entity_class, crossvalidation_enabled, folds, only_temporal_entities)
        converter.convert_dataset()
        print("\n" + "-" * 100 + "\n")