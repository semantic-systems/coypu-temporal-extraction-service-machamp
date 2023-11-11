#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import random
import os
from typing import List
from conversion_utils.preprocessing_utils import find_sublist_in_list
from conversion_utils.preprocessing_file_saver import generate_crossvalidation_folds, save_dataset_splits
from conversion_utils.preprocessing_utils import DatasetNltkTokenizer
import argparse
import sys

class TweetsDatasetConverter:
    """
    This class converts the Tweets dataset to the jsonline format. The class takes into account the details of the Tweets dataset.
    It also implements crossvalidation and saves the dataset in multiple copies.

    Dataset can be converted in different ways. The following options are available:
        - Single entity class: All timex3 types are mapped to a single entity class e.g. "tempexp"
        - Multiple entity classes: Keeps the original timex3 types e.g. "date", "time", "duration", "set"

    It outputs/splits the dataset into train and test dataset and also outputs a human readable version of the dataset for each split.
    Note: that this can be very memory intensive, if this script is applied to very large datasets.
    """
    def __init__(self, input_filepaths: List[str], output_directory_path: str, single_entity_class: bool, crossvalidation_enabled: bool = False, folds: int = 10) -> None:
        self.contents_regex_pattern_timex3 = r"<TIMEX3[^>]*>.*?</TIMEX3>"
        self.contents_regex_pattern_timeml = r"<TimeML[^>]*>.*?</TimeML>"
        self.contents_regex_pattern_text = r"<TEXT[^>]*>.*?</TEXT>"

        self.tag_regex_pattern_anyxml = "<[^>]+>"
        self.tag_regex_pattern_timeml_open = "<TimeML[^>]*>"
        self.tag_regex_pattern_timeml_close = "</TimeML[^>]*>"
        self.tag_regex_pattern_xml = "<\?xml[^>]+>"
        
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
        self.output_file_prefix = "tweets"
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

        #Load tml input files
        self.tml_files = self.load_tml_files(self.input_filepaths)
        self.dataset_texts = self.load_dataset_texts(self.tml_files)

        self.dataset = list()

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
            with open(filepath, "r", encoding="utf-8") as file:
                original_contents = file.read()
            contents = original_contents.replace("\n", "").strip()
            tml_file_contents += [contents]
        return tml_file_contents
        
    def load_dataset_texts(self, tml_files: List[str]) -> List[dict]:
        """
        Loads the dataset text from the tml files. Also does some basic data cleaning.

        Args:
            dataset_filepaths: List of filepath to the dataset files.

        Returns:
            A list of dictionaries. Each dictionary the text of a dataset entry.
        """
        dataset = list()
        for tml_file_contents in tml_files:
            text_contents = ""

            match = re.search(self.contents_regex_pattern_text, tml_file_contents, re.DOTALL)
            if match:
                text_contents = match.group()
            else:
                raise Exception("Couldn't find text tags! Breaking method...")

            #Remove web urls
            text_contents = re.sub(r"http://\S+", "", text_contents)
            text_contents = re.sub(r"https://\S+", "", text_contents)

            text_contents = text_contents.replace("<TEXT>", "").replace("</TEXT>", "").strip()
            dataset += [text_contents]
        return dataset

    def repair_twitter_names(self, text, tokens, symbol):
        """
        Repairs twitter names that were split into multiple tokens. This happens with the symbols "@" and "#".

        Args:
            text: The text that contains potentially broken twitter names.
            tokens: The tokens of the text.
            symbol: The symbol that is used to mark twitter names. Either "@" or "#".

        Returns:
            The repaired tokens. Broken tokens are combined into one token.
        """
        repaired_tokens = []
        skip_next = False

        for i in range(len(tokens)):
            if skip_next:
                skip_next = False
                continue
            if tokens[i].strip() == symbol and i < len(tokens) - 1:  #Make sure that special symbol is not the last token
                combined = tokens[i].strip() + tokens[i+1].strip()
                if combined in text:  #Check if "@" or "#" was originally together with the next token
                    repaired_tokens.append(combined)
                    skip_next = True  #Skip next token as it is already added
                else:
                    repaired_tokens.append(tokens[i])
            else:
                repaired_tokens.append(tokens[i])
        return repaired_tokens
    
    def convert_dataset(self) -> None:
        """
        Converts the dataset into json format and writes it to the filesystem.
        The dataset is saved in multiple copies:
            (1) Full dataset
            (2) Train dataset / Test dataset
            (3) Train dataset / Test dataset for each crossvalidation fold

        Prior to conversion the dataset is shuffled.
        """
        self.dataset = self.create_json_list(self.dataset_texts)
        json_list = self.dataset
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

    def create_json_list(self, original_dataset_texts: List[dict]) -> List[dict]:
        """
        Converts the original texts into a json list.

        Args:
            original_dataset_texts: The original texts of the dataset. Including inline XML tags.

        Returns:
            A list of dictionaries. Each dictionary represents a dataset entry in json format.
        """
        json_dataset = list()
        for dataset_entry_text in original_dataset_texts:
            xml_text = dataset_entry_text

            xml_text = xml_text.replace("-", " - ") #Adding spaces for durations like 9-10

            #Replace dots that are not part of a number with spaces
            single_dot_pattern = r'(?<!\.)\.(?!\.)'
            xml_text = re.sub(single_dot_pattern, ' . ', xml_text)

            xml_text = re.sub(r' {2,}', ' ', xml_text) #Replace multiple spaces with one space

            text_contents = re.sub(self.tag_regex_pattern_anyxml, "", xml_text) #Remove all xml tags including attributes

            #Tokenize text
            tokens = self.word_tokenizer.tokenize(text_contents)
            tokens = self.repair_twitter_names(text_contents, tokens, "@")
            tokens = self.repair_twitter_names(text_contents, tokens, "#")

            # Extract entities from time tags
            entities = []
            entity_start_index = -1
            entity_end_index = -1
            timex3_tags = re.findall(self.contents_regex_pattern_timex3, xml_text)
            for timex3_tag in timex3_tags:
                timex3_type_match = re.search(r'type="[^"]+"', timex3_tag)
                timex3_type = ""
                #Extract type of timex3 tag
                if timex3_type_match:
                    timex3_type = timex3_type_match.group()
                    timex3_type = timex3_type.replace('type="', "").replace('"', "")
                    timex3_type = timex3_type.strip().lower()
                    timex3_type = self.classes_dictionary[timex3_type]

                #Extract content of timex3 tag
                time_content = (re.sub(self.tag_regex_pattern_anyxml, "", timex3_tag)).strip()
                entity_tokens = []
                if time_content != "":
                    entity_tokens = self.word_tokenizer.tokenize(time_content)
                    if "@" in time_content: entity_tokens = self.repair_twitter_names(text_contents, entity_tokens, "@")
                    if "#" in time_content: entity_tokens = self.repair_twitter_names(text_contents, entity_tokens, "#")
                    entity_start_index, entity_end_index = find_sublist_in_list(tokens, entity_tokens)
                    entity_dict = {"text": time_content, "type": timex3_type, "start": entity_start_index, "end": entity_end_index}
            
                entities += [entity_dict]

            is_corrupted = False
            for entity in entities:
                if entity["start"] == -1 or entity["end"] == -1:
                    is_corrupted = True

            if not is_corrupted:
                json_dataset += [{
                    "text": text_contents,
                    "tokens": tokens,
                    "entity": entities
                }]
        return json_dataset
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_test_filepath",
        "-ite", 
        type = str,
        default = "../original_datasets/tweets/testset",
        help = "The original Tweets dataset consists of many TML files, which are split into test and training subsets. This parameter expects the testset directory."
    )

    parser.add_argument(
        "--input_train_filepath",
        "-itr", 
        type = str,
        default = "../original_datasets/tweets/trainingset",
        help = "The original Tweets dataset consists of many TML files in one directory, which is split into test and training subsets. This parameter expects the trainingset directory."
    )

    parser.add_argument(
        "--output_directory",
        "-o",
        type = str,
        default = "../entity/my_datasets/jsonlines/tweets_multi",
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

    tweets_directory_trainingset = args.input_train_filepath
    tweets_files_trainingset = [os.path.join(tweets_directory_trainingset, f) for f in os.listdir(tweets_directory_trainingset) if os.path.isfile(os.path.join(tweets_directory_trainingset, f)) and f.endswith(".tml")]

    tweets_directory_testset = args.input_test_filepath
    tweets_files_testset = [os.path.join(tweets_directory_testset, f) for f in os.listdir(tweets_directory_testset) if os.path.isfile(os.path.join(tweets_directory_testset, f)) and f.endswith(".tml")]

    tweets_filepaths = tweets_files_trainingset + tweets_files_testset
    tweets_filepaths.sort()


    #Validate input
    is_error: bool = False
    if (len(tweets_filepaths) == 0) or not isinstance(tweets_filepaths[0], str) or not tweets_filepaths[0].endswith(".tml"):
        is_error = True

    if args.output_directory is None:
        is_error = True

    if is_error:
        print("Problem with input arguments.")
        sys.exit()

    
    print(f"Loading Tweets conversion script...")
    print(f"Following arguments were passed:")
    print(f"Tweets dataset train input filepath:    {args.input_train_filepath} => {type(args.input_train_filepath)}")
    print(f"Tweets dataset test input filepath:     {args.input_test_filepath} => {type(args.input_test_filepath)}")
    print(f"Output directory:                       {args.output_directory} => {type(args.output_directory)}")
    print(f"Single class only:                      {args.single_class} => {type(args.single_class)}")
    print(f"Crossvalidation enabled:                {args.crossvalidation} => {type(args.crossvalidation)}")
    print(f"Number of folds:                        {args.folds} => {type(args.folds)}")


    print()
    if not os.path.exists(args.output_directory):
        print(f"Output directory does not exist. Creating directory '{os.path.abspath(args.output_directory)}'.\n")
        os.makedirs(os.path.abspath(args.output_directory))

    converter = TweetsDatasetConverter(
        input_filepaths=tweets_filepaths,
        output_directory_path=args.output_directory,
        single_entity_class=args.single_class,
        crossvalidation_enabled=args.crossvalidation,
        folds=args.folds
    )
    converter.convert_dataset()