from .preprocessing_utils import slice_list, slice_list_in_equal_parts
import json
import os
from .jsonlines_prettyfier import JSONLinesPrettifier
from typing import List

def generate_crossvalidation_folds(
        json_list: List[dict],
        output_dirname: str,
        folds: int,
        output_file_train_suffix: str,
        output_file_val_suffix: str,
        output_file_test_suffix: str,
        output_file_prefix: str
    ):
    """
    Generates crossvalidation folds from the dataset and saves them to the filesystem.

    Args:
        json_list: The dataset in json format.
        output_dirname: The directory where the folds should be saved.
        folds: The number of folds.
        output_file_train_suffix: The suffix for the train file e.g. "_train.jsonlines".
        output_file_val_suffix: The suffix for the val file e.g. "_val.jsonlines".
        output_file_test_suffix: The suffix for the test file e.g. "_test.jsonlines".
        output_file_prefix: The prefix for the output files e.g. "pate".
    """
    prettifier = JSONLinesPrettifier()
    slices = slice_list_in_equal_parts(json_list, folds)

    for i, slice in enumerate(slices):
        val_index = i
        val_dataset = slice
        test_index = (i + 1) % folds
        test_dataset = slices[test_index]

        #Train dataset contains everything except the test dataset
        train_dataset = slices.copy()

        #Remove test and val from train, delte larger index first
        if test_index > val_index:
            del train_dataset[test_index]
            del train_dataset[val_index]
        else:
            del train_dataset[val_index]
            del train_dataset[test_index]
        #Flatten
        train_dataset = [item for sublist in train_dataset for item in sublist]

        #Create new dir in output_dirname
        new_dirname = f"fold_{i}"
        new_dirpath = os.path.join(output_dirname, "folds", new_dirname)
        os.makedirs(new_dirpath, exist_ok=True)

        #Save fold
        train_output_filepath = os.path.join(new_dirpath, output_file_prefix + output_file_train_suffix)
        train_pretty_output_filepath = os.path.join(new_dirpath, "pretty_" + output_file_prefix + output_file_train_suffix)
        save_dataset(train_dataset, train_output_filepath)
        prettifier.append_to_inputfiles(train_output_filepath)
        prettifier.append_to_outputfiles(train_pretty_output_filepath)

        val_output_filepath = os.path.join(new_dirpath, output_file_prefix + output_file_val_suffix)
        val_pretty_output_filepath = os.path.join(new_dirpath, "pretty_" + output_file_prefix + output_file_val_suffix)
        save_dataset(val_dataset, val_output_filepath)
        prettifier.append_to_inputfiles(val_output_filepath)
        prettifier.append_to_outputfiles(val_pretty_output_filepath)

        test_output_filepath = os.path.join(new_dirpath, output_file_prefix + output_file_test_suffix)
        test_pretty_output_filepath = os.path.join(new_dirpath, "pretty_" + output_file_prefix + output_file_test_suffix)
        save_dataset(test_dataset, test_output_filepath)
        prettifier.append_to_inputfiles(test_output_filepath)
        prettifier.append_to_outputfiles(test_pretty_output_filepath)
    prettifier.run()

def save_dataset(json_list: List[dict], output_filepath: str) -> None:
    """
    Saves the dataset to the filesystem.

    Args:
        json_list: The dataset in json format.
        output_filepath: The filepath where the dataset should be saved.
    """
    #Check if directories exist
    output_dirname = os.path.dirname(output_filepath)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname, exist_ok=True)

    with open(output_filepath, 'w') as outputfile:
        entries_count = len(json_list)
        for j, element in enumerate(json_list):
            json_element = (json.dumps(element)).strip() + '\n'
            #Remove last newline character
            if j == entries_count - 1: #TODO check with debugger
                json_element = json_element[:-1]
            outputfile.write(json_element)
    print(f"Writing to file \"{output_filepath}\"")

def save_dataset_splits(
        json_list: List[dict],
        output_directory_path: str,
        train_percent: float,
        test_percent: float,
        val_percent: float,
        output_file_train_suffix: str, 
        output_file_test_suffix: str,
        output_file_val_suffix: str,
        output_file_full_suffix: str,
        output_file_prefix: str
    ):
    """
    Splits the dataset into train, test and val and saves them to the filesystem.

    Args:
        json_list: The dataset in json format.
        output_directory_path: The directory where the folds should be saved.
        train_percent: The percentage of the dataset that should be used for training e.g. 0.8.
        test_percent: The percentage of the dataset that should be used for testing e.g. 0.1.
        val_percent: The percentage of the dataset that should be used for validation e.g. 0.1.
        output_file_train_suffix: The suffix for the train file e.g. "_train.jsonlines".
        output_file_val_suffix: The suffix for the val file e.g. "_val.jsonlines".
        output_file_test_suffix: The suffix for the test file e.g. "_test.jsonlines".
        output_file_full_suffix: The suffix for the full file e.g. "_full.jsonlines".
        output_file_prefix: The prefix for the output files e.g. "pate".
    """
    #Saves copies that are more human readable
    prettifier = JSONLinesPrettifier()

    train_list, test_list, val_list = (L for L in slice_list(json_list, *(train_percent, test_percent, val_percent)))

    for output_dataset_pair in [
        (output_file_train_suffix, train_list),
        (output_file_test_suffix, test_list),
        (output_file_val_suffix, val_list),
        (output_file_full_suffix, json_list)
    ]:
        current_suffix, current_dataset = output_dataset_pair
        output_filepath = os.path.join(output_directory_path, output_file_prefix + current_suffix)
        pretty_output_filepath = os.path.join(output_directory_path, "pretty_" + output_file_prefix + current_suffix)
        save_dataset(current_dataset, output_filepath)

        prettifier.append_to_inputfiles(output_filepath)
        prettifier.append_to_outputfiles(pretty_output_filepath)
    prettifier.run()