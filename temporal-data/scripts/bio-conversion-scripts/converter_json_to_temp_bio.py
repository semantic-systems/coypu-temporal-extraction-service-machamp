#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import os
from typing import List
import argparse

class Bio:
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

def read_jsonlines_file(file_name: str) -> List[dict]:
    """
    Reads a jsonlines file and returns the contents as a list of dictionaries.

    Args:
        file_name (str): Name of the file to read.

    Returns:
        List[dict]: List of dictionaries containing the contents of the file.
    """
    return [json.loads(line) for line in open(file_name)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_base_directory_path",
        "-i",
        type = str,
        default = "../../entity/my_converted_datasets/jsonlines",
        help = "Path to the base directory which contains all the converted datasets in jsonline format.",
    )

    parser.add_argument(
        "--output_directory",
        "-o",
        type = str,
        default = "../../entity/my_converted_datasets/bio",
        help = "The directory for the newly converted dataset files."
    )

    parser.add_argument(
        "--dataset_names",
        "-d", 
        nargs='+', 
        default=[
                "aquaint_multi",
                "timebank_multi",
                "tempeval_multi",
                "pate_multi",
                "snips_multi",
                "fullpate_multi",
                "wikiwars-tagged_multi"
                "tweets_multi"
            ],
        help = "The names of the datasets to be converted. The datasets have to be present in the base directory. Multiple selections are possible."
    )
    args = parser.parse_args()

    temporal_types = [
        "date",
        "time",
        "duration",
        "set"
    ]

    bio_tags = [
        "O",
        "B-DATE",
        "I-DATE",
        "B-TIME",
        "I-TIME",
        "B-DURATION",
        "I-DURATION",
        "B-SET",
        "I-SET"
    ]

    mapping = {
        "date": ("B-DATE", "I-DATE"),
        "time": ("B-TIME", "I-TIME"),
        "duration": ("B-DURATION", "I-DURATION"),
        "set": ("B-SET", "I-SET"),
        "none": ("O", "O"),
        "tempexp": ("B-TEMPEXP", "I-TEMPEXP")
    }
    
    json_datasets_base_directory = args.input_base_directory_path
    dataset_names = args.dataset_names
    dataset_directories = [
        os.path.join(json_datasets_base_directory, dataset_name) for dataset_name in dataset_names
    ]

    dataset_core_names = [
        dataset_name.split("_")[0] for dataset_name in dataset_names
    ]

    files = [
        "DATASETNAME-full.jsonlines",

        "DATASETNAME-train.jsonlines", 
        "DATASETNAME-test.jsonlines",
        "DATASETNAME-val.jsonlines",

        "folds/fold_0/DATASETNAME-test.jsonlines",
        "folds/fold_0/DATASETNAME-train.jsonlines",
        "folds/fold_0/DATASETNAME-val.jsonlines",

        "folds/fold_1/DATASETNAME-test.jsonlines",
        "folds/fold_1/DATASETNAME-train.jsonlines",
        "folds/fold_1/DATASETNAME-val.jsonlines",

        "folds/fold_2/DATASETNAME-test.jsonlines",
        "folds/fold_2/DATASETNAME-train.jsonlines",
        "folds/fold_2/DATASETNAME-val.jsonlines",

        "folds/fold_3/DATASETNAME-test.jsonlines",
        "folds/fold_3/DATASETNAME-train.jsonlines",
        "folds/fold_3/DATASETNAME-val.jsonlines",

        "folds/fold_4/DATASETNAME-test.jsonlines",
        "folds/fold_4/DATASETNAME-train.jsonlines",
        "folds/fold_4/DATASETNAME-val.jsonlines",

        "folds/fold_5/DATASETNAME-test.jsonlines",
        "folds/fold_5/DATASETNAME-train.jsonlines",
        "folds/fold_5/DATASETNAME-val.jsonlines",

        "folds/fold_6/DATASETNAME-test.jsonlines",
        "folds/fold_6/DATASETNAME-train.jsonlines",
        "folds/fold_6/DATASETNAME-val.jsonlines",

        "folds/fold_7/DATASETNAME-test.jsonlines",
        "folds/fold_7/DATASETNAME-train.jsonlines",
        "folds/fold_7/DATASETNAME-val.jsonlines",

        "folds/fold_8/DATASETNAME-test.jsonlines",
        "folds/fold_8/DATASETNAME-train.jsonlines",
        "folds/fold_8/DATASETNAME-val.jsonlines",

        "folds/fold_9/DATASETNAME-test.jsonlines",
        "folds/fold_9/DATASETNAME-train.jsonlines",
        "folds/fold_9/DATASETNAME-val.jsonlines"
    ]

    output_base_directory = args.output_directory

    for dataset_name, dataset_directory in zip(dataset_core_names, dataset_directories):
        filepaths = [
            os.path.join(dataset_directory, file.replace("DATASETNAME", dataset_name)) for file in files
        ]
        dataset_directory_name = dataset_directory.split("/")[-1]
        
        output_dataset_base_directory = os.path.join(output_base_directory, dataset_directory_name)
        for filepath in filepaths:
            jsonlines_file = read_jsonlines_file(filepath)
            bio_sentences: List[Bio] = []
            fold = filepath.split("/")[-2] if "fold" in filepath else ""

            for jsonline in jsonlines_file:
                tokens = jsonline["tokens"]
                entities = jsonline["entity"]

                bio_labels = [mapping["none"][0]] * len(tokens)
                
                for entity in entities:
                    if "start" in entity:
                        entity_type = entity["type"]
                        entity_start = int(entity["start"])
                        entity_end = int(entity["end"])
                    elif "offset" in entity:
                        entity_type = entity["type"]
                        entity_start = int(entity["offset"][0])
                        entity_end = int(entity["offset"][-1])
                    else:
                        raise Exception("No start or offset found in entity")

                    #Currently: handle cases with subentities by ignoring the outer entity                    
                    is_outer_entity = False
                    for check_entity in entities:
                        #Check if entity is a outer-entity of another entity
                        if "start" in check_entity:
                            check_entity_type = check_entity["type"]
                            check_entity_start = int(check_entity["start"])
                            check_entity_end = int(check_entity["end"])
                        elif "offset" in check_entity:
                            check_entity_type = check_entity["type"]
                            check_entity_start = int(check_entity["offset"][0])
                            check_entity_end = int(check_entity["offset"][-1])
                        else:
                            raise Exception("No start or offset found in entity")
                        
                        unequal = entity_start != check_entity_start or entity_end != check_entity_end
                        if unequal and entity_start <= check_entity_start and entity_end >= check_entity_end:
                            entity_start = check_entity_start
                            entity_end = check_entity_end
                            is_outer_entity = True

                    if is_outer_entity:
                        continue
                    
                    bio_labels[entity_start] = mapping[entity_type][0]
                    entity_range = range(entity_start + 1, entity_end + 1)
                    for i in entity_range:
                        bio_labels[i] = mapping[entity_type][1]
                    
                assert(len(tokens) == len(bio_labels))
                bio_instance = Bio(tokens, bio_labels)
                bio_sentences.append(bio_instance)
            
            filename = filepath.split("/")[-1].split(".")[0] + ".bio"
            if "folds" in filepath:
                output_dirpath = os.path.join(output_dataset_base_directory, "folds", fold)
                output_filepath = os.path.join(output_dirpath, filename)
            else:
                output_dirpath = os.path.join(output_dataset_base_directory)
                output_filepath = os.path.join(output_dirpath, filename)

            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)
            
            with open(output_filepath, "w") as f:
                for bio_sentence in bio_sentences:
                    for token, label in zip(bio_sentence.tokens, bio_sentence.labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")
                print(f"Finished writing to '{output_filepath}'")