from typing import List, Dict
import json
import os


def create_json_config_file(dataset_name, train_data_path, dev_data_path, output_filepath):
    # {
    #     "TWEETS": {
    #         "train_data_path": "temporal_data/bio/tweets/train.txt",
    #         "dev_data_path": "temporal_data/bio/tweets/dev.txt",
    #         "word_idx": 0,
    #         "tasks": {
    #             "ner": {
    #                 "task_type": "seq_bio",
    #                 "column_idx": 1,
    #                 "metric": "span_f1"
    #             }
    #         }
    #     }
    # }
    object = {
        dataset_name: {
            "train_data_path": train_data_path,
            "dev_data_path": dev_data_path,
            "word_idx": 0,
            "tasks": {
                "ner": {
                    "task_type": "seq_bio",
                    "column_idx": 1,
                    "metric": "span_f1"
                }
            }
        }
    }

    #Write to file
    with open(output_filepath, "w") as outfile:
        json.dump(object, outfile, indent=4)
    print(f"Created config file: '{output_filepath}'")

datasets_base_directory = "/export/home/4kirsano/machamp/temporal_data/temp_bio"

dataset_directories = [
    os.path.join(datasets_base_directory, model_dir) for model_dir in os.listdir(datasets_base_directory) if os.path.isdir(os.path.join(datasets_base_directory, model_dir))
]

dataset_fullnames = [ds.split("/")[-1].strip() for ds in dataset_directories]

output_filename_patterns = [
    "DATASETNAME.json",
    "DATASETNAME_cv_fold_0.json",
    "DATASETNAME_cv_fold_1.json",
    "DATASETNAME_cv_fold_2.json",
    "DATASETNAME_cv_fold_3.json",
    "DATASETNAME_cv_fold_4.json",
    "DATASETNAME_cv_fold_5.json",
    "DATASETNAME_cv_fold_6.json",
    "DATASETNAME_cv_fold_7.json",
    "DATASETNAME_cv_fold_8.json",
    "DATASETNAME_cv_fold_9.json",
]

output_base_directory = "/export/home/4kirsano/machamp/configs/my_configs"
if not os.path.exists(output_base_directory):
    os.makedirs(output_base_directory)

for dataset_directory, dataset_fullname in zip(dataset_directories, dataset_fullnames):
    dataset_name = dataset_fullname.split("_")[0].strip()
    dataset_fullname = dataset_fullname.strip().upper()
    for output_filename_pattern in output_filename_patterns:
        fold = ""
        if "cv" in output_filename_pattern:
            fold = output_filename_pattern.split("_")[-1].split(".")[0]
            fold = f"fold_{fold}"
            train_data_path = os.path.join(dataset_directory, "folds", fold, f"{dataset_name}-train.bio")
            dev_data_path   = os.path.join(dataset_directory, "folds", fold, f"{dataset_name}-val.bio")
            json_name = f"{dataset_fullname}_{fold}".upper()
        else:
            train_data_path = os.path.join(dataset_directory, f"{dataset_name}-train.bio")
            dev_data_path = os.path.join(dataset_directory, f"{dataset_name}-val.bio")
            json_name = f"{dataset_fullname}".upper()
            
        output_filepath = os.path.join(output_base_directory, output_filename_pattern.replace("DATASETNAME", dataset_fullname))
        
        print("\nCreating file ...")
        create_json_config_file(
            dataset_name=json_name,
            train_data_path=train_data_path,
            dev_data_path=dev_data_path,
            output_filepath=output_filepath
        )
