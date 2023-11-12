"""
This script is used on the dirs produced during the finetuning process.
The goal is to clear the large model files and keep only the logs.
Further this script renames the directory to a shorter name.

meta_2023-09-13-07-17-28417_hf_models_uie-base-en_spotasoc_entity_snips_multi_fold_4_e15_linear_lr1e-4_ls0_b16_wu0.06_n-1_RP_sn0.1_an0.1_run1
 => original_logs_base_snips_multi_fold_4

meta_2023-09-13-07-06-31322_hf_models_uie-base-en_spotasoc_entity_snips_multi_fold_1_e15_linear_lr1e-4_ls0_b16_wu0.06_n-1_RP_sn0.1_an0.1_run1
 => original_logs_base_snips_multi_fold_1

meta_2023-09-14-09-38-5692_hf_models_uie-large-en_spotasoc_entity_fullpate_multi_fold_2_e15_linear_lr1e-4_ls0_b16_wu0.06_n-1_RP_sn0.1_an0.1_run1
 => original_logs_large_fullpate_multi_fold_2
"""

import os
import shutil
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-b", type=str, default="/export/home/4kirsano/uie/output")
    parser.add_argument("--model_size", "-m", type=str) # "base" or "large"
    parser.add_argument("--dataset_name", "-d", type=str)
    parser.add_argument("--classes", "-c", type=str) # "multi" or "single"
    args = parser.parse_args()




    finetuning_output_base_dir = args.base_dir
    model_size = args.model_size
    dataset_name = args.dataset_name
    classes = args.classes
    print("Input args are:")
    print("finetuning_output_base_dir: ", finetuning_output_base_dir)
    print("model_size: ", model_size)
    print("dataset_name: ", dataset_name)
    print("classes: ", classes)

    FILES_TO_DELETE = [
        "spice.model", 
        "spiece.model", 
        "pytorch_model.bin", 
        "tokenizer.json", 
        "training_args.bin", 
        "scheduler.pt", 
        "rng_state.pth",
        "optimizer.pt",
    ]


    #Check if base dir contains a folder with the result of the crossvalidation
    results_dirpath = os.path.join(finetuning_output_base_dir, f"{model_size}_{dataset_name}_{classes}_crossvalidation_logfiles")
    if not os.path.exists(results_dirpath):
        print("No crossvalidation results dir found...")
        print(f"Path '{results_dirpath}' does not exist.")
        print("Do you want to continue? (y/n)")
        while True:
            choice = input("Yes (y) or No (n)  > ")
            choice = choice.lower().strip()
            if choice == "y":
                break
            elif choice == "n":
                sys.exit()
            else:
                print("Invalid input. Try again.")

    #Move all train log files into a directory called train_logs
    all_files = [f for f in os.listdir(finetuning_output_base_dir) if os.path.isfile(os.path.join(finetuning_output_base_dir, f))]

    train_log_files = [f for f in all_files if f.startswith("meta") and (f"_{dataset_name}_" in f) and (model_size in f) and (classes in f)]

    train_log_dir_path = os.path.join(finetuning_output_base_dir, f"train_logs_{model_size}_{dataset_name}_{classes}")
    if not os.path.exists(train_log_dir_path):
        os.mkdir(train_log_dir_path)

        for train_log_file in train_log_files:
            #Example filenames:
            #meta_2023-09-14-16-21-14198_hf_models_uie-base-en_spotasoc_entity_tempeval_single_fold_6_e18_linear_lr1e-4_ls0_b16_wu0.06_n-1_RP_sn0.1_an0.1_run1
            #meta_2023-09-14-15-59-20184_hf_models_uie-base-en_spotasoc_entity_tempeval_single_fold_5_e18_linear_lr1e-4_ls0_b16_wu0.06_n-1_RP_sn0.1_an0.1_run1
            train_log_file_path = os.path.join(finetuning_output_base_dir, train_log_file)
            shutil.move(train_log_file_path, train_log_dir_path)
            print("Moving file: ", train_log_file_path, " to ", train_log_dir_path)


    all_dirs = [d for d in os.listdir(finetuning_output_base_dir) if os.path.isdir(os.path.join(finetuning_output_base_dir, d))]

    #Find all dirs that contain the dataset name and model size
    relevant_dirs = []
    for dir in all_dirs:
        if dir.startswith("meta") and (f"_{dataset_name}_" in dir) and (model_size in dir) and (classes in dir):
            relevant_dirs.append(dir)

    if len(relevant_dirs) == 0:
        print("No dirs found. Exiting.")
        sys.exit()


    #Move the dirs to a new folder
    new_dir_name = f"original-logs_{model_size}_{dataset_name}_{classes}"
    new_dir_path = os.path.join(finetuning_output_base_dir, new_dir_name)
    new_base_dir = new_dir_path
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    for relevant_dir in relevant_dirs:
        relevant_dir_path = os.path.join(finetuning_output_base_dir, relevant_dir)
        shutil.move(relevant_dir_path, new_dir_path)
        print("Moving dir: ", relevant_dir_path, " to ", new_dir_path)


    #Rename the dirs
    new_paths = []
    for relevant_dir in relevant_dirs:
        splits = relevant_dir.split("_")
        relevant_dir_path = os.path.join(new_base_dir, relevant_dir)
        new_dir_name = f"{splits[4]}_{splits[7]}_{splits[8]}_fold_{splits[10]}"
        new_dir_path = os.path.join(new_base_dir, new_dir_name)
        print("Renaming dir: ", relevant_dir_path, " to ", new_dir_path)
        os.rename(relevant_dir_path, new_dir_path)
        new_paths.append(new_dir_path)


    #Delete files in new paths
    for new_path in new_paths:
        for root, dirs, files in os.walk(new_path, topdown=False):
            for name in files:
                if name in FILES_TO_DELETE:
                    print("Deleting file: ", os.path.join(root, name))
                    os.remove(os.path.join(root, name))
