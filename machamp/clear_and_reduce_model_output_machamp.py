"""
This script is used on the dirs produced during the finetuning process.
The goal is to clear the large model files and keep only the logs.
"""

import os
import shutil
import argparse
import sys
import re



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-bd", type=str, default="./logs")
    parser.add_argument("--dataset_name", "-d", type=str)
    parser.add_argument("--classes", "-c", type=str) # "multi" or "single"
    parser.add_argument("--output_dir", "-o", type=str, default="./crossvalidation-output")
    args = parser.parse_args()

    finetuning_models_base_dir = args.base_dir
    dataset_name = args.dataset_name
    classes = args.classes
    crossvalidation_output_dir = args.output_dir
    print("Input args are:")
    print("finetuning output models dir: ", finetuning_models_base_dir)
    print("dataset_name: ", dataset_name)
    print("classes: ", classes)
    print("crossvalidation output_dir: ", crossvalidation_output_dir)
    FILE_REGEX_TO_DELETE = [
        r"model.pt",
        r"model_[0-9]+.pt"
    ]


    #Check if output dir contains a folder with the result of the crossvalidation
    results_dirpath = os.path.join(crossvalidation_output_dir, f"{dataset_name}_{classes}_crossvalidation_logfiles")
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


    #Find all dirs that contain the dataset name and model size
    all_dirs = [d for d in os.listdir(finetuning_models_base_dir) if os.path.isdir(os.path.join(finetuning_models_base_dir, d))]
    relevant_dirs = []
    for dir in all_dirs:
        if dir.lower().startswith(f"{dataset_name}_{classes}".lower()) and ("_fold_" in dir.lower()):
            relevant_dirs.append(dir)

    if len(relevant_dirs) == 0:
        print("No dirs found. Exiting.")
        sys.exit()
    relevant_dirs.sort()


    #Move the dirs to a new folder
    new_paths = []
    new_dir_name = f"original-logs_{dataset_name}_{classes}"
    new_dir_path = os.path.join(results_dirpath, new_dir_name)
    new_base_dir = new_dir_path
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    for relevant_dir in relevant_dirs:
        relevant_dir_path = os.path.join(finetuning_models_base_dir, relevant_dir)
        shutil.move(relevant_dir_path, new_dir_path)
        print("Moving dir: ", relevant_dir_path, " to ", new_dir_path)
        new_paths.append(new_dir_path)


    #Delete files in new paths
    for new_path in new_paths:
        for root, dirs, files in os.walk(new_path, topdown=False):
            for name in files:
                matches_regex = False
                for regex in FILE_REGEX_TO_DELETE:
                    if re.match(regex, name):
                        matches_regex = True
                        break

                if matches_regex:
                    print("Deleting file: ", os.path.join(root, name))
                    #os.remove(os.path.join(root, name)) #TODO uncomment this line
