import re
import os
import argparse
import sys



FILE_REGEX_TO_DELETE = [
    r"model.pt",
    r"model_[0-9]+.pt"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-b", type=str, default="/export/home/4kirsano/machamp/second_batch_temporal_finetuning_logs")
    parser.add_argument("--confirm_delete", "-c", action="store_true")
    args = parser.parse_args()
    base_dir = args.base_dir
    confirm_delete = args.confirm_delete


    count = 0
    filepaths_to_delete = []
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in files:
                    matches_regex = False
                    for regex in FILE_REGEX_TO_DELETE:
                        if re.match(regex, name):
                            matches_regex = True
                            break
                    if matches_regex:
                        print("Found file to delete: ", os.path.join(root, name))
                        count += 1
                        filepaths_to_delete.append(os.path.join(root, name))

    print("Found", count, "files to delete.")
    print("Confirm flag is set to", confirm_delete)
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

    for filepath in filepaths_to_delete:
        if confirm_delete:
            os.remove(filepath)
            print("Deleted file: ", filepath)
    print("Deleted", count, "files!")
