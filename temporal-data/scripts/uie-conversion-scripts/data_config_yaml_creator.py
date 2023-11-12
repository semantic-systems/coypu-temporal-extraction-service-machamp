import yaml
import os
import argparse

class UIEDataConfigYamlCreator:
    def __init__(self, input_directory_paths, output_directory_path, do_crossvalidation):
        self.input_directory_paths = input_directory_paths
        self.do_crossvalidation = do_crossvalidation
        self.output_directory_path = output_directory_path

        #If output path doesn't exist, create it
        if not os.path.exists(self.output_directory_path):
            os.makedirs(self.output_directory_path)

        #A dictionary that shows corresponence between dataset names and classnames used by UIE converter scripts
        self.dataset_dataclass_mapper = {
            "pate": "PATE",
            "fullpate": "FULLPATE",
            "snips": "SNIPS",
            "tweets": "TWEETS",
            "wikiwars": "WIKIWARS",
            "wikiwars-tagged": "WIKIWARSTAGGED",
            "timebank": "TIMEBANK",
            "aquaint": "AQUAINT",
            "tempeval": "TEMPEVAL"
        }

        self.yaml_type_mappers = {
            "multi": {
                "date": "date",
                "time": "time",
                "duration": "duration",
                "set": "set"
            },

            "single": {
                "tempexp": "tempexp",
            }
        }
        self.snips_yaml_mappers = {
            "cuisine": "cuisine",
            "party_size_number": "party_size_number",
            "facility": "facility",
            "spatial_relation": "spatial_relation",
            "sort": "sort",
            "restaurant_type": "restaurant_type",
            "poi": "poi",
            "state": "state",
            "country": "country",
            "party_size_description": "party_size_description",
            "served_dish": "served_dish",
            "restaurant_name": "restaurant_name",
            "city": "city"
        }

        for i, directory_path in enumerate(self.input_directory_paths):
            dataset_fullname = (os.path.basename(directory_path)).lower().strip()
            dataset_name = dataset_fullname.split("_")[0]
            dataset_type = dataset_fullname.split("_")[1]  

            yaml_files = list()
            mapper = self.yaml_type_mappers[dataset_type]
            # if dataset_name == "snips" or dataset_name == "fullpate":
            #     mapper = {**mapper, **self.snips_yaml_mappers}

            yaml_filename = f"{dataset_name}_{dataset_type}.yaml"
            yaml_data = {
                "name": f"{dataset_name}_{dataset_type}",
                "path": f"{os.path.abspath(directory_path)}",
                "data_class": self.dataset_dataclass_mapper[dataset_name],
                "split": {
                    "train": f"{dataset_name}-train.jsonlines",
                    "val": f"{dataset_name}-val.jsonlines",
                    "test": f"{dataset_name}-test.jsonlines"
                },
                "language": "en",
                "mapper": mapper
            }
            yaml_output_filepath = os.path.join(self.output_directory_path, yaml_filename)
            yaml_files += [(yaml_filename, yaml_output_filepath, yaml_data)]

            if self.do_crossvalidation:
                dataset_variation_path = directory_path
                dirs = [dir for dir in os.listdir(dataset_variation_path) if os.path.isdir(dataset_variation_path)]
                if "folds" in dirs:
                    folds_path = os.path.join(dataset_variation_path, "folds")
                    folds = [fold for fold in os.listdir(folds_path) if os.path.isdir(folds_path)]
                    folds.sort()
                    for fold in folds:
                        fold_path = os.path.join(folds_path, fold)
                        yaml_filename = f"{dataset_name}_{dataset_type}_{fold}.yaml"
                        yaml_data = {
                            "name": f"{dataset_name}_{dataset_type}_{fold}",
                            "path": f"{os.path.abspath(fold_path)}",
                            "data_class": self.dataset_dataclass_mapper[dataset_name],
                            "split": {
                                "train": f"{dataset_name}-train.jsonlines",
                                "val": f"{dataset_name}-val.jsonlines",
                                "test": f"{dataset_name}-test.jsonlines"
                            },
                            "language": "en",
                            "mapper": mapper
                        }
                        yaml_output_filepath = os.path.join(self.output_directory_path, yaml_filename)
                        yaml_files += [(yaml_filename, yaml_output_filepath, yaml_data)]


            
            for yaml_file in yaml_files:
                filename, filepath, yaml_data = yaml_file
                with open(filepath, "w") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False)
                    print(f"Created {filename}{' ' * 10}Path: {filepath}")


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
        default = "data_config/entity",
        help = "The directory for the newly converted data-config files."
    )

    parser.add_argument(
        "--crossvalidation",
        "-c",
        action = "store_true",
        help = "Wether to generate data-config files for the crossvalidation folds or not."
    )
    args = parser.parse_args()


    dataset_converter_output_dir = args.input_base_directory_path
    dataset_root_dirs = [os.path.join(dataset_converter_output_dir, dataset_root_dir) for dataset_root_dir in os.listdir(dataset_converter_output_dir)]
    dataset_root_dirs.sort(reverse=True)

    print("Found the following datasets:")
    [print(dataset_name) for dataset_name in dataset_root_dirs]
    print("\n" * 3)

    output_directory_path = args.output_directory

    yaml_creator = UIEDataConfigYamlCreator(
        input_directory_paths = dataset_root_dirs,
        output_directory_path = output_directory_path,
        do_crossvalidation = args.crossvalidation
    )

    