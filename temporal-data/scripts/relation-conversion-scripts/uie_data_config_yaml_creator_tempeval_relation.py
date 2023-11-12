import yaml
import os
import argparse

class UIETempevalRelationDataConfigYamlCreator:
    def __init__(self, input_directory_paths, dataset_type, output_directory_path, do_crossvalidation, dataclass, type_mapper):
        #If output path doesn't exist, create it
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)

        for i, directory_path in enumerate(input_directory_paths):
            dataset_fullname = (os.path.basename(directory_path)).lower().strip()

            dataset_name = dataset_fullname
            if "_" in dataset_fullname:
                dataset_name = dataset_fullname.split("_")[0]

            yaml_files = list()

            yaml_filename = f"{dataset_name}_{dataset_type}.yaml"
            yaml_data = {
                "name": f"{dataset_name}_{dataset_type}",
                "path": f"{os.path.abspath(directory_path)}",
                "data_class": dataclass,
                "split": {
                    "train": f"{dataset_name}-train.jsonlines",
                    "val": f"{dataset_name}-val.jsonlines",
                    "test": f"{dataset_name}-test.jsonlines"
                },
                "language": "en",
                "mapper": type_mapper
            }
            yaml_output_filepath = os.path.join(output_directory_path, yaml_filename)
            yaml_files += [(yaml_filename, yaml_output_filepath, yaml_data)]

            if do_crossvalidation:
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
                            "data_class": dataclass,
                            "split": {
                                "train": f"{dataset_name}-train.jsonlines",
                                "val": f"{dataset_name}-val.jsonlines",
                                "test": f"{dataset_name}-test.jsonlines"
                            },
                            "language": "en",
                            "mapper": type_mapper
                        }
                        yaml_output_filepath = os.path.join(output_directory_path, yaml_filename)
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
        default = "../../relation/my_converted_datasets",
        help = "Path to the base directory which contains all the converted datasets in jsonline format.",
    )

    parser.add_argument(
        "--root_directories",
        "-r", 
        nargs='+', 
        default=["timebank", "aquaint", "tempeval"],
        help = "The names of the relation-dataset-root-directories."
    )

    parser.add_argument(
        "--output_directory",
        "-o",
        type = str,
        default = "data_config/relation_configs",
        help = "The directory for the newly converted data-config files."
    )

    parser.add_argument(
        "--crossvalidation",
        "-c",
        action = "store_true",
        help = "Wether to generate data-config files for the crossvalidation folds or not."
    )

    parser.add_argument(
        "--single_class",
        "-s",
        action = "store_true",
        help = "Wether to have the four timex3 temporal classes or only a single generic one."
    )

    parser.add_argument(
        "--create_entity",
        "-cen",
        action = "store_true",
        help = "Wether to generate data-config files for the entity data."
    )

    parser.add_argument(
        "--create_relation",
        "-cre",
        action = "store_true",
        help = "Wether to generate data-config files for the relation data."
    )

    parser.add_argument(
        "--create_event",
        "-cev",
        action = "store_true",
        help = "Wether to generate data-config files for the event data."
    )
    args = parser.parse_args()

    dataset_root_directories = args.root_directories


    if args.create_entity and not args.single_class:
        dirpath_prefix = args.input_base_directory_path
        dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in dataset_root_directories]
        dataset_root_dirs.sort(reverse=True)

        print("Found the following datasets:")
        [print(dataset_name) for dataset_name in dataset_root_dirs]
        print("\n" * 3)

        output_directory_path = os.path.join(args.output_directory, "tempeval_entity")

        dataset_dataclass = "TEMPEVALENTITY"

        yaml_temporal_type_mappers = {
            "date": "date",
            "time": "time",
            "duration": "duration",
            "set": "set"
        }

        yaml_creator = UIETempevalRelationDataConfigYamlCreator(
            input_directory_paths = dataset_root_dirs,
            dataset_type = "multi",
            output_directory_path = output_directory_path,
            do_crossvalidation = args.crossvalidation,
            dataclass = dataset_dataclass,
            type_mapper = yaml_temporal_type_mappers
        )


    #-----------------------------------------------------------------------------------------------------------------

    if args.create_entity and args.single_class:
        dirpath_prefix = args.input_base_directory_path
        dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in dataset_root_directories]
        dataset_root_dirs.sort(reverse=True)

        print("Found the following datasets:")
        [print(dataset_name) for dataset_name in dataset_root_dirs]
        print("\n" * 3)

        output_directory_path = os.path.join(args.output_directory, "tempeval_entity")

        dataset_dataclass = "TEMPEVALENTITY"

        yaml_temporal_type_mappers = {
            "tempexp": "tempexp",
        }

        yaml_creator = UIETempevalRelationDataConfigYamlCreator(
            input_directory_paths = dataset_root_dirs,
            dataset_type = "single",
            output_directory_path = output_directory_path,
            do_crossvalidation = args.crossvalidation,
            dataclass = dataset_dataclass,
            type_mapper = yaml_temporal_type_mappers
        )


    #-----------------------------------------------------------------------------------------------------------------
    
    if args.create_relation and not args.single_class:
        dirpath_prefix = args.input_base_directory_path
        dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in dataset_root_directories]
        dataset_root_dirs.sort(reverse=True)

        print("Found the following datasets:")
        [print(dataset_name) for dataset_name in dataset_root_dirs]
        print("\n" * 3)

        output_directory_path = os.path.join(args.output_directory, "tempeval_relation")

        dataset_dataclass = "TEMPEVALRELATION"

        type_mapper = {
            "date": "date",
            "duration": "duration",
            "set": "set",
            "time": "time",
            "event-occurrence": "event-occurrence",
            "event-reporting": "event-reporting",
            "event-state": "event-state",
            "event-i-action": "event-i-action",
            "event-i-state": "event-i-state",
            "event-aspectual": "event-aspectual",
            "event-perception": "event-perception",
            "event-empty": "event-empty",
            "before": "before",
            "is-included": "is-included",
            "after": "after",
            "includes": "includes",
            "simultaneous": "simultaneous",
            "identity": "identity",
            "during": "during",
            "ended-by": "ended-by",
            "begins": "begins",
            "ends": "ends",
            "begun-by": "begun-by",
            "iafter": "iafter",
            "ibefore": "ibefore",
            "during-inv": "during-inv",
        }

        yaml_creator = UIETempevalRelationDataConfigYamlCreator(
            input_directory_paths = dataset_root_dirs,
            dataset_type = "relation",
            output_directory_path = output_directory_path,
            do_crossvalidation = args.crossvalidation,
            dataclass = dataset_dataclass,
            type_mapper = type_mapper
        )


    #-----------------------------------------------------------------------------------------------------------------
    
    #Events = TIMEX + EVENT
    if args.create_event and not args.single_class:
        dirpath_prefix = args.input_base_directory_path
        dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in dataset_root_directories]
        dataset_root_dirs.sort(reverse=True)

        print("Found the following datasets:")
        [print(dataset_name) for dataset_name in dataset_root_dirs]
        print("\n" * 3)

        output_directory_path = os.path.join(args.output_directory, "tempeval_event")

        dataset_dataclass = "TEMPEVALRELATION"

        type_mapper = {
            "date": "date",
            "duration": "duration",
            "set": "set",
            "time": "time",
            "event-occurrence": "event-occurrence",
            "event-reporting": "event-reporting",
            "event-state": "event-state",
            "event-i-action": "event-i-action",
            "event-i-state": "event-i-state",
            "event-aspectual": "event-aspectual",
            "event-perception": "event-perception",
            "event-empty": "event-empty",
        }

        yaml_creator = UIETempevalRelationDataConfigYamlCreator(
            input_directory_paths = dataset_root_dirs,
            dataset_type = "multi",
            output_directory_path = output_directory_path,
            do_crossvalidation = args.crossvalidation,
            dataclass = dataset_dataclass,
            type_mapper = type_mapper
        )


    #-----------------------------------------------------------------------------------------------------------------
    
    #Events = TIMEX + EVENT
    if args.create_event and args.single_class:
        dirpath_prefix = args.input_base_directory_path
        dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in dataset_root_directories]
        dataset_root_dirs.sort(reverse=True)

        print("Found the following datasets:")
        [print(dataset_name) for dataset_name in dataset_root_dirs]
        print("\n" * 3)

        output_directory_path = os.path.join(args.output_directory, "tempeval_event")

        dataset_dataclass = "TEMPEVALRELATION"

        type_mapper = {
            "tempexp": "tempexp",
            "event-occurrence": "event-occurrence",
            "event-reporting": "event-reporting",
            "event-state": "event-state",
            "event-i-action": "event-i-action",
            "event-i-state": "event-i-state",
            "event-aspectual": "event-aspectual",
            "event-perception": "event-perception",
            "event-empty": "event-empty",
        }

        yaml_creator = UIETempevalRelationDataConfigYamlCreator(
            input_directory_paths = dataset_root_dirs,
            dataset_type = "single",
            output_directory_path = output_directory_path,
            do_crossvalidation = args.crossvalidation,
            dataclass = dataset_dataclass,
            type_mapper = type_mapper
        )