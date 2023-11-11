import yaml
import os

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
                "path": f"{directory_path}",
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
                            "path": f"{fold_path}",
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
    dirpath_prefix = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted"
    dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in ["timebank_multi", "aquaint_multi", "tempeval_multi"]]
    dataset_root_dirs.sort(reverse=True)

    print("Found the following datasets:")
    [print(dataset_name) for dataset_name in dataset_root_dirs]
    print("\n" * 3)

    output_directory_path = "/export/home/4kirsano/uie/dataset_processing/data_config/tempeval_entity"

    dataset_dataclass = "TEMPEVALENTITY"

    yaml_temporal_type_mappers = {
        "timex-date": "timex-date",
        "timex-time": "timex-time",
        "timex-duration": "timex-duration",
        "timex-set": "timex-set"
    }

    yaml_creator = UIETempevalRelationDataConfigYamlCreator(
        input_directory_paths = dataset_root_dirs,
        dataset_type = "multi",
        output_directory_path = output_directory_path,
        do_crossvalidation = True,
        dataclass = dataset_dataclass,
        type_mapper = yaml_temporal_type_mappers
    )


    #-----------------------------------------------------------------------------------------------------------------


    dirpath_prefix = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/converted"
    dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in ["timebank_single", "aquaint_single", "tempeval_single"]]
    dataset_root_dirs.sort(reverse=True)

    print("Found the following datasets:")
    [print(dataset_name) for dataset_name in dataset_root_dirs]
    print("\n" * 3)

    output_directory_path = "/export/home/4kirsano/uie/dataset_processing/data_config/tempeval_entity"

    dataset_dataclass = "TEMPEVALENTITY"

    yaml_temporal_type_mappers = {
        "tempexp": "tempexp",
    }

    yaml_creator = UIETempevalRelationDataConfigYamlCreator(
        input_directory_paths = dataset_root_dirs,
        dataset_type = "single",
        output_directory_path = output_directory_path,
        do_crossvalidation = True,
        dataclass = dataset_dataclass,
        type_mapper = yaml_temporal_type_mappers
    )


    #-----------------------------------------------------------------------------------------------------------------
    

    dirpath_prefix = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/temporal_relation"
    dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in ["timebank", "aquaint", "tempeval"]]
    dataset_root_dirs.sort(reverse=True)

    print("Found the following datasets:")
    [print(dataset_name) for dataset_name in dataset_root_dirs]
    print("\n" * 3)

    output_directory_path = "/export/home/4kirsano/uie/dataset_processing/data_config/tempeval_relation"

    dataset_dataclass = "TEMPEVALRELATION"

    type_mapper = {
        "timex-date": "timex-date",
        "timex-duration": "timex-duration",
        "timex-set": "timex-set",
        "timex-time": "timex-time",
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
        do_crossvalidation = True,
        dataclass = dataset_dataclass,
        type_mapper = type_mapper
    )


    #-----------------------------------------------------------------------------------------------------------------
    

    #Events = TIMEX + EVENT
    dirpath_prefix = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/temporal_event"
    dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in ["timebank_multi", "aquaint_multi", "tempeval_multi"]]
    dataset_root_dirs.sort(reverse=True)

    print("Found the following datasets:")
    [print(dataset_name) for dataset_name in dataset_root_dirs]
    print("\n" * 3)

    output_directory_path = "/export/home/4kirsano/uie/dataset_processing/data_config/tempeval_event"

    dataset_dataclass = "TEMPEVALRELATION"

    type_mapper = {
        "timex-date": "timex-date",
        "timex-duration": "timex-duration",
        "timex-set": "timex-set",
        "timex-time": "timex-time",
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
        do_crossvalidation = True,
        dataclass = dataset_dataclass,
        type_mapper = type_mapper
    )


    #-----------------------------------------------------------------------------------------------------------------
    

    #Events = TIMEX + EVENT
    dirpath_prefix = "/export/home/4kirsano/uie/dataset_processing/data/my_datasets/temporal_event"
    dataset_root_dirs = [os.path.join(dirpath_prefix, path) for path in ["timebank_single", "aquaint_single", "tempeval_single"]]
    dataset_root_dirs.sort(reverse=True)

    print("Found the following datasets:")
    [print(dataset_name) for dataset_name in dataset_root_dirs]
    print("\n" * 3)

    output_directory_path = "/export/home/4kirsano/uie/dataset_processing/data_config/tempeval_event"

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
        do_crossvalidation = True,
        dataclass = dataset_dataclass,
        type_mapper = type_mapper
    )