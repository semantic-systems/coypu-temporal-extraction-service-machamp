# Installation of the punkt Tokenizer

Most of the scripts in this directory depend on the nltk punkt tokenizer.
It can be installed with the following bash script:

```
bash download_punkt_tokenizer.bash
```





# Quickstart

All the datasets can be created with the call of a single bash script:

```
bash create_all_datasets.bash
```

Note that this operation will generate the datasets in the directory: "[../../entity/my_converted_datasets](../../entity/my_converted_datasets)" with a size of about 1GB.
The script calls three other bash scripts that generate JSONLINES, BIO and UIE data respectively.
Each of them can be called alone: 

```
cd jsonlines-conversion-scripts && bash create_all_jsonline_datasets.bash
```

```
cd ../bio-conversion-scripts && bash create_all_bio_datasets.bash
```

```
cd ../uie-conversion-scripts && bash create_all_uie_datasets.bash
```

The bash scripts simply call Python scripts with a predefined set of parameters, which were also used in the thesis.
For more control, each of the Python scripts can be called.
Note that the JSONLINES datasets always need to be generated before BIO and UIE-format.





# How to use the dataset JSONLINE converters?

The Python converter scripts have all a similar set of parameters.
They differ slightly from case to case.
The main difference is that the "union" datasets TempEval-3 and Fullpate are created from previously converted subsets.
The other differences are based on the fact that the original datasets have different directory structures and differ in the amount of files used.
In this section, an example call is presented together with the synopsis for all the other scripts.

```
cd jsonlines-conversion-scripts
```

```
python converter_pate.py --input_filepaths ../../original_datasets/pate_and_snips/pate.json \
    --output_directory ../../entity/my_converted_datasets/jsonlines/pate_multi \
    --crossvalidation \
    --folds 10
```

To create the single-class version of this dataset use the `--single_class` flag.


## Synopsis of JSONLINE Converters

```
converter_pate.py [--input_filepaths INPUT_FILEPATHS [INPUT_FILEPATHS ...]] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS]

converter_snips.py [--input_filepaths INPUT_FILEPATHS [INPUT_FILEPATHS ...]] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS] [--only_temporal]

converter_fullpate.py [--input_filepath_snips INPUT_FILEPATH_SNIPS] [--input_filepath_pate INPUT_FILEPATH_PATE] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS] [--only_temporal]

converter_aquaint.py [--input_filepaths INPUT_FILEPATHS [INPUT_FILEPATHS ...]] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS]

converter_timebank.py [--input_filepath_timeml INPUT_FILEPATH_TIMEML] [--input_filepath_extra INPUT_FILEPATH_EXTRA] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS]

converter_tempeval.py [--input_filepath_timebank INPUT_FILEPATH_TIMEBANK] [--input_filepath_aquaint INPUT_FILEPATH_AQUAINT] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS]

converter_tweets.py [--input_test_filepath INPUT_TEST_FILEPATH] [--input_train_filepath INPUT_TRAIN_FILEPATH] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS]

converter_wikiwars-tagged.py [--input_parent_filepath INPUT_PARENT_FILEPATH] [--output_directory OUTPUT_DIRECTORY] [--single_class] [--crossvalidation] [--folds FOLDS]
```





# How to use the dataset BIO converters?

The BIO converter simply takes the datasets in the JSONLINES format and converts them to the BIO format.
Therefore, this step should be done after the JSONLINES conversion.

```
cd bio-conversion-scripts
```

The script requires a base directory with all the datasets in the JSONLINE format.
Furthermore, each dataset name (directory name) has to be specified.

```
converter_json_to_temp_bio.py [--input_base_directory_path INPUT_BASE_DIRECTORY_PATH] [--output_directory OUTPUT_DIRECTORY] [--dataset_names DATASET_NAMES [DATASET_NAMES ...]]
```

An Example, which converts all the multi-class datasets to the BIO format:

```
python converter_json_to_temp_bio.py --input_base_directory_path ../../entity/my_converted_datasets/jsonlines \
    --output_directory ../../entity/my_converted_datasets/bio \
    --dataset_names aquaint_multi timebank_multi tempeval_multi pate_multi snips_multi fullpate_multi \
     wikiwars-tagged_multi tweets_multi
```





# How to use the dataset UIE converters?

The UIE converter requires the creation of YAML configuration files.
For this the YAML creation script should be called first.
Example configuration files can be found under [data_config/examples](data_config/examples).

```
cd uie-conversion-scripts
```

```
python data_config_yaml_creator.py --input_base_directory_path ../../entity/my_converted_datasets/jsonlines \
    --output_directory data_config/entity \
    --crossvalidation
```

The script generates configuration files for all datasets present in the ``--input_base_directory_path``.
The ``--crossvalidation`` switch should be used to also generate files for each of the crossvalidation folds. 

The UIE converter requires the directory with the configuration files and the output directory:

```
python uie_convert.py -config data_config/entity/ \
    -output ../../entity/my_converted_datasets/uie
```





# How to use the analyis scripts?

The analysis scripts generate statistics for the datasets previously converted to the JSONLINE format.
The usage is as follows:

```
generic_dataset_analysis.py [--input_jsonlines_dataset_filepath INPUT_JSONLINES_DATASET_FILEPATH] [--output_filepath OUTPUT_FILEPATH] [--verbose]
```

The ``--verbose`` flag sets whether to print the output to the terminal as well.
The statistics can also be found in the [dataset-statistics directory](../dataset-statistics/).

Example usage:

```
python generic_dataset_analysis.py --input_jsonlines_dataset_filepath ../../entity/jsonlines/wikiwars_single/wikiwars-full.jsonlines \
    --output_filepath ../../dataset-statistics/statistics-wikiwars.txt
```