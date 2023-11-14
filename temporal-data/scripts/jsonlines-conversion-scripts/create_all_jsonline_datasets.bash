# Multi
python converter_pate.py --input_filepaths ../../original_datasets/pate_and_snips/pate.json --output_directory ../../entity/my_converted_datasets/jsonlines/pate_multi --crossvalidation --folds 10

python converter_snips.py --input_filepaths ../../original_datasets/pate_and_snips/snips_train.json ../../original_datasets/pate_and_snips/snips_valid.json --output_directory ../../entity/my_converted_datasets/jsonlines/snips_multi --crossvalidation --folds 10 --only_temporal

python converter_wikiwars-tagged.py --input_parent_filepath ../../original_datasets/wikiwars-tagged --output_directory ../../entity/my_converted_datasets/jsonlines/wikiwars-tagged_multi --crossvalidation --folds 10

python converter_aquaint.py --input_filepaths ../../original_datasets/aquaint --output_directory ../../entity/my_converted_datasets/jsonlines/aquaint_multi --crossvalidation --folds 10

python converter_timebank.py --input_filepath_timeml ../../original_datasets/timebank/data/timeml --input_filepath_extra ../../original_datasets/timebank/data/extra --output_directory ../../entity/my_converted_datasets/jsonlines/timebank_multi --crossvalidation --folds 10

python converter_tweets.py --input_test_filepath ../../original_datasets/tweets/testset --input_train_filepath ../../original_datasets/tweets/trainingset/ --output_directory ../../entity/my_converted_datasets/jsonlines/tweets_multi --crossvalidation --folds 10

python converter_tempeval.py --input_filepath_timebank ../../entity/my_converted_datasets/jsonlines/timebank_multi/timebank-full.jsonlines --input_filepath_aquaint ../../entity/my_converted_datasets/jsonlines/aquaint_multi/aquaint-full.jsonlines --output_directory ../../entity/my_converted_datasets/jsonlines/tempeval_multi --crossvalidation --folds 10

python converter_fullpate.py --input_filepath_snips ../../entity/my_converted_datasets/jsonlines/snips_multi/snips-full.jsonlines --input_filepath_pate ../../entity/my_converted_datasets/jsonlines/pate_multi/pate-full.jsonlines --output_directory ../../entity/my_converted_datasets/jsonlines/fullpate_multi --crossvalidation --folds 10 --only_temporal



# Single
python converter_pate.py --input_filepaths ../../original_datasets/pate_and_snips/pate.json --output_directory ../../entity/my_converted_datasets/jsonlines/pate_single --crossvalidation --folds 10 --single_class

python converter_snips.py --input_filepaths ../../original_datasets/pate_and_snips/snips_train.json ../../original_datasets/pate_and_snips/snips_valid.json --output_directory ../../entity/my_converted_datasets/jsonlines/snips_single --crossvalidation --folds 10 --only_temporal --single_class

python converter_wikiwars-tagged.py --input_parent_filepath ../../original_datasets/wikiwars-tagged --output_directory ../../entity/my_converted_datasets/jsonlines/wikiwars-tagged_single --crossvalidation --folds 10 --single_class

python converter_aquaint.py --input_filepaths ../../original_datasets/aquaint --output_directory ../../entity/my_converted_datasets/jsonlines/aquaint_single --crossvalidation --folds 10 --single_class

python converter_timebank.py --input_filepath_timeml ../../original_datasets/timebank/data/timeml --input_filepath_extra ../../original_datasets/timebank/data/extra --output_directory ../../entity/my_converted_datasets/jsonlines/timebank_single --crossvalidation --folds 10 --single_class

python converter_tweets.py --input_test_filepath ../../original_datasets/tweets/testset --input_train_filepath ../../original_datasets/tweets/trainingset/ --output_directory ../../entity/my_converted_datasets/jsonlines/tweets_single --crossvalidation --folds 10 --single_class

python converter_tempeval.py --input_filepath_timebank ../../entity/my_converted_datasets/jsonlines/timebank_single/timebank-full.jsonlines --input_filepath_aquaint ../../entity/my_converted_datasets/jsonlines/aquaint_single/aquaint-full.jsonlines --output_directory ../../entity/my_converted_datasets/jsonlines/tempeval_single --crossvalidation --folds 10 --single_class

python converter_fullpate.py --input_filepath_snips ../../entity/my_converted_datasets/jsonlines/snips_single/snips-full.jsonlines --input_filepath_pate ../../entity/my_converted_datasets/jsonlines/pate_single/pate-full.jsonlines --output_directory ../../entity/my_converted_datasets/jsonlines/fullpate_single --crossvalidation --folds 10 --only_temporal --single_class