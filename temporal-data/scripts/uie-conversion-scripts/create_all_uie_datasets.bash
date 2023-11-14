python data_config_yaml_creator.py --input_base_directory_path ../../entity/my_converted_datasets/jsonlines \
    --output_directory data_config/entity \
    --crossvalidation

python uie_convert.py -config data_config/entity/ \
    -output ../../entity/my_converted_datasets/uie