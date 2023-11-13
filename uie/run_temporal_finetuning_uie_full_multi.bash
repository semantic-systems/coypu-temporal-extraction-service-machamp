source config/data_conf/base_model_conf_temporal.ini
export model_name="uie-base-en"
path_prefix="../temporal-data/entity/uie-format/"

dataset_names=(
	pate_multi_fold_0
	pate_multi_fold_1
	pate_multi_fold_2
	pate_multi_fold_3
	pate_multi_fold_4
	pate_multi_fold_5
	pate_multi_fold_6
	pate_multi_fold_7
	pate_multi_fold_8
	pate_multi_fold_9
	
	snips_multi_fold_0
	snips_multi_fold_1
	snips_multi_fold_2
	snips_multi_fold_3
	snips_multi_fold_4
	snips_multi_fold_5
	snips_multi_fold_6
	snips_multi_fold_7
	snips_multi_fold_8
	snips_multi_fold_9

	fullpate_multi_fold_0
	fullpate_multi_fold_1
	fullpate_multi_fold_2
	fullpate_multi_fold_3
	fullpate_multi_fold_4
	fullpate_multi_fold_5
	fullpate_multi_fold_6
	fullpate_multi_fold_7
	fullpate_multi_fold_8
	fullpate_multi_fold_9

	tweets_multi_fold_0
	tweets_multi_fold_1
	tweets_multi_fold_2
	tweets_multi_fold_3
	tweets_multi_fold_4
	tweets_multi_fold_5
	tweets_multi_fold_6
	tweets_multi_fold_7
	tweets_multi_fold_8
	tweets_multi_fold_9

	wikiwars-tagged_multi_fold_0
	wikiwars-tagged_multi_fold_1
	wikiwars-tagged_multi_fold_2
	wikiwars-tagged_multi_fold_3
	wikiwars-tagged_multi_fold_4
	wikiwars-tagged_multi_fold_5
	wikiwars-tagged_multi_fold_6
	wikiwars-tagged_multi_fold_7
	wikiwars-tagged_multi_fold_8
	wikiwars-tagged_multi_fold_9

	aquaint_multi_fold_0
	aquaint_multi_fold_1
	aquaint_multi_fold_2
	aquaint_multi_fold_3
	aquaint_multi_fold_4
	aquaint_multi_fold_5
	aquaint_multi_fold_6
	aquaint_multi_fold_7
	aquaint_multi_fold_8
	aquaint_multi_fold_9

	timebank_multi_fold_0
	timebank_multi_fold_1
	timebank_multi_fold_2
	timebank_multi_fold_3
	timebank_multi_fold_4
	timebank_multi_fold_5
	timebank_multi_fold_6
	timebank_multi_fold_7
	timebank_multi_fold_8
	timebank_multi_fold_9

	tempeval_multi_fold_0
	tempeval_multi_fold_1
	tempeval_multi_fold_2
	tempeval_multi_fold_3
	tempeval_multi_fold_4
	tempeval_multi_fold_5
	tempeval_multi_fold_6
	tempeval_multi_fold_7
	tempeval_multi_fold_8
	tempeval_multi_fold_9
)

for dataset_name in "${dataset_names[@]}"
do
    dataset_name="$dataset_name"
	data_folder=../temporal-data/entity/my_converted_datasets/uie-format/${dataset_name}
	echo "Datafolder is: $data_folder"
    echo "Initializing: $dataset_name ..."
    export dataset_name
	export data_folder
    bash scripts_exp/run_exp.bash
done
