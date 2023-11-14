source config/data_conf/large_model_conf_temporal.ini
export model_name="uie-large-en"
# This script uses the converted datasets by the scripts in "../temporal-data/scripts"
path_prefix="../temporal-data/entity/my_converted_datasets/uie-format/"

dataset_names=(
	pate_single_fold_0
	pate_single_fold_1
	pate_single_fold_2
	pate_single_fold_3
	pate_single_fold_4
	pate_single_fold_5
	pate_single_fold_6
	pate_single_fold_7
	pate_single_fold_8
	pate_single_fold_9
	
	snips_single_fold_0
	snips_single_fold_1
	snips_single_fold_2
	snips_single_fold_3
	snips_single_fold_4
	snips_single_fold_5
	snips_single_fold_6
	snips_single_fold_7
	snips_single_fold_8
	snips_single_fold_9

	fullpate_single_fold_0
	fullpate_single_fold_1
	fullpate_single_fold_2
	fullpate_single_fold_3
	fullpate_single_fold_4
	fullpate_single_fold_5
	fullpate_single_fold_6
	fullpate_single_fold_7
	fullpate_single_fold_8
	fullpate_single_fold_9

	tweets_single_fold_0
	tweets_single_fold_1
	tweets_single_fold_2
	tweets_single_fold_3
	tweets_single_fold_4
	tweets_single_fold_5
	tweets_single_fold_6
	tweets_single_fold_7
	tweets_single_fold_8
	tweets_single_fold_9

	wikiwars-tagged_single_fold_0
	wikiwars-tagged_single_fold_1
	wikiwars-tagged_single_fold_2
	wikiwars-tagged_single_fold_3
	wikiwars-tagged_single_fold_4
	wikiwars-tagged_single_fold_5
	wikiwars-tagged_single_fold_6
	wikiwars-tagged_single_fold_7
	wikiwars-tagged_single_fold_8
	wikiwars-tagged_single_fold_9

	aquaint_single_fold_0
	aquaint_single_fold_1
	aquaint_single_fold_2
	aquaint_single_fold_3
	aquaint_single_fold_4
	aquaint_single_fold_5
	aquaint_single_fold_6
	aquaint_single_fold_7
	aquaint_single_fold_8
	aquaint_single_fold_9

	timebank_single_fold_0
	timebank_single_fold_1
	timebank_single_fold_2
	timebank_single_fold_3
	timebank_single_fold_4
	timebank_single_fold_5
	timebank_single_fold_6
	timebank_single_fold_7
	timebank_single_fold_8
	timebank_single_fold_9

	tempeval_single_fold_0
	tempeval_single_fold_1
	tempeval_single_fold_2
	tempeval_single_fold_3
	tempeval_single_fold_4
	tempeval_single_fold_5
	tempeval_single_fold_6
	tempeval_single_fold_7
	tempeval_single_fold_8
	tempeval_single_fold_9
)

for dataset_name in "${dataset_names[@]}"
do
    dataset_name="$dataset_name"
	data_folder=${path_prefix}${dataset_name}
	echo "Datafolder is: $data_folder"
    echo "Initializing: $dataset_name ..."
    export dataset_name
	export data_folder
    bash scripts_exp/run_exp.bash
done
