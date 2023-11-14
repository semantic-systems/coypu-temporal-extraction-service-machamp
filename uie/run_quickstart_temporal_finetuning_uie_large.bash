source config/data_conf/large_model_conf_temporal.ini
export model_name="uie-large-en"
path_prefix="../temporal-data/entity/uie-format/"

dataset_names=(
	tweets_multi
	fullpate_multi
	tempeval_multi
	wikiwars-tagged_multi
	# aquaint_multi
	# timebank_multi
	# pate_multi
	# snips_multi
)

for dataset_name in "${dataset_names[@]}"
do
    dataset_name="$dataset_name"
	data_folder=${path_prefix}/${dataset_name}
	echo "Datafolder is: $data_folder"
    echo "Initializing: $dataset_name ..."
    export dataset_name
	export data_folder
    bash scripts_exp/run_exp.bash
done