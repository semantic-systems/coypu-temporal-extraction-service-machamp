source config/base_model_conf_temporal.ini
export model_name="uie-base-en"
path_prefix="../temporal-data/entity/uie-format/"

dataset_names=(
	tweets_multi
	fullpate_multi
	tempeval_multi
	wikiwars-tagged_multi
	aquaint_multi
	timebank_multi
	pate_multi
	snips_multi
)

for dataset_name in "${dataset_names[@]}"
do
    dataset_name="$path_prefix$dataset_name"
    echo "Initializing: $dataset_name ..."
    export dataset_name
    bash scripts_exp/run_exp.bash
done
