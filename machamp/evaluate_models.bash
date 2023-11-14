base_dir="./logs"
gold_base_dir="../temporal-data/entity/my_converted_datasets/bio"
dataset_type_name="test.bio" 

for main_model_dirname in "$base_dir"/*; do #e.g. AQUAINT_MULTI_CV_FOLD_0
    model_name=$(basename "$main_model_dirname")
    model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    model_name=$(echo "$model_name" | sed -e 's/_cv_fold_[0-9]//g')
    #Model name is the first split of modelname.split("_")
    # Split the string into an array
    IFS="_"
    read -ra parts <<< "$model_name"
    model_notype="${parts[0]}"
    fold_number=$(echo "$main_model_dirname" | grep -oP '(?<=fold_)[0-9]')
    if [ -z "$fold_number" ]; then
        fold_number=-1
    fi

    gold_dataset_filepath=$gold_base_dir/$model_name
    if [ "$fold_number" -ne -1 ]; then
        gold_dataset_filepath=$gold_dataset_filepath/folds/fold_$fold_number
    fi
    gold_dataset_filepath=$gold_dataset_filepath/$model_notype-$dataset_type_name
    #echo "$gold_dataset_filepath" 


    if [ -d "$main_model_dirname" ]; then
        for model_dir in "$main_model_dirname"/*; do #e.g. 2023.08.30_20.19.52
            if [ -d "$model_dir" ]; then
                for file in "$model_dir"/*; do #e.g. model_14.pt
                    if [ -f "$file" ] && [[ "$file" == *.pt ]]; then
                        model_file_basename=$(basename "$file")
                        IFS="."
                        read -ra parts <<< "$model_file_basename"
                        model_file_basename="${parts[0]}"
                        model_file=$file
                        model_name_upper=$(echo "$model_name" | tr '[:lower:]' '[:upper:]')
                        output_filepath="$model_dir/$model_name_upper-$model_file_basename.out.test"
                        echo ""
                        echo Running predict script with arguments:
                        echo "model_file: $model_file"
                        echo "gold_dataset_filepath: $gold_dataset_filepath"
                        echo "output_filepath: $output_filepath"
                        #echo "dataset: $model_name_upper"
                        command="python predict.py $model_file $gold_dataset_filepath $output_filepath --device 0"
                        echo "$command"
                        python predict.py "$model_file" "$gold_dataset_filepath" "$output_filepath" --device 0
                        echo ""
                        echo ""
                    fi
                done
            fi
        done
    fi
done

echo "Done evaluating models"