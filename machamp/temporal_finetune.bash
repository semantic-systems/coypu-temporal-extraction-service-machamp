config_files_directory=$1

#Get all files inside directory
files=$(ls $config_files_directory)

#Iterate and print filename
for file in $files
do
    #full filepath
    current_filepath="$config_files_directory/$file"
    echo $current_filepath
    python3 train.py --dataset_configs $current_filepath --device 0
done