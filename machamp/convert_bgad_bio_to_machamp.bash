dirnames=(
    tempeval
    tweets
    wikiwars
    mixed
)

filenames=(
    train.txt
    dev.txt
    test.txt
)

dataset_base_dir="temporal_data/bio"

for dirname in "${dirnames[@]}"; do
    #concatenate 
    dirpath="$dataset_base_dir/$dirname"

    #Check if dirpath contains a directory with the name original_files
    if [ -d "$dirpath/original_files" ]; then
        echo "Directory '$dirpath/original_files' exists."
        echo "Proceeding to file processing."
    else
        echo "Directory '$dirpath/original_files' does not exists."
        echo "Create the directory and place the original files there."

        mkdir "$dirpath/original_files"
        cp "$dirpath/"*.txt "$dirpath/original_files"

        echo "Created '$dirpath/original_files' and copied the original files there."
    fi

    for filename in "${filenames[@]}"; do
        filepath="$dirpath/$filename"
        echo "Processing '$filepath' ..."

        sed -i -e '/^#/d' -e 's/ /\t/g' "$filepath"
        echo "Done"
    done

    echo ""
done



