#!/bin/bash

download_and_unzip() {
    local download_url=$1
    local model_name=$(basename "$download_url" .zip)

    echo "Downloading the $model_name model..."
    wget "$download_url"

    echo "Unzipping..."
    unzip "$model_name.zip"

    echo "Removing the zip file..."
    rm "$model_name.zip"
}


if [ ! -d "finetuned_models" ]; then
    mkdir finetuned_models
    echo "Directory 'finetuned_models' created."
fi

cd finetuned_models

if [ ! -d "base" ]; then
    mkdir base
    echo "Directory 'finetuned_models/base' created."
fi

cd base

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13599/files/uie-base-tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13595/files/uie-base-wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13597/files/uie-base-tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13601/files/uie-base-fullpate_multi.zip"

cd ..

if [ ! -d "large" ]; then
    mkdir large
    echo "Directory 'finetuned_models/large' created."
fi

cd large

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13615/files/large_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13617/files/large_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13619/files/large_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13621/files/large_fullpate_multi.zip"

cd ..