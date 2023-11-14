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

if [ ! -d "mbert" ]; then
    mkdir mbert
    echo "Directory 'finetuned_models/mbert' created."
fi

cd mbert

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13690/files/mbert_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13694/files/mbert_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13692/files/mbert_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13688/files/mbert_fullpate_multi.zip"