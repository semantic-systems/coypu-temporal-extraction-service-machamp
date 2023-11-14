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

if [ ! -d "roberta_base" ]; then
    mkdir roberta_base
    echo "Directory 'finetuned_models/roberta_base' created."
fi

cd roberta_base

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13681/files/roberta-base_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13686/files/roberta-base_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13683/files/roberta-base_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13679/files/roberta-base_fullpate_multi.zip"


cd ..

if [ ! -d "roberta_large" ]; then
    mkdir roberta_large
    echo "Directory 'finetuned_models/roberta_large' created."
fi

cd roberta_large

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13623/files/roberta-large_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13627/files/roberta-large_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13629/files/roberta-large_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13625/files/roberta-large_fullpate_multi.zip"


cd ..