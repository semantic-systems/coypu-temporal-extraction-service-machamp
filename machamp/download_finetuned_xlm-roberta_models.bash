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

if [ ! -d "xlm-roberta_base" ]; then
    mkdir xlm-roberta_base
    echo "Directory 'finetuned_models/xlm-roberta_base' created."
fi

cd xlm-roberta_base

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13631/files/xlm-roberta-base_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13635/files/xlm-roberta-base_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13637/files/xlm-roberta-base_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13633/files/xlm-roberta-base_fullpate_multi.zip"

cd ..

if [ ! -d "xlm-roberta_large" ]; then
    mkdir xlm-roberta_large
    echo "Directory 'finetuned_models/xlm-roberta_large' created."
fi

cd xlm-roberta_large

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13589/files/xlm-roberta-large_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13591/files/xlm-roberta-large_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13587/files/xlm-roberta-large_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13593/files/xlm-roberta-large_fullpate_multi.zip"

cd ..