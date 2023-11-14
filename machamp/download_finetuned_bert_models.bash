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

if [ ! -d "bert_base" ]; then
    mkdir bert_base
    echo "Directory 'finetuned_models/bert_base' created."
fi

cd bert_base

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13698/files/bert-base_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13702/files/bert-base_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13700/files/bert-base_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13696/files/bert-base_fullpate_multi.zip"

cd ..

if [ ! -d "bert_large" ]; then
    mkdir bert_large
    echo "Directory 'finetuned_models/bert_large' created."
fi

cd bert_large

download_and_unzip "https://www.fdr.uni-hamburg.de/record/13706/files/bert-large_tempeval_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13710/files/bert-large_wikiwars_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13708/files/bert-large_tweets_multi.zip"
download_and_unzip "https://www.fdr.uni-hamburg.de/record/13704/files/bert-large_fullpate_multi.zip"

cd ..