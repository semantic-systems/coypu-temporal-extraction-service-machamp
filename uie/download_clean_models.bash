#!/bin/bash

if [ ! -d "hf_models" ]; then
    # Create the directory
    mkdir hf_models
    echo "Directory 'hf_models' created."
fi

cd hf_models

# Get the base model
echo "Downloading the base model..."
wget https://www.fdr.uni-hamburg.de/record/13712/files/uie-base-en.zip
echo "Unzipping the base model..."
unzip uie-base-en.zip
echo "Removing the base model zip file..."
rm uie-base-en.zip

# Get the large model
echo "Downloading the large model..."
wget https://www.fdr.uni-hamburg.de/record/13714/files/uie-large-en.zip
echo "Unzipping the large model..."
unzip uie-large-en.zip
echo "Removing the large model zip file..."
rm uie-large-en.zip