#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

usage() {
    echo "Usage: $0 <base_directory> <model_name>"
    echo "Example: $0 /path/to/destination model_name"
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage
fi

base_dir=$1
model_name=$2

if [ ! -d "$base_dir" ]; then
    echo "Error: Base directory '$base_dir' does not exist."
    exit 1
fi


remote_user="antonio"
remote_host="10.64.10.96"
remote_path="medical-image-semantic-segmentation/src/python/saved_models/$model_name"

rsync -av -e ssh "$remote_user@$remote_host:$remote_path/" "$base_dir/$model_name"


if [ $? -eq 0 ]; then
    echo "Completed successfully."
else
    echo "Failed. Please check the source or destination paths and try again."
    exit 1
fi


# rsync -av -e ssh antonio@10.64.10.96:medical-image-semantic-segmentation/src/python/models.tar.gz ./models.tar.gz