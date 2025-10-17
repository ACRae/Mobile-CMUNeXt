#!/bin/bash

# Check if the base directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

base_dir=$1

rsync -av -e ssh --exclude='*.pth' antonio@10.64.10.96:medical-image-semantic-segmentation/src/python/saved_models/* "$base_dir"