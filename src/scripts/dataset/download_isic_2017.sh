#!/bin/bash

# Check if the base directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

# Base directory where the dataset will be saved
base_dir=$1

# Dataset URLs
train_img_url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
train_mask_url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip"
val_img_url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip"
val_mask_url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip"
test_img_url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
test_mask_url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"

# Dataset folder name
dataset_name="ISIC2017"
dataset_dir="$base_dir/$dataset_name"

# Create dataset structure
mkdir -p "$dataset_dir/train/images"
mkdir -p "$dataset_dir/train/masks/0"
mkdir -p "$dataset_dir/validation/images"
mkdir -p "$dataset_dir/validation/masks/0"
mkdir -p "$dataset_dir/test/images"
mkdir -p "$dataset_dir/test/masks/0"

# Function to download and unzip
download_and_extract() {
    local url=$1
    local output_dir=$2
    
    # Get the filename from the URL
    filename=$(basename "$url")
    
    # Download the file
    echo "Downloading $filename..."
    curl -L -o "$filename" "$url"
    
    # Unzip the file to a temporary directory
    temp_dir=$(mktemp -d)
    echo "Unzipping $filename..."
    unzip -q "$filename" -d "$temp_dir"
    
    # Move the contents of the unzipped folder to the output directory
    echo "Moving contents to $output_dir..."
    mv "$temp_dir"/*/* "$output_dir"  # This moves all contents inside the inner folder to the target folder
    
    # Clean up the temporary directory and zip file
    rm -rf "$temp_dir"
    rm "$filename"
}

# Download and extract train images and masks
download_and_extract "$train_img_url" "$dataset_dir/train/images"
download_and_extract "$train_mask_url" "$dataset_dir/train/masks/0"

# Download and extract validation images and masks
download_and_extract "$val_img_url" "$dataset_dir/validation/images"
download_and_extract "$val_mask_url" "$dataset_dir/validation/masks/0"

# Download and extract test images and masks
download_and_extract "$test_img_url" "$dataset_dir/test/images"
download_and_extract "$test_mask_url" "$dataset_dir/test/masks/0"

echo "All files downloaded, extracted, and moved successfully to $dataset_dir."
