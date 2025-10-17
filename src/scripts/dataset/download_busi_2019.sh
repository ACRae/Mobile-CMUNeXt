#!/bin/bash

# Check if the base directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

# Base directory where the dataset will be saved
base_dir=$1

# Dataset URLs
dataset_url="https://www.kaggle.com/api/v1/datasets/download/subhajournal/busi-breast-ultrasound-images-dataset"

# Dataset folder name
dataset_name="BUSI"
dataset_dir="$base_dir/$dataset_name"

# Create dataset structure
mkdir -p "$dataset_dir/train/images"
mkdir -p "$dataset_dir/train/masks/0"

# Function to organize files
organize_files() {
    local source_dir=$1
    
    # First, find all image files that are not masks
    find "$source_dir" -type f -name "*.png" ! -name "*_mask.png" ! -name "normal *.png" | while read img_file; do
        # Get the base filename without extension
        base_name=$(basename "$img_file" .png)
        # Construct the mask filename
        mask_file="${img_file%.*}_mask.png"
        
        # Check if corresponding mask exists
        if [ -f "$mask_file" ]; then
            # Move the image and mask to their respective directories
            echo "Moving pair: $base_name"
            mv "$img_file" "$dataset_dir/train/images/"
            mv "$mask_file" "$dataset_dir/train/masks/0"
        else
            echo "Warning: No matching mask found for $img_file"
        fi
    done
}

# Function to download and extract using unzip
download_and_extract() {
    local url=$1
    local output_dir=$2
    
    # Get the filename from the URL
    filename="busi.zip"
    
    # Download the file
    echo "Downloading $filename..."
    curl -L -o "$filename" "$url"
    
    # Create and use a temporary directory
    temp_dir=$(mktemp -d)
    echo "Unzipping $filename..."
    unzip -q "$filename" -d "$temp_dir"
    
    # Use the hardcoded directory name after extraction
    extracted_dir="$temp_dir/Dataset_BUSI_with_GT"
    
    # Organize files from each category
    echo "Organizing contents into their respective directories..."
    for category in "benign" "malignant" "normal"; do
        if [ -d "$extracted_dir/$category" ]; then
            echo "Processing $category images..."
            organize_files "$extracted_dir/$category"
        else
            echo "Warning: Category directory $category not found"
        fi
    done
    
    # Clean up
    echo "Cleaning up temporary files..."
    rm -rf "$temp_dir"
    rm "$filename"
}

# Download and extract the dataset
download_and_extract "$dataset_url" "$dataset_dir"

# Verify the organization
image_count=$(ls "$dataset_dir/train/images" | wc -l)
mask_count=$(ls "$dataset_dir/train/masks/0" | wc -l)
echo "Organization complete!"
echo "Total images: $image_count"
echo "Total masks: $mask_count"
if [ "$image_count" -eq "$mask_count" ]; then
    echo "Successfully matched all images with their masks!"
else
    echo "Warning: Number of images and masks don't match!"
fi

echo "All files downloaded, extracted, and organized successfully to $dataset_dir"