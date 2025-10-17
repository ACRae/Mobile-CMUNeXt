#!/bin/bash

# Check if the base directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

# Base directory where the dataset will be saved
base_dir=$1

# Dataset URLs
dataset_url="https://figshare.com/ndownloader/files/34969398"

# Dataset folder name
dataset_name="FIVES2022"
dataset_dir="$base_dir/$dataset_name"

# Create dataset structure
mkdir -p "$dataset_dir/train/images"
mkdir -p "$dataset_dir/train/masks/0"
mkdir -p "$dataset_dir/test/images"
mkdir -p "$dataset_dir/test/masks/0"

# Function to download and extract using unrar
download_and_extract() {
    local url=$1
    local output_dir=$2

    # Get the filename from the URL
    filename=$(basename "$url")

    # Download the file
    echo "Downloading $filename..."
    curl -L -o "$filename" "$url"

    # Unrar the file to a temporary directory
    temp_dir=$(mktemp -d)
    echo "Extracting $filename with unrar..."
    unrar x "$filename" "$temp_dir/" > /dev/null

    # Use the hardcoded directory name after extraction
    extracted_dir="$temp_dir/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"

    # Move the contents of the extracted folder to the appropriate output directories
    echo "Moving contents to $output_dir..."

    # Move the test data
    mv "$extracted_dir/test/Ground truth/"* "$dataset_dir/test/masks/0/" 2>/dev/null || echo "Test Ground truth folder not found."
    mv "$extracted_dir/test/Original/"* "$dataset_dir/test/images/" 2>/dev/null || echo "Test Original folder not found."

    # Move the train data
    mv "$extracted_dir/train/Ground truth/"* "$dataset_dir/train/masks/0/" 2>/dev/null || echo "Train Ground truth folder not found."
    mv "$extracted_dir/train/Original/"* "$dataset_dir/train/images/" 2>/dev/null || echo "Train Original folder not found."

    # Move the Quality Assessment.xlsx file to the base dataset directory
    mv "$extracted_dir/Quality Assessment.xlsx" "$dataset_dir/" 2>/dev/null || echo "Quality Assessment.xlsx file not found."

    # Clean up the temporary directory and rar file
    rm -rf "$temp_dir"
    rm "$filename"
}

# Download and extract the dataset
download_and_extract "$dataset_url" "$dataset_dir"

echo "All files downloaded, extracted, and moved successfully to $dataset_dir."
