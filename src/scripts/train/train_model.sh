#!/bin/bash

set -euo pipefail

# Function to get current GPU utilization per device
get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' '
    else
        echo "No GPU found"
        return 1
    fi
}

# Function to find least utilized GPU
get_free_gpu() {
    local utilizations=($(get_gpu_usage))
    local min_util=100
    local min_idx=0
    
    for i in "${!utilizations[@]}"; do
        if (( ${utilizations[$i]} < min_util )); then
            min_util=${utilizations[$i]}
            min_idx=$i
        fi
    done
    
    echo "$min_idx"
}

# Function to count active training processes
count_training_processes() {
    pgrep -f "python.*main.py" | wc -l
}

# Function to read datasets from CSV file
read_datasets_csv() {
    declare -n dataset_map_ref=$1
    while IFS="," read -r dataset img_ext mask_ext || [ -n "$dataset" ]; do
        [[ "$dataset" == "dataset_name" ]] && continue
        dataset_map_ref["$dataset"]="$img_ext $mask_ext"
    done < "$2"
}

# Get arguments: src folder, data_dir, model, max jobs, and the dataset CSV file
src_dir="${1:-../src/python}"  # Default src directory
data_dir="${2:-../data}"  # Default data directory
model_name="$3"
MAX_JOBS="${4:-10}"
datasets_file="${5:-"datasets.csv"}"
# Optional: Capture the model arguments passed in as a string
model_arguments="${6:-}"

# Declare an associative array for dataset map
declare -A dataset_map
read_datasets_csv dataset_map "$datasets_file"

# Create the logs directory if it doesn't exist
mkdir -p logs

# Function to wait until resources are available
wait_for_resources() {
    while true; do
        local current_jobs=$(count_training_processes)
        echo "Current training processes: $current_jobs"
        
        if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
            break
        fi
        echo "Waiting for resources to become available..."
        sleep 30  # Longer sleep time since training takes a while
    done
}

# Loop through each dataset found in data_dir that matches keys in dataset_map
for dataset_name in "${!dataset_map[@]}"; do
    dataset_path="$data_dir/$dataset_name"
    
    if [ -d "$dataset_path" ]; then
        echo "Found dataset: $dataset_name"

        # Extract img_ext and mask_ext from the map
        read -r img_ext mask_ext <<< "${dataset_map[$dataset_name]}"
        
        # Wait until we have available resources
        wait_for_resources

        # Try to find least utilized GPU
        if command -v nvidia-smi &> /dev/null; then
            export CUDA_VISIBLE_DEVICES=$(get_free_gpu)
            echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
        fi

        # Start training the specified model for this dataset in the background
        echo "Starting training for $model_name on dataset $dataset_name"
        {
            python3 "$src_dir/main.py" \
                --model "$model_name" \
                --data_dir "$data_dir" \
                --dataset_name "$dataset_name" \
                --img_ext "$img_ext" \
                --mask_ext "$mask_ext" \
                ${model_arguments} \
                --verbose False
        } > "logs/${model_name}-${dataset_name}.log" 2>&1 &

        pid=$!
        echo "Launched process for $model_name on $dataset_name with parent PID: $pid"

        # Give some time for the process to spawn its children before starting the next one
        sleep 10
    else
        echo "Dataset $dataset_name not found in data directory."
    fi
done

echo "All training jobs for available datasets have been launched. Monitor progress in the logs directory."
echo "You can check GPU usage with: nvidia-smi"
echo "You can check process status with: ps aux | grep main.py"

# Wait for all background processes to complete
wait
