#!/bin/bash

set -euo pipefail


# Handle input arguments
src_dir="${1:-../src/python}"
data_dir="${2:-../data}"
MAX_JOBS="${3:-10}"
networks_file="${4:-networks.csv}"
datasets_file="${5:-datasets.csv}"




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
        if (( utilizations[i] < min_util )); then
            min_util=${utilizations[i]}
            min_idx=$i
        fi
    done

    echo "$min_idx"
}




# Function to count active training processes
count_training_processes() {
    pgrep -f "python.*main.py" | wc -l
}

# Wait until resources are available
wait_for_resources() {
    while true; do
        local current_jobs=$(count_training_processes)
        echo "Current training processes: $current_jobs"

        if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
            break
        fi
        echo "Waiting for resources to become available..."
        sleep 30
    done
}




# Validate CSV input files
if [[ ! -f "$networks_file" ]]; then
    echo "Error: Networks file '$networks_file' not found."
    exit 1
fi

if [[ ! -f "$datasets_file" ]]; then
    echo "Error: Datasets file '$datasets_file' not found."
    exit 1
fi

# Function to read the models and their arguments from the CSV file
read_models_csv() {
    declare -n model_names_ref=$1
    declare -n model_args_ref=$2
    local csv_file="$3"

    while IFS=',' read -r model_name arguments || [ -n "$model_name" ]; do
        [[ "$model_name" == "model_name" || -z "$model_name" ]] && continue
        model_names_ref+=("$model_name")
        model_args_ref+=("$arguments")
    done < "$csv_file"
}

# Declare arrays to hold model names and arguments
declare -a model_names
declare -a model_args
read_models_csv model_names model_args "$networks_file"

# Function to read datasets from CSV file
read_datasets_csv() {
    declare -n dataset_map_ref=$1
    local csv_file="$2"
    while IFS=',' read -r dataset img_ext mask_ext || [ -n "$dataset" ]; do
        [[ "$dataset" == "dataset_name" || -z "$dataset" ]] && continue
        dataset_map_ref["$dataset"]="$img_ext $mask_ext"
    done < "$csv_file"
}

# Declare an associative array for dataset map
declare -A dataset_map
read_datasets_csv dataset_map "$datasets_file"

mkdir -p logs

# Loop through all models using index
for i in "${!model_names[@]}"; do
    model="${model_names[$i]}"
    args="${model_args[$i]}"

    echo "Starting training for model: $model with args: $args"

    for dataset_name in "${!dataset_map[@]}"; do
        dataset_path="$data_dir/$dataset_name"

        if [ -d "$dataset_path" ]; then
            echo "Found dataset: $dataset_name"
            read -r img_ext mask_ext <<< "${dataset_map[$dataset_name]}"

            wait_for_resources

            if command -v nvidia-smi &> /dev/null; then
                export CUDA_VISIBLE_DEVICES=$(get_free_gpu)
                echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
            fi

            log_file="logs/${model}-${dataset_name}-${i}.log"
            echo "Launching training (log: $log_file)..."

            {
                python3 "$src_dir/main.py" \
                    --model "$model" \
                    --data_dir "$data_dir" \
                    --dataset_name "$dataset_name" \
                    --img_ext "$img_ext" \
                    --mask_ext "$mask_ext" \
                    $args \
                    --verbose False
            } > "$log_file" 2>&1 &

            pid=$!
            echo "Launched PID: $pid for $model on $dataset_name"
            sleep 10
        else
            echo "Dataset $dataset_name not found in data directory."
        fi
    done
    # Wait for all background processes to finish
    wait
done



echo "All model training jobs have been launched and completed."
