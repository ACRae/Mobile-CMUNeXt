#!/bin/bash

networks_file="${1:-"networks.csv"}"
datasets_file="${2:-"datasets.csv"}"



# Function to validate if a value is a number
is_number() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

# Function to read models from the networks.csv file
read_models_csv() {
    models=()
    while IFS="," read -r model_name _ || [ -n "$model_name" ]; do
        # Skip header line or empty lines
        [[ "$model_name" == "model_name" || -z "$model_name" ]] && continue
        models+=("$model_name")
    done < "$1"
}

# Read models from the networks.csv file
read_models_csv "$networks_file"


# Display the list of available scripts with clearer descriptions
echo "Which training option would you like to choose?"
select script in "Train a singular model" "Train multiple models"; do
    case $script in
        "Train a singular model")
            echo "You selected to train a singular model."
            break
            ;;
        "Train multiple models")
            echo "You selected to train multiple models."
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done



if [ "$script" == "Train a singular model" ]; then
    # Prompt for data directory (with default value)
    echo "Enter the data directory (default: ../data):"
    read -r data_dir
    data_dir="${data_dir:-../data}"

    # Prompt for maximum jobs (ensure it's a valid number)
    while true; do
        echo "Enter the maximum number of concurrent jobs (default: 10):"
        read -r max_jobs
        max_jobs="${max_jobs:-10}"
        
        if is_number "$max_jobs"; then
            break
        else
            echo "Please enter a valid number."
        fi
    done

    # Select model from the list of models
    echo "Select a model to train:"
    select model_name in "${models[@]}"; do
        if [ -n "$model_name" ]; then
            echo "You selected model: $model_name"
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done

    
    echo -e "\nEnter any custom parameters (i.e --description Hello) (Optional):"
    read -r custom_params
    custom_params="${custom_params}"

    command="./train/train_model.sh ../src/python $data_dir $model_name $max_jobs $datasets_file \"$custom_params\""
    
elif [ "$script" == "Train multiple models" ]; then
    # Prompt for data directory (with default value)
    echo "Enter the data directory (default: ../data):"
    read -r data_dir
    data_dir="${data_dir:-../data}"

    command="./train/train_models.sh ../src/python $data_dir 1 $networks_file $datasets_file"
fi


echo "Running: nohup $command > /dev/null 2>&1 &"
nohup bash -c "$command" > /dev/null 2>&1 &

echo "The command is running in the background. You can check the logs for progress."
