import argparse
from glob import glob
import os

import pandas as pd
import yaml


def load_yaml(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)


def parse_params(value):
    """Convert PARAMS value to millions (M), assuming value is in string format."""
    if isinstance(value, str):
        if value.endswith("M"):
            return float(value.replace("M", ""))  # Already in millions
        if value.endswith("K"):
            return float(value.replace("K", "")) / 1000  # Convert thousands to millions
    return float(value) / 1e6


def parse_macs(value):
    """Convert MACS value by stripping the 'G' unit and returning as a float."""
    if isinstance(value, str) and value.endswith("G"):
        return float(value.replace("G", ""))
    if value.endswith("M"):
        return float(value.replace("M", "")) / 1000  # Convert M to G
    return float(value)


def analyze_models(model_dir, dataset, export_path=None):
    # Gather the latest results for each model
    latest_results = {}

    # Traverse through all model directories to get the latest files
    for model_path in glob(f"{model_dir}/*/"):
        model_name = os.path.basename(os.path.normpath(model_path))
        dataset_dir = os.path.join(model_path, dataset)
        timestamp_dirs = sorted(glob(f"{dataset_dir}/*/"), reverse=True)

        for ts_dir in timestamp_dirs:
            best_file = os.path.join(ts_dir, "best_metrics.yml")
            config_file = os.path.join(ts_dir, "config.yml")

            # Only consider models with both `best.yml` and `config.yml`
            if os.path.exists(best_file) and os.path.exists(config_file):
                latest_results[model_name] = {
                    "best_metrics": load_yaml(best_file),
                    "config": load_yaml(config_file),
                }
                break  # Only the latest entry is needed

    # Prepare data for CSV
    model_data = []
    for model_name, data in latest_results.items():
        best_metrics = data["best_metrics"]
        config_data = data["config"]

        model_info = {
            "Model": model_name,
            "PARAMS (M) ↓": parse_params(config_data.get("PARAMS", "0")),
            "MACS (G) ↓": parse_macs(config_data.get("MACS", "0")),
            "iou ↑": best_metrics.get("val_iou", "N/A"),
            "dice ↑": best_metrics.get("val_dice", "N/A"),
            "F1 ↑": best_metrics.get("F1", "N/A"),
            "AAC ↑": best_metrics.get("AAC", "N/A"),
            "PC ↑": best_metrics.get("PC", "N/A"),
        }
        model_data.append(model_info)

    # Create DataFrame
    df = pd.DataFrame(model_data)

    # Sort tables
    df_complete = df.sort_values(["F1 ↑", "iou ↑"], ascending=[False, False])
    params_macs_df = df[["Model", "PARAMS (M) ↓", "MACS (G) ↓"]].sort_values("PARAMS (M) ↓")
    f1_iou_df = df[["Model", "F1 ↑", "iou ↑"]].sort_values(["F1 ↑", "iou ↑"], ascending=[False, False])

    # Export CSV files if export_path is provided
    if export_path is not None:
        os.makedirs(export_path, exist_ok=True)
        df_complete.to_csv(os.path.join(export_path, f"{dataset}_metrics_complete.csv"), index=False)
        params_macs_df.to_csv(os.path.join(export_path, f"{dataset}_params_vs_macs.csv"), index=False)
        f1_iou_df.to_csv(os.path.join(export_path, f"{dataset}_f1_vs_iou.csv"), index=False)

    # Print preview of the tables with dataset name
    print(f"\n{dataset} Complete Metrics Table (sorted by F1↑, then iou↑):")
    print(df_complete.to_string(index=False))
    print(f"\n{dataset} PARAMS↓ vs MACS↓ Table (sorted by PARAMS):")
    print(params_macs_df.to_string(index=False))
    print(f"\n{dataset} F1↑ vs iou↑ Table (sorted by F1, then iou):")
    print(f1_iou_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze model metrics and generate CSV reports")
    parser.add_argument("--model_dir", required=True, help="Path to the models directory")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--export_path", help="Path to export CSV files (optional)", default=None)

    args = parser.parse_args()
    analyze_models(args.model_dir, args.dataset, args.export_path)


if __name__ == "__main__":
    main()
