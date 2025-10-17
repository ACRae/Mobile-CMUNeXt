import argparse
from datetime import datetime
from glob import glob
import os

from lib.LatexHelper import LatexHelper
import yaml


class Parser:
    @staticmethod
    def _load_yaml(file_path) -> dict:
        with open(file_path) as file:
            return yaml.safe_load(file)

    @staticmethod
    def _create_df_structure(sel_datasets):
        df_data = {}
        for dataset in sel_datasets:
            df_data[dataset] = {"Model": []}
        return df_data

    @staticmethod
    def _update_df_list(dict1: dict, dict2: dict):
        for key, value in dict2.items():
            if key in dict1:
                dict1[key].append(value)
            else:
                dict1[key] = [value]

    @staticmethod
    def parse_attributes(data_path):
        """
        Looks for:
            * models
            * datasets
            * profiles
            * metrics
        """
        models = set()
        datasets = set()
        profiles = set()
        metrics = set()

        for model_path in glob(f"{data_path}/*/"):
            models.add(os.path.basename(os.path.normpath(model_path)))

            for dataset_path in glob(f"{os.path.normpath(model_path)}/*/"):
                datasets.add(os.path.basename(os.path.normpath(dataset_path)))
                timestamp_dirs = sorted(glob(f"{dataset_path}/*/"), reverse=True)

                for ts_dir in timestamp_dirs:
                    best_file = os.path.join(ts_dir, "metrics.yml")
                    profile_file = os.path.join(ts_dir, "profile.yml")

                    if os.path.exists(best_file) and os.path.exists(profile_file):
                        best_metrics_data = Parser._load_yaml(best_file)
                        profile_data = Parser._load_yaml(profile_file)
                        if best_metrics_data:
                            metrics.update(best_metrics_data.keys())
                        if profile_data:
                            profiles.update(profile_data.keys())
                        break

        return sorted(models), sorted(datasets), sorted(profiles), sorted(metrics)

    @staticmethod
    def parse(data_path, models: list, datasets: list, profiles: list, metrics: list):
        latex_data = {
            "Columns": len(metrics) * len(datasets) + len(profiles) + 1,  # 1 is for model name
            "Data": {},
            "Lables": [],
        }
        df_data = Parser._create_df_structure(datasets)

        for model in models:
            latex_data.setdefault(model, {})
            results = {"Profile": {}, "Metrics": {}}
            for dataset in datasets:
                timestamps_path = os.path.join(data_path, model, dataset)
                timestamps = sorted(glob(f"{timestamps_path}/*/"), reverse=True)
                if timestamps:
                    latest_ts = timestamps[0]
                    profile_path = os.path.join(latest_ts, "profile.yml")
                    metrics_path = os.path.join(latest_ts, "metrics.yml")
                    if os.path.exists(profile_path) and os.path.exists(metrics_path):
                        profiles_file = Parser._load_yaml(profile_path)
                        metrics_dict = Parser._load_yaml(metrics_path)
                        if len(metrics_dict) > 0 and len(profiles_file) > 0:
                            df_data[dataset]["Model"].append(model)
                            if profiles:
                                filtered_profiles = {key: profiles_file[key] for key in profiles}
                                results["Profile"] = filtered_profiles
                                Parser._update_df_list(df_data[dataset], filtered_profiles)
                            if metrics:
                                filtered_metrics = {key: metrics_dict[key] for key in metrics}
                                results["Metrics"][dataset] = filtered_metrics
                                Parser._update_df_list(df_data[dataset], filtered_metrics)

            latex_data["Data"][model] = results

        return latex_data, df_data

    @staticmethod
    def _format_profile(profile: str, value: float):
        """
        Hard coded function to convert profiles
        """
        if profile == "MACS":
            return round(value / 1_000_000_000, 2)
        if profile == "PARAMS":
            return round(value / 1_000_000, 2)
        return value

    @staticmethod
    def _format_metric(value):
        """
        Hard coded function to format metrics
        """
        if isinstance(value, float | int) and value < 1:
            return round(value * 100, 2)
        return value

    @staticmethod
    def export_latex(latex_data: dict, export_dir="./"):
        export_file = os.path.join(
            export_dir, "export_" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss") + ".tex"
        )
        DELIMITER = ", "
        with open(export_file, "w") as file:
            write_data = [LatexHelper.Tabular.open_tabular(latex_data["Columns"])]
            for model, model_data in latex_data["Data"].items():
                write_data.append(LatexHelper.normalize_latexstring(model))
                write_data.append("\n")

                profiles = model_data["Profile"]
                metrics = model_data["Metrics"]
                if profiles:
                    profiles_comment = ""
                    profiles_list = []
                    for idx, profiles_data in enumerate(profiles.items()):
                        profile, value = profiles_data
                        profiles_list.append(Parser._format_profile(profile, value))
                        profiles_comment += profile
                        if idx != len(profiles) - 1:
                            profiles_comment += DELIMITER

                    write_data.append(LatexHelper.write_entries(profiles_list, profiles_comment))
                    write_data.append("\n")

                if metrics:
                    for dataset, dataset_data in metrics.items():
                        dataset_comment = dataset
                        metrics_str = " ("
                        metrics_list = []
                        for idx, metrics_data in enumerate(dataset_data.items()):
                            metric, value = metrics_data
                            metrics_list.append(Parser._format_metric(value))
                            metrics_str += metric
                            if idx != len(dataset_data) - 1:
                                metrics_str += DELIMITER

                        metrics_str += ")"
                        dataset_comment += metrics_str
                        write_data.append(LatexHelper.write_entries(metrics_list, dataset_comment))
                        write_data.append("\n")

                write_data.append("\\\\\n")  # break entry
                write_data.append("\n")

            write_data.append(LatexHelper.Tabular.close_tabular())
            file.writelines(write_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the models directory")
    args = parser.parse_args()
    data_path = args.data
    models, datasets, profiles, metrics = Parser.parse_attributes(data_path)
    latex_data, df_data = Parser.parse(data_path, models, datasets, profiles[:-1], [])
    Parser.export_latex(latex_data)
