import argparse

from lib.Parser import Parser
import pandas as pd
import questionary
from questionary import Choice, Style
from tabulate import tabulate


custom_style = Style(
    [
        ("separator", "fg:#6C6C6C bold"),
        ("qmark", "fg:#673ab7 bold"),
        ("question", ""),
        ("selected", "fg:#673AB7 bold"),
        ("pointer", "fg:#673ab7 bold"),
        ("highlighted", "fg:#673ab7 bold"),
        ("answer", "fg:#FF9D00 bold"),
        ("disabled", "fg:#858585 italic"),
    ]
)


def checkbox(name: str, choices: list):
    choices = [Choice(c) for c in choices]
    if len(choices) == 0:
        return []

    answer = questionary.checkbox(f"Select {name}:", choices=choices, style=custom_style).ask()

    selected_values = [item.value if hasattr(item, "value") else item for item in answer]

    return selected_values


def main():
    parser = argparse.ArgumentParser(description="Analyze model metrics and generate CSV reports")
    parser.add_argument("--data", required=True, help="Path to the models directory")
    parser.add_argument("--export", help="Path to export files (optional)", default=None)
    parser.add_argument("--latex", help="Export metrics in LaTeX table format", default=False)
    parser.add_argument("--n_round", help="Number of decimals to round", default=2)
    args = parser.parse_args()
    data_path = args.data

    models, datasets, profiles, metrics = Parser.parse_attributes(data_path)

    selected_models = checkbox("models", models)
    selected_datasets = checkbox("datasets", datasets)
    selected_profiles = checkbox("profiles", profiles)
    selected_metrics = checkbox("metrics", metrics)

    latex_data, df_data = Parser.parse(
        data_path, selected_models, selected_datasets, selected_profiles, selected_metrics
    )

    for dataset, results in df_data.items():
        df = pd.DataFrame(results)
        print(f"\nResults for {dataset}")
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))

    if args.latex is True:
        Parser.export_latex(latex_data)


if __name__ == "__main__":
    main()
