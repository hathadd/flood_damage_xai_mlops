from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.monitoring.utils import import_evidently_components, load_monitoring_config, load_table, report_to_dict, write_json

MESSAGE_NO_LABELS = "Performance monitoring requires true labels. Current inference data has no true_label column."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Evidently classification report when true labels exist.")
    parser.add_argument("--config-path", type=str, default="configs/monitoring.yaml")
    parser.add_argument("--current-path", type=str, default=None)
    parser.add_argument("--html-output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_monitoring_config(args.config_path)
    current_path = args.current_path or config["paths"]["current_data_path"]
    current_df = load_table(current_path)

    target_column = config.get("reporting", {}).get("target_column", "true_label")
    prediction_column = config.get("reporting", {}).get("prediction_column", "predicted_label")
    if target_column not in current_df.columns or current_df[target_column].dropna().empty:
        print(MESSAGE_NO_LABELS)
        return
    if prediction_column not in current_df.columns:
        raise ValueError(f"Missing prediction column '{prediction_column}' in current data.")

    evaluation_df = current_df.dropna(subset=[target_column, prediction_column]).copy()

    components = import_evidently_components()
    from evidently import DataDefinition, Dataset, MulticlassClassification

    classification = MulticlassClassification(
        target=target_column,
        prediction_labels=prediction_column,
    )
    dataset = Dataset.from_pandas(
        evaluation_df,
        data_definition=DataDefinition(classification=[classification]),
    )

    report = components["Report"](metrics=[components["ClassificationPreset"]()])
    snapshot = report.run(current_data=dataset)

    html_output = args.html_output or f"{config['paths']['reports_output_dir']}/performance_simulated/performance_report.html"
    json_output = args.json_output or f"{config['paths']['reports_output_dir']}/performance_simulated/performance_report.json"
    html_path = Path(html_output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    metrics_payload = {
        "accuracy": float(accuracy_score(evaluation_df[target_column], evaluation_df[prediction_column])),
        "macro_f1": float(f1_score(evaluation_df[target_column], evaluation_df[prediction_column], average="macro")),
        "macro_precision": float(
            precision_score(evaluation_df[target_column], evaluation_df[prediction_column], average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(evaluation_df[target_column], evaluation_df[prediction_column], average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            evaluation_df[target_column],
            evaluation_df[prediction_column],
            zero_division=0,
            output_dict=True,
        ),
        "report": report_to_dict(snapshot),
    }
    json_path = write_json(metrics_payload, json_output)
    print(f"Performance monitoring report saved to {html_path.as_posix()}")
    print(f"Performance monitoring JSON saved to {json_path.as_posix()}")


if __name__ == "__main__":
    main()
