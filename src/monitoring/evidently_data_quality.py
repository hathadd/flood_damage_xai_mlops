from __future__ import annotations

import argparse
from pathlib import Path

from src.monitoring.utils import (
    choose_feature_columns,
    import_evidently_components,
    load_monitoring_config,
    load_table,
    report_to_dict,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Evidently data quality report.")
    parser.add_argument("--config-path", type=str, default="configs/monitoring.yaml")
    parser.add_argument("--reference-path", type=str, default=None)
    parser.add_argument("--current-path", type=str, default=None)
    parser.add_argument("--html-output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_monitoring_config(args.config_path)
    components = import_evidently_components()

    reference_path = args.reference_path or config["paths"]["reference_data_path"]
    current_path = args.current_path or config["paths"]["current_data_path"]
    html_output = args.html_output or f"{config['paths']['reports_output_dir']}/data_quality/data_quality_report.html"
    json_output = args.json_output or f"{config['paths']['reports_output_dir']}/data_quality/data_quality_report.json"

    reference_df = load_table(reference_path)
    current_df = load_table(current_path)
    feature_columns = choose_feature_columns(
        reference_df=reference_df,
        current_df=current_df,
        ignore_columns=config.get("reporting", {}).get("ignore_columns", []),
        preferred_columns=config.get("reporting", {}).get("preferred_feature_columns", []),
    )
    if not feature_columns:
        raise ValueError("No common columns available for data quality analysis.")

    report = components["Report"](metrics=[components["DataSummaryPreset"]()])
    snapshot = report.run(reference_data=reference_df[feature_columns].copy(), current_data=current_df[feature_columns].copy())
    html_path = Path(html_output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    json_path = write_json(
        {
            "reference_path": reference_path,
            "current_path": current_path,
            "feature_columns": feature_columns,
            "report": report_to_dict(snapshot),
        },
        json_output,
    )
    print(f"Data quality report saved to {html_path.as_posix()}")
    print(f"Data quality JSON saved to {json_path.as_posix()}")


if __name__ == "__main__":
    main()
