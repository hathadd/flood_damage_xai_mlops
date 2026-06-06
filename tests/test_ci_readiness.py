from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import build_reference_dataset, collect_inference_logs, evidently_data_drift, evidently_data_quality
from src.serving.app import app


def test_fastapi_app_is_importable_and_has_core_routes() -> None:
    assert isinstance(app, FastAPI)

    route_paths = {route.path for route in app.routes}
    assert "/" in route_paths
    assert "/health" in route_paths
    assert "/model-info" in route_paths
    assert "/predict" in route_paths
    assert "/predict-scene" in route_paths
    assert "/explain-building" in route_paths


def test_monitoring_scripts_expose_entrypoints() -> None:
    for module in (
        build_reference_dataset,
        collect_inference_logs,
        evidently_data_quality,
        evidently_data_drift,
    ):
        assert hasattr(module, "main")
        assert hasattr(module, "parse_args")


def test_docker_readiness_files_exist() -> None:
    for relative_path in (
        "Dockerfile.api",
        "Dockerfile.streamlit",
        "docker-compose.yml",
        "docs/ci_docker.md",
    ):
        assert (PROJECT_ROOT / relative_path).exists(), f"Missing Docker/CI artifact: {relative_path}"
