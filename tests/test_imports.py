from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CRITICAL_MODULES = [
    'src.data.dataset',
    'src.data.transforms',
    'src.training.losses',
    'src.evaluation.metrics',
    'src.serving.app',
    'src.serving.api',
    'src.serving.scene_api',
    'src.demo.streamlit_app',
    'src.monitoring.utils',
    'src.monitoring.build_reference_dataset',
    'src.monitoring.collect_inference_logs',
    'src.monitoring.evidently_data_drift',
    'src.monitoring.evidently_data_quality',
    'src.monitoring.evidently_performance_simulated',
]

ESSENTIAL_FILES = [
    'configs/data.yaml',
    'configs/serving.yaml',
    'configs/monitoring.yaml',
    'requirements.txt',
    'requirements-ci.txt',
    'src/serving/api.py',
    'src/serving/app.py',
    'src/demo/streamlit_app.py',
    'Dockerfile.api',
    'Dockerfile.streamlit',
    'docker-compose.yml',
    'docs/ci_docker.md',
]


@pytest.mark.parametrize('module_name', CRITICAL_MODULES)
def test_critical_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


@pytest.mark.parametrize('relative_path', ESSENTIAL_FILES)
def test_essential_files_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).exists(), f'Missing required file: {relative_path}'
