"""Monitoring utilities for post-deployment MLOps workflows."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MONITORING_CONFIG_PATH = PROJECT_ROOT / "configs" / "monitoring.yaml"
