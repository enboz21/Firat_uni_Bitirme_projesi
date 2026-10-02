"""Compatibility entry point for the reorganized inference example."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parent / "scripts" / "inference" / "predict_water_mask.py", run_name="__main__")
