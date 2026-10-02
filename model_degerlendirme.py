"""Compatibility entry point for the reorganized evaluation script."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parent / "scripts" / "evaluation" / "evaluate_model.py", run_name="__main__")
