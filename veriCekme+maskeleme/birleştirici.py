"""Compatibility entry point for the reorganized band-merge script."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parents[1] / "scripts" / "data_pipeline" / "merge_bands.py", run_name="__main__")
