"""Compatibility entry point for the reorganized mask-generation script."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parents[1] / "scripts" / "data_pipeline" / "generate_ndwi_mask.py", run_name="__main__")
