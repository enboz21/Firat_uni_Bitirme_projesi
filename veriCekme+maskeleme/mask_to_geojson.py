"""Compatibility entry point for the reorganized vector-export script."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parents[1] / "scripts" / "data_pipeline" / "mask_to_geojson.py", run_name="__main__")
