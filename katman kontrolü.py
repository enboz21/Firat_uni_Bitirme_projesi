"""Compatibility entry point for the reorganized band-inspection utility."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parent / "scripts" / "utilities" / "inspect_bands.py", run_name="__main__")
