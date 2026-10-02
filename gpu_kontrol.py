"""Compatibility entry point for the reorganized GPU utility."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parent / "scripts" / "utilities" / "check_gpu.py", run_name="__main__")
