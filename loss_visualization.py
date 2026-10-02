"""Compatibility entry point for the reorganized loss-plot script."""

from pathlib import Path
from runpy import run_path

run_path(Path(__file__).parent / "scripts" / "analysis" / "plot_training_loss.py", run_name="__main__")
