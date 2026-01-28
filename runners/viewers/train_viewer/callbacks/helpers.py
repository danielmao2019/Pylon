"""Helper functions for training losses callbacks."""

from pathlib import Path
from typing import List

from dash import dcc, html

from runners.viewers.train_viewer.backend.read_losses import read_losses
from runners.viewers.train_viewer.backend.visualize_losses import visualize_losses

LOG_DIRS_FILEPATH = Path(__file__).resolve().parent.parent / "log_dirs.txt"


def load_log_dirs(filepath: Path) -> List[str]:
    # Input validations
    assert isinstance(filepath, Path), f"filepath must be Path, got {type(filepath)}"
    assert filepath.is_file(), f"Log directories file not found: {filepath}"

    log_dirs = [
        line.strip()
        for line in filepath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert log_dirs, "No log directories found in log_dirs.txt"
    return log_dirs


def build_loss_plots(log_dirs: List[str], smoothing_window: int) -> List[html.Div]:
    # Input validations
    assert isinstance(log_dirs, list) and log_dirs, "log_dirs must be a non-empty list"
    assert all(
        isinstance(log_dir, str) and log_dir for log_dir in log_dirs
    ), "log_dirs must contain non-empty strings"
    assert isinstance(smoothing_window, int), "smoothing_window must be an integer"

    plots: List[html.Div] = []
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        assert log_path.exists(), f"Log directory does not exist: {log_dir}"

        losses = read_losses(log_dir)
        fig = visualize_losses(
            losses=losses,
            smoothing_window=smoothing_window,
            title=log_dir,
        )
        plots.append(
            dcc.Graph(
                figure=fig,
                style={"height": "600px", "marginBottom": "20px"},
            )
        )

    return plots


def build_smoothing_info(smoothing_window: int) -> str:
    # Input validations
    assert isinstance(smoothing_window, int), "smoothing_window must be an integer"

    suffix = "(no smoothing)" if smoothing_window == 1 else "iterations"
    return f"Window size: {smoothing_window} {suffix}"
