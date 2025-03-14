"""Dataset viewer module for visualizing datasets.

This module provides a user interface for visualizing various types of datasets,
including image pairs, point clouds, and change detection datasets.

Main Components:
- DatasetViewer: Main class for launching the dataset visualization UI
- CLI: Command-line interface for running the viewer

Example usage:
    from data.viewer import DatasetViewer
    viewer = DatasetViewer()
    viewer.run()
"""

from data.viewer.viewer import DatasetViewer

__all__ = ['DatasetViewer']
