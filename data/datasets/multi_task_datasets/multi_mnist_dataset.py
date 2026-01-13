import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from dash import html
from PIL import Image

from data.datasets.multi_task_datasets.base_multi_task_dataset import (
    BaseMultiTaskDataset,
)
from data.viewer.utils.atomic_displays import (
    create_image_display,
    get_image_display_stats,
)
from data.viewer.utils.display_utils import (
    ParallelFigureCreator,
    create_figure_grid,
    create_standard_datapoint_layout,
    create_statistics_display,
)


class MultiMNISTDataset(BaseMultiTaskDataset):
    __doc__ = r"""
    Used in:
        Multi-Task Learning as Multi-Objective Optimization (https://arxiv.org/pdf/1810.04650.pdf)
        Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
    """

    SPLIT_OPTIONS = ['train', 'val']
    DATASET_SIZE = {
        'train': 60000,
        'val': 10000,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['left', 'right']
    SHA1SUM = None
    NUM_CLASSES = 10

    def _init_annotations(self) -> None:
        self.annotations = torchvision.datasets.MNIST(
            root=self.data_root,
            train=(self.split == 'train'),
            download=True,
        )
        return

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # MultiMNISTDataset uses deterministic random sampling based on index
        return version_dict

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Any],
    ]:
        random.seed(idx)
        indices = random.sample(population=range(len(self.annotations)), k=2)
        l_dp = self.annotations[indices[0]]
        r_dp = self.annotations[indices[1]]
        inputs = {
            'image': self._get_image(l_dp[0], r_dp[0]),
        }

        # Only load selected labels (but both come from same computation)
        labels = {}
        if 'left' in self.selected_labels:
            labels['left'] = torch.tensor(l_dp[1], dtype=torch.int64)
        if 'right' in self.selected_labels:
            labels['right'] = torch.tensor(r_dp[1], dtype=torch.int64)

        meta_info = {
            'image_resolution': inputs['image'].shape,
        }
        return inputs, labels, meta_info

    def _get_image(self, l_image: Image.Image, r_image: Image.Image) -> torch.Tensor:
        # Convert PIL Images to tensors using numpy and torch.from_numpy
        l_image = torch.from_numpy(np.array(l_image)).float() / 255.0
        r_image = torch.from_numpy(np.array(r_image)).float() / 255.0

        assert l_image.ndim == r_image.ndim == 2, f"{l_image.shape=}, {r_image.shape=}"
        assert l_image.shape == r_image.shape, f"{l_image.shape=}, {r_image.shape=}"
        left = torch.cat(
            [
                l_image,
                torch.zeros(
                    size=(r_image.shape[0], l_image.shape[1]),
                    dtype=torch.float32,
                    device=l_image.device,
                ),
            ],
            dim=0,
        )
        right = torch.cat(
            [
                torch.zeros(
                    size=(l_image.shape[0], r_image.shape[1]),
                    dtype=torch.float32,
                    device=r_image.device,
                ),
                r_image,
            ],
            dim=0,
        )
        image = torch.cat([left, right], dim=1)
        image = image.unsqueeze(0)
        assert image.shape == (
            1,
            l_image.shape[0] + r_image.shape[0],
            l_image.shape[1] + r_image.shape[1],
        ), f"{image.shape=}"
        return image

    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None,
    ) -> 'html.Div':
        """Display MultiMNIST multi-task datapoint with classification labels.

        This method visualizes MultiMNIST tasks: composite MNIST image and
        left/right digit classification labels.

        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info
            class_labels: Optional mapping from class indices to label names
            camera_state: Optional camera state (unused for 2D displays)
            settings_3d: Optional 3D settings (unused for 2D displays)

        Returns:
            HTML div containing the multi-task visualization

        Raises:
            AssertionError: If datapoint structure is invalid
        """
        # CRITICAL: Input validation with fail-fast assertions
        assert isinstance(
            datapoint, dict
        ), f"datapoint must be dict, got {type(datapoint)}"
        assert (
            'inputs' in datapoint
        ), f"datapoint missing 'inputs', got keys: {list(datapoint.keys())}"
        assert (
            'labels' in datapoint
        ), f"datapoint missing 'labels', got keys: {list(datapoint.keys())}"

        inputs = datapoint['inputs']
        labels = datapoint['labels']

        assert isinstance(inputs, dict), f"inputs must be dict, got {type(inputs)}"
        assert isinstance(labels, dict), f"labels must be dict, got {type(labels)}"

        # Validate expected MultiMNIST data keys
        assert (
            'image' in inputs
        ), f"inputs missing 'image', got keys: {list(inputs.keys())}"

        # Create figure task for the composite image
        figure_tasks = [
            lambda: create_image_display(
                image=inputs['image'], title="Composite MNIST Image (Left + Right)"
            )
        ]

        # Create figures in parallel for better performance
        figure_creator = ParallelFigureCreator(max_workers=1, enable_timing=False)
        figures = figure_creator.create_figures_parallel(figure_tasks)

        # Create grid layout for the single image
        figure_components = create_figure_grid(
            figures=figures, width_style="50%", height_style="400px"
        )

        # Create statistics for the image
        stats_data = [get_image_display_stats(inputs['image'])]

        stats_titles = ["Composite Image Statistics"]

        stats_components = create_statistics_display(
            stats_data=stats_data, titles=stats_titles, width_style="50%"
        )

        # Create classification labels display conditionally
        label_sections = []

        if 'left' in labels:
            left_digit = int(labels['left'].item())
            label_sections.append(
                html.Div(
                    [
                        html.H5(
                            "Left Digit: ",
                            style={'display': 'inline', 'margin-right': '5px'},
                        ),
                        html.Span(
                            str(left_digit),
                            style={
                                'font-size': '24px',
                                'font-weight': 'bold',
                                'color': '#2E86AB',
                            },
                        ),
                    ],
                    style={'margin-bottom': '10px'},
                )
            )

        if 'right' in labels:
            right_digit = int(labels['right'].item())
            label_sections.append(
                html.Div(
                    [
                        html.H5(
                            "Right Digit: ",
                            style={'display': 'inline', 'margin-right': '5px'},
                        ),
                        html.Span(
                            str(right_digit),
                            style={
                                'font-size': '24px',
                                'font-weight': 'bold',
                                'color': '#A23B72',
                            },
                        ),
                    ]
                )
            )

        # Create labels component only if there are labels to show
        if label_sections:
            labels_component = html.Div(
                [
                    html.H4("Classification Labels", style={'margin-bottom': '10px'}),
                    html.Div(label_sections),
                ],
                style={
                    'width': '50%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'padding': '20px',
                    'border': '1px solid #ddd',
                    'border-radius': '5px',
                    'margin': '10px',
                },
            )
        else:
            # If no labels selected, show info message
            labels_component = html.Div(
                [
                    html.H4("Classification Labels", style={'margin-bottom': '10px'}),
                    html.P(
                        "No classification labels selected for display.",
                        style={'color': '#666', 'font-style': 'italic'},
                    ),
                ],
                style={
                    'width': '50%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'padding': '20px',
                    'border': '1px solid #ddd',
                    'border-radius': '5px',
                    'margin': '10px',
                },
            )

        # Combine figure and labels components
        content_components = figure_components + [labels_component]

        # Use standard layout with all components
        return create_standard_datapoint_layout(
            figure_components=content_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug'),
        )
