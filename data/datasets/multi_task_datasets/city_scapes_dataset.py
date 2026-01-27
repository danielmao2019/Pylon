import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from dash import html

import utils
from data.datasets.multi_task_datasets.base_multi_task_dataset import (
    BaseMultiTaskDataset,
)
from data.viewer.utils.atomic_displays import (
    create_depth_display,
    create_image_display,
    create_instance_surrogate_display,
    create_segmentation_display,
    get_depth_display_stats,
    get_image_display_stats,
    get_instance_surrogate_display_stats,
    get_segmentation_display_stats,
)
from data.viewer.utils.display_utils import (
    ParallelFigureCreator,
    create_figure_grid,
    create_standard_datapoint_layout,
    create_statistics_display,
)


class CityScapesDataset(BaseMultiTaskDataset):
    __doc__ = r"""Reference:

    For implementation of dataset class:
        https://github.com/SamsungLabs/MTL/blob/master/code/data/datasets/cityscapes.py
    For definition of labels:
        https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    Download:
        images: https://www.cityscapes-dataset.com/file-handling/?packageID=3
        segmentation labels: https://www.cityscapes-dataset.com/file-handling/?packageID=1
        disparity labels: https://www.cityscapes-dataset.com/file-handling/?packageID=7

    Used in:
        Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (https://arxiv.org/pdf/1705.07115.pdf)
        Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
        Conflict-Averse Gradient Descent for Multi-task Learning (https://arxiv.org/pdf/2110.14048.pdf)
        FAMO: Fast Adaptive Multitask Optimization (https://arxiv.org/pdf/2306.03792.pdf)
        Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
        Multi-Task Learning as a Bargaining Game (https://arxiv.org/pdf/2202.01017.pdf)
        Multi-Task Learning as Multi-Objective Optimization (https://arxiv.org/pdf/1810.04650.pdf)
        Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
        Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
    """

    SPLIT_OPTIONS = ['train', 'val']
    DATASET_SIZE = {
        'train': 2975 - 9,
        'val': 500 - 7,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['depth_estimation', 'semantic_segmentation', 'instance_segmentation']
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    IGNORE_INDEX = 250
    INSTANCE_VOID: List[int] = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    SEMANTIC_VOID: List[int] = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    VALID_CLASSES: List[int] = [
        7,
        8,
        11,
        12,
        13,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
    ]
    CLASS_MAP_F = dict(zip(VALID_CLASSES, range(19), strict=True))
    CLASS_MAP_C = {
        7: 0,  # flat
        8: 0,  # flat
        11: 1,  # construction
        12: 1,  # construction
        13: 1,  # construction
        17: 5,  # object
        19: 5,  # object
        20: 5,  # object
        21: 2,  # nature
        22: 2,  # nature
        23: 4,  # sky
        24: 6,  # human
        25: 6,  # human
        26: 3,  # vehicle
        27: 3,  # vehicle
        28: 3,  # vehicle
        31: 3,  # vehicle
        32: 3,  # vehicle
        33: 3,  # vehicle
    }
    NUM_CLASSES_F = 19
    NUM_CLASSES_C = 7

    IMAGE_MEAN = [123.675, 116.28, 103.53]
    DEPTH_STD = 2729.0680031169923
    DEPTH_MEAN = 0.0

    REMOVE_INDICES = {
        'train': [253, 926, 931, 1849, 1946, 1993, 2051, 2054, 2778],
        'val': [284, 285, 286, 288, 299, 307, 312],
    }

    CLASS_COLORS = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(self, semantic_granularity: str = 'coarse', *args, **kwargs) -> None:
        assert isinstance(semantic_granularity, str), f"{type(semantic_granularity)=}"
        assert semantic_granularity in ['fine', 'coarse'], f"{semantic_granularity=}"
        self.semantic_granularity = semantic_granularity
        if semantic_granularity == 'fine':
            self.CLASS_MAP = self.CLASS_MAP_F
            self.NUM_CLASSES = self.NUM_CLASSES_F
        else:
            self.CLASS_MAP = self.CLASS_MAP_C
            self.NUM_CLASSES = self.NUM_CLASSES_C
        super(CityScapesDataset, self).__init__(*args, **kwargs)

    def _init_annotations(self) -> None:
        # initialize image filepaths
        image_root = os.path.join(self.data_root, "leftImg8bit", self.split)
        image_paths: List[str] = sorted(
            glob.glob(os.path.join(image_root, "**", "*.png"))
        )
        image_postfix = "leftImg8bit.png"
        # depth estimation labels
        depth_root = os.path.join(self.data_root, "disparity", self.split)
        depth_paths = [
            os.path.join(
                depth_root,
                fp.split(os.sep)[-2],
                os.path.basename(fp)[: -len(image_postfix)] + "disparity.png",
            )
            for fp in image_paths
        ]
        # semantic and instance segmentation labels
        segmentation_root = os.path.join(self.data_root, "gtFine", self.split)
        semantic_paths = [
            os.path.join(
                segmentation_root,
                fp.split(os.sep)[-2],
                os.path.basename(fp)[: -len(image_postfix)] + "gtFine_labelIds.png",
            )
            for fp in image_paths
        ]
        instance_paths = [
            os.path.join(
                segmentation_root,
                fp.split(os.sep)[-2],
                os.path.basename(fp)[: -len(image_postfix)] + "gtFine_instanceIds.png",
            )
            for fp in image_paths
        ]
        # construct annotations
        self.annotations: List[Dict[str, str]] = [
            {
                'image': image_paths[idx],
                'depth': depth_paths[idx],
                'semantic': semantic_paths[idx],
                'instance': instance_paths[idx],
            }
            for idx in range(len(image_paths))
        ]
        self._filter_dataset_(remove_indices=self.REMOVE_INDICES[self.split])

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()

        # Include semantic granularity parameter
        version_dict['semantic_granularity'] = self.semantic_granularity

        return version_dict

    def _filter_dataset_(
        self, remove_indices: List[int], cache: Optional[bool] = True
    ) -> None:
        if not cache:
            remove_indices = []
            for idx in range(len(self.annotations)):
                depth_estimation = self._get_depth_label_(idx)['depth_estimation']
                segmentation = self._get_segmentation_labels_(idx)
                semantic_segmentation = segmentation['semantic_segmentation']
                instance_segmentation = segmentation['instance_segmentation']
                if torch.all(depth_estimation == 0):
                    remove_indices.append(idx)
                    continue
                if torch.all(semantic_segmentation == self.IGNORE_INDEX):
                    remove_indices.append(idx)
                    continue
                if torch.all(instance_segmentation == self.IGNORE_INDEX):
                    remove_indices.append(idx)
                    continue
        self.annotations = [
            self.annotations[idx]
            for idx in range(len(self.annotations))
            if idx not in remove_indices
        ]

    # ====================================================================================================
    # load methods
    # ====================================================================================================

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Any],
    ]:
        r"""
        Returns:
            Dict[str, Dict[str, Any]]: data point at index `idx` in the following format:
            inputs = {
                'image': float32 tensor of shape (3, H, W).
            }
            labels = {
                'depth_estimation': float32 tensor of shape (H, W). (if selected)
                'semantic_segmentation': int64 tensor of shape (H, W). (if selected)
                'instance_segmentation': float32 tensor of shape (2, H, W). (if selected)
            }
            meta_info = {
                'image_filepath': str object for image file path.
                'image_resolution': 2-tuple object for image height and width.
            }
        """
        inputs = self._get_image_(idx)
        labels = {}

        # Only load selected labels to avoid unnecessary disk I/O
        needs_depth = 'depth_estimation' in self.selected_labels
        needs_segmentation = (
            'semantic_segmentation' in self.selected_labels
            or 'instance_segmentation' in self.selected_labels
        )

        if needs_depth:
            labels.update(self._get_depth_label_(idx))

        if needs_segmentation:
            # This method loads both semantic and instance segmentation
            seg_labels = self._get_segmentation_labels_(idx)
            # Only include the selected ones
            for key, value in seg_labels.items():
                if key in self.selected_labels:
                    labels[key] = value

        meta_info = {
            'image_filepath': os.path.relpath(
                path=self.annotations[idx]['image'], start=self.data_root
            ),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info

    def _get_image_(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'image': utils.io.image.load_image(
                filepath=self.annotations[idx]['image'],
                dtype=torch.float32,
                sub=self.IMAGE_MEAN[::-1],
                div=255.0,
            )
        }

    def _get_depth_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'depth_estimation': utils.io.image.load_image(
                filepath=self.annotations[idx]['depth'],
                dtype=torch.float32,
                sub=None,
                div=self.DEPTH_STD,
            )
        }

    def _get_segmentation_labels_(self, idx: int) -> Dict[str, torch.Tensor]:
        # get semantic segmentation labels
        semantic = utils.io.image.load_image(
            filepath=self.annotations[idx]['semantic'],
            dtype=torch.int64,
        )
        for void_class in self.SEMANTIC_VOID:
            semantic[semantic == void_class] = self.IGNORE_INDEX
        for valid in self.VALID_CLASSES:
            semantic[semantic == valid] = self.CLASS_MAP[valid]
        # get instance segmentation labels
        instance = utils.io.image.load_image(
            filepath=self.annotations[idx]['instance'],
            dtype=torch.int64,
        )
        instance[semantic == self.IGNORE_INDEX] = self.IGNORE_INDEX
        for void_class in self.INSTANCE_VOID:
            instance[instance == void_class] = self.IGNORE_INDEX
        instance[instance == 0] = self.IGNORE_INDEX
        assert len(instance.shape) == 2, f"{instance.shape=}"
        height, width = instance.shape
        ymap, xmap = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing='ij'
        )
        assert ymap.max() == height - 1, f"{ymap.max()=}, {height=}"
        assert xmap.max() == width - 1, f"{xmap.max()=}, {width=}"
        ymap = ymap.type(torch.float32) / ymap.max()
        xmap = xmap.type(torch.float32) / xmap.max()
        instance_y = torch.ones_like(instance, dtype=torch.float32) * self.IGNORE_INDEX
        instance_x = torch.ones_like(instance, dtype=torch.float32) * self.IGNORE_INDEX
        assert instance_y.shape == ymap.shape == instance_x.shape == xmap.shape
        for instance_id in torch.unique(instance):
            if instance_id == self.IGNORE_INDEX:
                continue
            mask = instance == instance_id
            instance_y[mask] = ymap[mask] - torch.mean(ymap[mask])
            instance_x[mask] = xmap[mask] - torch.mean(xmap[mask])
        instance_surrogate = torch.stack([instance_y, instance_x], dim=0)
        # return result
        return {
            'semantic_segmentation': semantic,
            'instance_segmentation': instance_surrogate,
        }

    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None,
    ) -> 'html.Div':
        """Display CityScapes multi-task datapoint with all modalities.

        This method visualizes CityScapes tasks: RGB image, depth estimation,
        semantic segmentation, and instance segmentation.

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

        # Validate expected CityScapes data keys
        assert (
            'image' in inputs
        ), f"inputs missing 'image', got keys: {list(inputs.keys())}"

        # Create figure tasks and statistics conditionally based on available labels
        figure_tasks = []
        stats_data = []
        stats_titles = []

        # Always include RGB image
        figure_tasks.append(
            lambda: create_image_display(image=inputs['image'], title="RGB Image")
        )
        stats_data.append(get_image_display_stats(inputs['image']))
        stats_titles.append("RGB Image Statistics")

        # Conditionally add depth estimation
        if 'depth_estimation' in labels:
            figure_tasks.append(
                lambda: create_depth_display(
                    depth=labels['depth_estimation'], title="Depth Estimation"
                )
            )
            stats_data.append(get_depth_display_stats(labels['depth_estimation']))
            stats_titles.append("Depth Statistics")

        # Conditionally add semantic segmentation
        if 'semantic_segmentation' in labels:
            figure_tasks.append(
                lambda: create_segmentation_display(
                    segmentation=labels['semantic_segmentation'],
                    title="Semantic Segmentation",
                    class_labels=class_labels,
                )
            )
            stats_data.append(
                get_segmentation_display_stats(labels['semantic_segmentation'])
            )
            stats_titles.append("Semantic Segmentation Statistics")

        # Conditionally add instance segmentation
        if 'instance_segmentation' in labels:
            figure_tasks.append(
                lambda: create_instance_surrogate_display(
                    instance_surrogate=labels['instance_segmentation'],
                    title="Instance Segmentation",
                    ignore_value=self.IGNORE_INDEX,
                )
            )
            stats_data.append(
                get_instance_surrogate_display_stats(
                    instance_surrogate=labels['instance_segmentation'],
                    ignore_index=self.IGNORE_INDEX,
                )
            )
            stats_titles.append("Instance Segmentation Statistics")

        # Create figures in parallel for better performance
        max_workers = min(len(figure_tasks), 4)  # Adjust based on number of tasks
        figure_creator = ParallelFigureCreator(
            max_workers=max_workers, enable_timing=False
        )
        figures = figure_creator.create_figures_parallel(figure_tasks)

        # Create grid layout (adjust based on number of figures)
        if len(figures) <= 2:
            width_style = "50%"
        else:
            width_style = "50%"  # 2x2 grid for 3-4 figures

        figure_components = create_figure_grid(
            figures=figures, width_style=width_style, height_style="400px"
        )

        stats_components = create_statistics_display(
            stats_data=stats_data, titles=stats_titles, width_style="25%"
        )

        # Use standard layout with all components
        return create_standard_datapoint_layout(
            figure_components=figure_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug'),
        )
