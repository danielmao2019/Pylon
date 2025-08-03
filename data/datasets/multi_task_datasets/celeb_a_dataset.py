from typing import Tuple, List, Dict, Any, Optional
import os
import torch
from data.datasets import BaseDataset
import utils


class CelebADataset(BaseDataset):
    __doc__ = r"""
    CelebA dataset for multi-task learning with facial attribute classification tasks.

    For detailed documentation, see: docs/datasets/multi_task/celeba.md
    """

    TOTAL_SIZE = 202599
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 162770,
        'val': 19867,
        'test': 19962,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['landmarks'] + [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
    ]
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(self, use_landmarks: Optional[bool] = False, **kwargs) -> None:
        assert type(use_landmarks) == bool, f"{type(use_landmarks)=}"
        self.use_landmarks = use_landmarks
        super(CelebADataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        image_filepaths = self._init_images_()
        landmark_labels = self._init_landmark_labels_(image_filepaths=image_filepaths)
        attribute_labels = self._init_attribute_labels_(image_filepaths=image_filepaths)
        self.annotations = list(zip(image_filepaths, landmark_labels, attribute_labels))

    def _init_images_(self) -> List[str]:
        # initialize
        split_enum = {0: 'train', 1: 'val', 2: 'test'}
        # images
        images_root = os.path.join(self.data_root, "images", "img_align_celeba")
        image_filepaths: List[str] = []
        with open(os.path.join(self.data_root, "list_eval_partition.txt"), mode='r') as f:
            lines = f.readlines()
            assert len(lines) == self.TOTAL_SIZE
            for idx in range(self.TOTAL_SIZE):
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{line[0]=}, {idx=}"
                filepath = os.path.join(images_root, line[0])
                assert os.path.isfile(filepath)
                if split_enum[int(line[1])] == self.split:
                    image_filepaths.append(filepath)
        return image_filepaths

    def _init_landmark_labels_(self, image_filepaths: List[str]) -> List[torch.Tensor]:
        if not self.use_landmarks:
            return [None] * len(image_filepaths)
        with open(os.path.join(self.data_root, "list_landmarks_align_celeba.txt"), mode='r') as f:
            lines = f.readlines()
            assert len(lines[0].strip().split()) == 10, f"{lines[0].strip().split()=}"
            lines = lines[1:]
            assert len(lines) == self.TOTAL_SIZE
            landmark_labels: List[torch.Tensor] = []
            for fp in image_filepaths:
                idx = int(os.path.basename(fp).split('.')[0]) - 1
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{fp=}, {line[0]=}, {idx=}"
                landmarks = torch.tensor(list(map(int, line[1:])), dtype=torch.uint8)
                assert landmarks.shape == (10,), f"{landmarks.shape=}"
                landmark_labels.append(landmarks)
        return landmark_labels

    def _init_attribute_labels_(self, image_filepaths: List[str]) -> List[Dict[str, torch.Tensor]]:
        with open(os.path.join(self.data_root, "list_attr_celeba.txt"), mode='r') as f:
            lines = f.readlines()
            assert set(lines[0].strip().split()) == set(self.LABEL_NAMES[1:])
            lines = lines[1:]
            assert len(lines) == self.TOTAL_SIZE
            attribute_labels: List[Dict[str, torch.Tensor]] = []
            for fp in image_filepaths:
                idx = int(os.path.basename(fp).split('.')[0]) - 1
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{fp=}, {line[0]=}, {idx=}"
                attributes: Dict[str, torch.Tensor] = dict(
                    (name, torch.tensor((1 if val == "1" else 0), dtype=torch.int64))
                    for name, val in zip(self.LABEL_NAMES[1:], line[1:])
                )
                attribute_labels.append(attributes)
        return attribute_labels

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # use_landmarks affects which labels are loaded
        version_dict.update({
            'use_landmarks': self.use_landmarks,
        })
        return version_dict

    # ====================================================================================================
    # load methods
    # ====================================================================================================

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {'image': utils.io.load_image(
            filepath=self.annotations[idx][0],
            dtype=torch.float32, sub=None, div=255.,
        )}
        labels = {}
        if self.use_landmarks:
            labels.update({'landmarks': self.annotations[idx][1]})
        labels.update(self.annotations[idx][2])
        meta_info = {
            'image_filepath': os.path.relpath(path=self.annotations[idx][0], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info

    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> 'html.Div':
        """Display CelebA multi-task datapoint with facial attributes.
        
        This method visualizes CelebA tasks: face image and all facial attribute
        classification labels (40 binary attributes).
        
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
        from dash import html
        from data.viewer.utils.atomic_displays import (
            create_image_display,
            get_image_display_stats
        )
        from data.viewer.utils.display_utils import (
            ParallelFigureCreator,
            create_figure_grid,
            create_standard_datapoint_layout,
            create_statistics_display
        )
        
        # CRITICAL: Input validation with fail-fast assertions
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        assert 'inputs' in datapoint, f"datapoint missing 'inputs', got keys: {list(datapoint.keys())}"
        assert 'labels' in datapoint, f"datapoint missing 'labels', got keys: {list(datapoint.keys())}"
        
        inputs = datapoint['inputs']
        labels = datapoint['labels']
        
        assert isinstance(inputs, dict), f"inputs must be dict, got {type(inputs)}"
        assert isinstance(labels, dict), f"labels must be dict, got {type(labels)}"
        
        # Validate expected CelebA data keys
        assert 'image' in inputs, f"inputs missing 'image', got keys: {list(inputs.keys())}"
        
        # Create figure task for the face image
        figure_tasks = [
            lambda: create_image_display(
                image=inputs['image'],
                title="Face Image"
            )
        ]
        
        # Create figures in parallel for better performance
        figure_creator = ParallelFigureCreator(max_workers=1, enable_timing=False)
        figures = figure_creator.create_figures_parallel(figure_tasks)
        
        # Create grid layout for the single image
        figure_components = create_figure_grid(
            figures=figures,
            width_style="40%",
            height_style="400px"
        )
        
        # Create statistics for the image
        stats_data = [
            get_image_display_stats(inputs['image'])
        ]
        
        stats_titles = [
            "Face Image Statistics"
        ]
        
        stats_components = create_statistics_display(
            stats_data=stats_data,
            titles=stats_titles,
            width_style="30%"
        )
        
        # Create facial attributes display
        # Group attributes into positive and negative for better visualization
        positive_attrs = []
        negative_attrs = []
        
        for attr_name in self.LABEL_NAMES[1:]:  # Skip 'landmarks' if it exists
            if attr_name in labels:
                attr_value = int(labels[attr_name].item())
                formatted_name = attr_name.replace('_', ' ')
                if attr_value == 1:
                    positive_attrs.append(formatted_name)
                else:
                    negative_attrs.append(formatted_name)
        
        # Create attributes component
        attributes_component = html.Div([
            html.H4("Facial Attributes", style={'margin-bottom': '15px'}),
            
            # Positive attributes
            html.Div([
                html.H5("Present Attributes:", style={
                    'color': '#2E86AB', 
                    'margin-bottom': '10px'
                }),
                html.Div([
                    html.Span(attr, style={
                        'display': 'inline-block',
                        'background-color': '#E8F4F8',
                        'color': '#2E86AB',
                        'padding': '3px 8px',
                        'margin': '2px',
                        'border-radius': '3px',
                        'font-size': '12px',
                        'border': '1px solid #2E86AB'
                    }) for attr in positive_attrs
                ])
            ], style={'margin-bottom': '20px'}),
            
            # Negative attributes  
            html.Div([
                html.H5("Absent Attributes:", style={
                    'color': '#A23B72',
                    'margin-bottom': '10px'
                }),
                html.Div([
                    html.Span(attr, style={
                        'display': 'inline-block',
                        'background-color': '#F8E8F0',
                        'color': '#A23B72',
                        'padding': '3px 8px',
                        'margin': '2px',
                        'border-radius': '3px',
                        'font-size': '12px',
                        'border': '1px solid #A23B72'
                    }) for attr in negative_attrs
                ])
            ])
        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'vertical-align': 'top',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'margin': '10px',
            'max-height': '400px',
            'overflow-y': 'auto'
        })
        
        # Combine all components
        content_components = figure_components + [attributes_component]
        
        # Use standard layout with all components
        return create_standard_datapoint_layout(
            figure_components=content_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug')
        )
