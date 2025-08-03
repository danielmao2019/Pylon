from typing import Tuple, List, Dict, Any, Optional
import os
import torch
from data.datasets import BaseDataset
import utils


class MultiTaskFacialLandmarkDataset(BaseDataset):
    __doc__ = r"""

    Download:
        https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html

    Used in:
        Facial Landmark Detection by Deep Multi-task Learning (https://link.springer.com/chapter/10.1007/978-3-319-10599-4_7)
    """

    SPLIT_OPTIONS = ['train', 'test']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['landmarks', 'gender', 'smile', 'glasses', 'pose']

    ####################################################################################################
    ####################################################################################################

    def _init_annotations(self) -> None:
        image_filepaths = self._init_images_()
        all_labels = self._init_labels_(image_filepaths=image_filepaths)
        self.annotations = list(zip(image_filepaths, all_labels))

    def _init_images_(self) -> None:
        image_filepaths = []
        with open(os.path.join(self.data_root, f"{self.split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                filepath = os.path.join(self.data_root, line[0])
                assert os.path.isfile(filepath), f"{filepath=}"
                image_filepaths.append(filepath)
        return image_filepaths

    def _init_labels_(self, image_filepaths: List[str]) -> None:
        # image
        all_labels: List[Dict[str, torch.Tensor]] = []
        with open(os.path.join(self.data_root, f"{self.split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split()
                assert line[0] == os.path.relpath(path=image_filepaths[idx], start=self.data_root), \
                    f"{idx=}, {line[0]=}, {image_filepaths[idx]=}, {self.data_root=}"
                landmarks = torch.tensor(list(map(float, [c for coord in zip(line[1:6], line[6:11]) for c in coord])), dtype=torch.float32)
                attributes = dict(
                    (name, torch.tensor(int(val), dtype=torch.int8))
                    for name, val in zip(self.LABEL_NAMES[1:], line[11:15])
                )
                labels: Dict[str, torch.Tensor] = {}
                labels.update({'landmarks': landmarks})
                labels.update(attributes)
                all_labels.append(labels)
        return all_labels

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # MultiTaskFacialLandmarkDataset uses standard loading without dataset-specific parameters
        return version_dict

    ####################################################################################################
    ####################################################################################################

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {'image': utils.io.load_image(
            filepath=self.annotations[idx][0],
            dtype=torch.float32, sub=None, div=255.,
        )}
        labels = self.annotations[idx][1]
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
        """Display Multi-Task Facial Landmark datapoint with landmarks and attributes.
        
        This method visualizes facial landmark detection and multi-task classification:
        face image with overlaid landmarks, gender, smile, glasses, and pose attributes.
        
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
        import plotly.graph_objects as go
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
        
        # Validate expected data keys
        assert 'image' in inputs, f"inputs missing 'image', got keys: {list(inputs.keys())}"
        assert 'landmarks' in labels, f"labels missing 'landmarks', got keys: {list(labels.keys())}"
        
        # Create figure with landmarks overlay
        def create_landmark_display():
            # Start with base image display
            image_fig = create_image_display(
                image=inputs['image'],
                title="Face Image with Landmarks"
            )
            
            # Extract landmarks - they are stored as flattened [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
            landmarks = labels['landmarks'].cpu().numpy()
            assert len(landmarks) == 10, f"Expected 10 landmark coordinates, got {len(landmarks)}"
            
            # Reshape to (5, 2) for 5 landmarks with (x, y) coordinates
            landmark_points = landmarks.reshape(5, 2)
            x_coords = landmark_points[:, 0]
            y_coords = landmark_points[:, 1]
            
            # Get image dimensions for coordinate scaling
            image_height, image_width = inputs['image'].shape[-2:]
            
            # Add landmark points as scatter trace
            image_fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name='Facial Landmarks',
                hovertemplate='Landmark: (%{x:.1f}, %{y:.1f})<extra></extra>'
            ))
            
            return image_fig
        
        figure_tasks = [create_landmark_display]
        
        # Create figures in parallel for better performance
        figure_creator = ParallelFigureCreator(max_workers=1, enable_timing=False)
        figures = figure_creator.create_figures_parallel(figure_tasks)
        
        # Create grid layout for the single image
        figure_components = create_figure_grid(
            figures=figures,
            width_style="50%",
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
            width_style="25%"
        )
        
        # Create attributes display
        attribute_mapping = {
            'gender': {0: 'Female', 1: 'Male'},
            'smile': {0: 'No Smile', 1: 'Smiling'},
            'glasses': {0: 'No Glasses', 1: 'Wearing Glasses'},
            'pose': {0: 'Frontal', 1: 'Non-Frontal'}
        }
        
        attributes_component = html.Div([
            html.H4("Facial Attributes", style={'margin-bottom': '15px'}),
            html.Div([
                html.Div([
                    html.H5(f"{attr_name.title()}: ", style={
                        'display': 'inline', 
                        'margin-right': '10px'
                    }),
                    html.Span(
                        attribute_mapping[attr_name][int(labels[attr_name].item())],
                        style={
                            'font-size': '16px', 
                            'font-weight': 'bold', 
                            'color': '#2E86AB',
                            'background-color': '#E8F4F8',
                            'padding': '5px 10px',
                            'border-radius': '5px',
                            'border': '1px solid #2E86AB'
                        }
                    )
                ], style={'margin-bottom': '15px'})
                for attr_name in ['gender', 'smile', 'glasses', 'pose']
                if attr_name in labels
            ]),
            
            # Landmark coordinates display
            html.Div([
                html.H5("Landmark Coordinates:", style={'margin-bottom': '10px'}),
                html.Div([
                    html.Span(f"Point {i+1}: ({landmarks[i*2]:.1f}, {landmarks[i*2+1]:.1f})", 
                              style={
                                  'display': 'block',
                                  'font-family': 'monospace',
                                  'font-size': '12px',
                                  'margin-bottom': '3px',
                                  'color': '#666'
                              })
                    for i, landmarks in enumerate([labels['landmarks'].cpu().numpy()])
                    for i in range(5)
                ])
            ])
        ], style={
            'width': '25%', 
            'display': 'inline-block', 
            'vertical-align': 'top',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'margin': '10px'
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
