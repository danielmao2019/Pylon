# Data Viewer Points Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/points/dash/apis.py`

```text
apis.py
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.viewer.utils.displays.points.dash.core_points_display import create_dash_points_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display
│   ├── # Builds a Dash color point-cloud display from an already-colorized point-cloud path.
│   └── calls create_dash_points_display
├── def create_segmentation_pc_display
│   ├── # Builds a Dash segmentation point-cloud display by recoloring each point from its class id.
│   ├── calls load_point_cloud
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
│   ├── calls _map_segmentation_pc_to_rgb(segmentation_pc_path=segmentation_pc_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_points_display
└── def _map_segmentation_pc_to_rgb
    └── # Recolors the segmentation point cloud's per-point class labels to RGB via the class-to-RGB mapping for Dash display.
```

`data/viewer/utils/displays/points/dash/core_points_display.py`

```text
core_points_display.py
├── import logging
├── from typing import Any, Dict, Optional, Tuple, Union
├── import numpy as np
├── import plotly.graph_objects as go
├── import torch
├── from dash import dcc
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.random_select import RandomSelect
├── from data.transforms.vision_3d.pclod import create_lod_function
├── from data.viewer.dataset.context import get_viewer_context
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── from data.viewer.utils.segmentation import get_color
├── logger = logging.getLogger(__name__)
├── DEFAULT_POINT_SIZE_FLOOR = 0.005  # absolute floor for visibility at typical canonical-world camera framings, used by the bounding-sphere heuristic when point_size is not supplied; lib-owned default, overridable
├── DEFAULT_POINT_SIZE_RATIO = 0.002  # fraction of the point-cloud bounding-sphere radius used as the heuristic default size; lib-owned default, overridable
├── DEFAULT_POINT_COLOR = "#cccccc"   # uniform fallback used when the point cloud has no per-point colors and the caller supplies no point_color; lib-owned default, overridable
├── def build_point_cloud_id(datapoint: Dict[str, Any], component: str) -> Tuple[str, int, str]
│   ├── # Builds the structured point-cloud id of one datapoint component from the viewer context.
│   ├── assert datapoint is a dict            # reporting its type
│   ├── assert component is a str             # reporting its type
│   ├── assert datapoint carries 'meta_info'  # reporting the datapoint's keys
│   ├── impls meta_info = datapoint['meta_info']
│   ├── assert meta_info is a dict      # reporting its type
│   ├── assert meta_info carries 'idx'  # reporting meta_info's keys
│   ├── impls datapoint_idx = meta_info['idx']
│   ├── assert datapoint_idx is an int  # reporting its type
│   ├── calls get_viewer_context()
│   ├── impls dataset_name = that context's backend.current_dataset
│   ├── assert dataset_name is a str  # reporting its type
│   ├── impls point_cloud_id = (dataset_name, datapoint_idx, component)
│   └── return point_cloud_id
├── def create_point_cloud_display(pc: PointCloud, title: str, color_key: Optional[str] = None, color_type: Optional[str] = None, highlight_indices: Optional[torch.Tensor] = None, point_size: float = 2, point_opacity: float = 0.8, camera_state: Optional[Dict[str, Any]] = None, lod_type: str = "none", lod_config: Optional[Dict[str, Any]] = None, point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None, axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None, **kwargs: Any) -> go.Figure
│   ├── # Creates the Plotly point-cloud display, LOD-processed and browser-downsampled, under a camera-synced layout.
│   ├── assert pc is a PointCloud  # reporting its type
│   ├── impls points = pc.xyz
│   ├── impls colors = pc.rgb, or None when the point cloud carries none
│   ├── if color_key is not None and pc carries that field
│   │   └── impls getattr(pc, color_key)
│   ├── else
│   │   └── impls None
│   ├── impls labels = that value
│   ├── assert points is a torch.Tensor  # reporting its type
│   ├── impls logger.info(the call's point shape, lod_type and point_cloud_id)  # impls-node-one-step:skip — names logged values
│   ├── assert points is 2D                # reporting its shape
│   ├── assert points holds 3 coordinates  # reporting its second dimension
│   ├── assert points is non-empty
│   ├── assert title is a str                  # reporting its type
│   ├── assert point_size is numeric           # reporting its type
│   ├── assert point_opacity is numeric        # reporting its type
│   ├── assert point_opacity is within [0, 1]  # reporting point_opacity
│   ├── if colors is not None
│   │   ├── assert colors is a torch.Tensor      # reporting its type
│   │   └── assert colors has one row per point  # reporting both lengths
│   ├── if labels is not None
│   │   ├── assert labels is a torch.Tensor      # reporting its type
│   │   └── assert labels has one row per point  # reporting both lengths
│   ├── if camera_state is not None
│   │   └── assert camera_state is a dict  # reporting its type
│   ├── if lod_config is not None
│   │   └── assert lod_config is a dict  # reporting its type
│   ├── if axis_ranges is not None
│   │   └── assert axis_ranges is a dict  # reporting its type
│   ├── impls original_count = len(points)
│   ├── calls _apply_lod_processing(point_cloud=pc, key=color_key, lod_type=lod_type, lod_config=lod_config, camera_state=camera_state, point_cloud_id=point_cloud_id, point_size=point_size, point_opacity=point_opacity, axis_ranges=axis_ranges, original_count=original_count, title=title)  # -> (processed_pc, title)
│   ├── calls _apply_browser_downsampling(processed_pc=processed_pc, highlight_indices=highlight_indices, original_count=original_count, title=title)  # -> (processed_pc, highlight_indices, title)
│   ├── calls _create_point_cloud_figure(pc=processed_pc, color_key=color_key, color_type=color_type, highlight_indices=highlight_indices, point_size=point_size, point_opacity=point_opacity, axis_ranges=axis_ranges, camera_state=camera_state)  # -> fig
│   ├── impls fig.update_layout(title=title, uirevision='camera')  # one uirevision keeps the camera in sync across redraws
│   └── return fig
├── def _apply_lod_processing(point_cloud: PointCloud, key: Optional[str], lod_type: str, lod_config: Optional[Dict[str, Any]], camera_state: Optional[Dict[str, Any]], point_cloud_id: Optional[Union[str, Tuple[str, int, str]]], point_size: float, point_opacity: float, axis_ranges: Optional[Dict[str, Tuple[float, float]]], original_count: int, title: str) -> Tuple[PointCloud, str]
│   ├── # Applies the requested level-of-detail selection to the point cloud and states the reduction in its title.
│   ├── assert point_cloud is a PointCloud  # reporting its type
│   ├── impls effective_lod_config = lod_config, or an empty dict when it is falsy
│   ├── if lod_type is 'continuous' or 'discrete'
│   │   ├── if camera_state is None
│   │   │   ├── impls logger.info(that an advanced LOD was requested with auto-camera)
│   │   │   ├── calls _create_point_cloud_figure(pc=point_cloud, color_key=key, color_type=None, highlight_indices=None, point_size=point_size, point_opacity=point_opacity, axis_ranges=axis_ranges, camera_state=None)  # -> temp_fig, handed to _extract_default_camera_from_figure below
│   │   │   └── calls _extract_default_camera_from_figure(temp_fig, point_cloud.xyz)  # -> effective_camera_state
│   │   ├── else
│   │   │   └── impls effective_camera_state = camera_state
│   │   ├── impls effective_lod_config = a copy of effective_lod_config
│   │   └── impls effective_lod_config['camera_state'] = effective_camera_state
│   ├── if point_cloud_id
│   │   └── calls normalize_point_cloud_id(point_cloud_id)
│   ├── else
│   │   └── impls None
│   ├── impls normalized_id = that value
│   ├── calls create_lod_function(lod_type=lod_type, lod_config=effective_lod_config, point_cloud_id=normalized_id)  # -> lod_function
│   ├── impls processed_pc = lod_function(point_cloud)
│   ├── impls points_tensor = processed_pc.xyz
│   ├── impls updated_title = title
│   ├── if lod_type is 'density' and points_tensor is shorter than original_count
│   │   ├── impls density_pct = effective_lod_config.get('density', 100)
│   │   ├── impls density_suffix = the density percentage with the kept and original counts  # impls-node-one-step:skip — names two counts
│   │   ├── impls updated_title = title followed by density_suffix
│   │   └── impls logger.info(that density was applied, with both counts)
│   ├── elif lod_type is 'continuous' or 'discrete' and points_tensor is shorter than original_count
│   │   ├── impls lod_suffix = the LOD type with the kept and original counts  # impls-node-one-step:skip — names two counts
│   │   ├── impls updated_title = title followed by lod_suffix
│   │   └── impls logger.info(that LOD was applied, with both counts)
│   ├── else
│   │   └── impls logger.info(that no title update applies, with lod_type and both counts)  # impls-node-one-step:skip — names logged values
│   ├── if points_tensor is empty
│   │   └── impls processed_pc.xyz = the single origin point, float32 on the point cloud's device
│   ├── impls lod_result = (processed_pc, updated_title)
│   └── return lod_result
├── def normalize_point_cloud_id(point_cloud_id: Union[str, Tuple[str, ...]]) -> str
│   ├── # Normalizes a point-cloud id to its string cache key.
│   ├── if point_cloud_id is a str
│   │   └── return point_cloud_id
│   └── else
│       ├── for part in point_cloud_id
│       │   └── impls str(part)
│       ├── impls normalized_point_cloud_id = those parts joined on ":"
│       └── return normalized_point_cloud_id
├── def _extract_default_camera_from_figure(fig: go.Figure, points: torch.Tensor) -> Dict[str, Any]
│   ├── # Approximates Plotly's own auto-calculated camera state for a figure, so LOD can run against a framing.
│   ├── impls points_np = points.cpu().numpy()
│   ├── impls pc_center = points_np.mean(axis=0)
│   ├── impls pc_size = the per-axis extent of points_np
│   ├── impls max_dim = np.max(pc_size)
│   ├── impls distance_factor = 1.5
│   ├── impls camera_offset = max_dim * distance_factor
│   ├── impls eye_pos = pc_center offset diagonally by camera_offset in x and y and by 0.7 of it in z  # impls-node-one-step:skip — one offset expression
│   ├── impls camera_state = the eye at eye_pos, the center at pc_center, and up along +Z  # impls-node-one-step:skip — names three camera fields
│   ├── impls logger.info(the extracted eye and center)  # impls-node-one-step:skip — names logged values
│   └── return camera_state
├── def _apply_browser_downsampling(processed_pc: PointCloud, highlight_indices: Optional[torch.Tensor], original_count: int, title: str) -> Tuple[PointCloud, Optional[torch.Tensor], str]
│   ├── # Caps the point count at what a browser can hold, remapping the highlight indices onto the kept points.
│   ├── impls MAX_BROWSER_POINTS = 50000  # the most points the browser handles reliably
│   ├── impls points_tensor = processed_pc.xyz
│   ├── if points_tensor holds at most MAX_BROWSER_POINTS points
│   │   ├── impls downsampling_result = (processed_pc, highlight_indices, title)
│   │   └── return downsampling_result
│   ├── impls logger.info(the downsampling, with both counts)
│   ├── calls RandomSelect(count=MAX_BROWSER_POINTS)     # -> browser_downsample
│   ├── calls browser_downsample(processed_pc, seed=42)  # -> downsampled_pc, one fixed seed for reproducibility
│   ├── impls final_indices = downsampled_pc.indices, or None when it carries none
│   ├── impls updated_highlight_indices = highlight_indices
│   ├── if highlight_indices is not None and final_indices is not None
│   │   ├── impls reverse_mapping = an original_count-long int64 tensor of -1 on the highlight indices' device
│   │   ├── impls new_positions = torch.arange(len(final_indices)) as int64 on that device
│   │   ├── impls reverse_mapping[final_indices] = new_positions
│   │   ├── impls new_highlight_indices = reverse_mapping[highlight_indices]
│   │   ├── impls valid_mask = new_highlight_indices >= 0
│   │   ├── impls filtered_highlight_indices = new_highlight_indices[valid_mask]
│   │   ├── if filtered_highlight_indices is empty
│   │   │   └── impls updated_highlight_indices = None
│   │   └── else
│   │       └── impls updated_highlight_indices = filtered_highlight_indices
│   ├── impls updated_title = title followed by the browser limit and the original count  # impls-node-one-step:skip — names two appended values
│   ├── impls downsampling_result = (downsampled_pc, updated_highlight_indices, updated_title)
│   └── return downsampling_result
├── def _create_point_cloud_figure(pc: PointCloud, color_key: Optional[str], color_type: Optional[str], highlight_indices: Optional[torch.Tensor], point_size: float, point_opacity: float, axis_ranges: Optional[Dict[str, Tuple[float, float]]], camera_state: Optional[Dict[str, Any]]) -> go.Figure
│   ├── # Builds the Plotly figure of one point cloud, its highlighted points carried in their own trace.
│   ├── assert pc is a PointCloud  # reporting its type
│   ├── impls points = pc.xyz
│   ├── assert points is a torch.Tensor    # reporting its type
│   ├── assert points is 2D                # reporting its shape
│   ├── assert points holds 3 coordinates  # reporting its second dimension
│   ├── assert points is non-empty
│   ├── if color_key is not None
│   │   ├── assert pc carries the color_key field  # reporting color_key
│   │   ├── impls labels = getattr(pc, color_key)
│   │   ├── assert labels is a torch.Tensor      # reporting its type
│   │   ├── assert labels has one row per point  # reporting both lengths
│   │   ├── if color_type is None
│   │   │   ├── if color_key is 'classification' or 'change_map'
│   │   │   │   └── impls color_type = 'classification'
│   │   │   ├── elif color_key is 'density'
│   │   │   │   └── impls color_type = 'regression'
│   │   │   └── else
│   │   │       └── raise ValueError  # color_type cannot be inferred from this color_key
│   │   ├── assert color_type is a str                             # reporting its type
│   │   ├── assert color_type is 'classification' or 'regression'  # reporting color_type
│   │   ├── if color_type is 'classification'
│   │   │   ├── calls _convert_labels_to_colors_torch(labels)  # -> colors
│   │   │   ├── impls colors_np = colors.cpu().numpy()
│   │   │   └── impls marker_color_config = the per-point colors
│   │   └── else
│   │       ├── impls labels_np = labels.cpu().numpy()
│   │       ├── impls (cmin, cmax) = the label range, as floats
│   │       └── impls marker_color_config = the raw labels on the Viridis colorscale with its scale shown, its colorbar titled by color_key or 'Value', spanning cmin to cmax
│   ├── else
│   │   ├── assert pc carries 'rgb'
│   │   ├── impls colors = pc.rgb
│   │   ├── assert colors is a torch.Tensor      # reporting its type
│   │   ├── assert colors has one row per point  # reporting both lengths
│   │   ├── impls colors_np = colors.cpu().numpy()
│   │   └── impls marker_color_config = the per-point colors
│   ├── if highlight_indices is not None
│   │   ├── assert highlight_indices is a torch.Tensor          # reporting its type
│   │   ├── assert highlight_indices is int32 or int64          # reporting its dtype
│   │   ├── assert highlight_indices are non-negative           # reporting their minimum
│   │   └── assert highlight_indices are below the point count  # reporting the count and their maximum
│   ├── impls points_np = points.cpu().numpy()
│   ├── impls fig = go.Figure()
│   ├── if highlight_indices is not None
│   │   ├── impls highlight_indices_np = highlight_indices.cpu().numpy()
│   │   ├── impls all_indices = np.arange(len(points))
│   │   ├── impls highlight_mask = whether each of all_indices sits in highlight_indices_np
│   │   ├── impls non_highlight_mask = the complement of highlight_mask
│   │   ├── if any point is not highlighted
│   │   │   ├── impls marker_config = size point_size at 0.05 of point_opacity
│   │   │   ├── assert marker_color_config carries 'color'  # reporting its keys
│   │   │   ├── impls marker_config.update(marker_color_config)
│   │   │   ├── impls marker_config['color'] = the color config masked to the non-highlighted points
│   │   │   ├── impls non_highlight_kwargs = those points' x, y and z as markers, with that marker config, hover skipped and no legend  # impls-node-one-step:skip — one kwargs dict's fields
│   │   │   └── impls fig.add_trace(that Scatter3d)
│   │   └── if any point is highlighted
│   │       ├── impls marker_config = size point_size at full point_opacity
│   │       ├── assert marker_color_config carries 'color'  # reporting its keys
│   │       ├── impls marker_config.update(marker_color_config)
│   │       ├── impls marker_config['color'] = the color config masked to the highlighted points
│   │       ├── impls marker_config['showscale'] = False  # the non-highlighted trace already carries the scale
│   │       ├── impls highlight_kwargs = those points' x, y and z as markers, with that marker config, hover skipped and no legend  # impls-node-one-step:skip — one kwargs dict's fields
│   │       └── impls fig.add_trace(that Scatter3d)
│   ├── else
│   │   ├── impls marker_config = size point_size at point_opacity
│   │   ├── impls marker_config.update(marker_color_config)
│   │   ├── impls scatter3d_kwargs = every point's x, y and z as markers, with that marker config, hover skipped and no legend  # impls-node-one-step:skip — one kwargs dict's fields
│   │   └── impls fig.add_trace(that Scatter3d)
│   ├── if axis_ranges
│   │   ├── impls x_range = axis_ranges['x'], or the data's own x extent
│   │   ├── impls y_range = axis_ranges['y'], or the data's own y extent
│   │   └── impls z_range = axis_ranges['z'], or the data's own z extent
│   ├── else
│   │   ├── impls x_range = the data's own x extent
│   │   ├── impls y_range = the data's own y extent
│   │   └── impls z_range = the data's own z extent
│   ├── impls scene_dict = the X, Y and Z axis titles, aspectmode 'data', and those three ranges  # impls-node-one-step:skip — one dict's fields
│   ├── if camera_state is not None
│   │   └── impls scene_dict['camera'] = camera_state
│   ├── impls fig.update_layout(scene=scene_dict)
│   └── return fig
├── def _convert_labels_to_colors_torch(labels: torch.Tensor) -> torch.Tensor
│   ├── # Converts integer labels to their segmentation RGB colors, as a (N, 3) uint8 tensor.
│   ├── assert labels is a torch.Tensor  # reporting its type
│   ├── assert labels is 1D              # reporting its shape
│   ├── impls device = labels.device
│   ├── impls unique_labels = torch.unique(labels)
│   ├── impls colors = an (N, 3) uint8 tensor of zeros on device
│   ├── for label in unique_labels
│   │   ├── calls get_color(label.item())  # -> color_hex
│   │   ├── impls r = the hex color's red byte
│   │   ├── impls g = the hex color's green byte
│   │   ├── impls b = the hex color's blue byte
│   │   ├── impls mask = labels == label
│   │   └── impls colors[mask, :] = (r, g, b) as uint8 on device
│   └── return colors
├── def apply_lod_to_point_cloud(points: torch.Tensor, colors: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, camera_state: Optional[Dict[str, Any]] = None, lod_type: str = "none", density_percentage: int = 100, point_cloud_id: Optional[Union[str, Tuple[str, ...]]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
│   ├── # Selects a level-of-detail subset of the points, carrying their colors and labels along.
│   ├── assert points is a torch.Tensor                   # reporting its type
│   ├── assert points is $N \times 3$                     # reporting its shape
│   ├── assert colors is None or a torch.Tensor           # reporting its type
│   ├── assert labels is None or a torch.Tensor           # reporting its type
│   ├── assert colors is None or has one row per point    # reporting both shapes
│   ├── assert labels is None or has one row per point    # reporting both shapes
│   ├── assert density_percentage is within [1, 100]      # reporting density_percentage
│   ├── assert lod_type is a str                          # reporting lod_type
│   ├── assert point_cloud_id is None, a str, or a tuple  # reporting point_cloud_id
│   ├── impls target_count = that percentage of the point count, at least 1
│   ├── impls target_count = min(target_count, the point count)
│   ├── if lod_type is 'none' or 'density'
│   │   └── impls indices = target_count indices spread evenly over the point range, as longs on the points' device
│   ├── elif lod_type is 'continuous' or 'discrete'
│   │   ├── impls effective_camera_state = camera_state
│   │   ├── if effective_camera_state is None
│   │   │   └── impls effective_camera_state = the eye at the origin
│   │   ├── impls eye = effective_camera_state.get('eye')
│   │   ├── impls camera_eye = the eye's (x, y, z) in the points' dtype on their device
│   │   ├── impls distances = the per-point distance to camera_eye
│   │   ├── impls torch.quantile(input=distances, q=0.5)
│   │   └── impls indices = the target_count nearest points, by distance
│   ├── else
│   │   └── assert False  # "lod_type is unsupported", reporting lod_type
│   ├── impls selected_colors = None
│   ├── if colors is not None
│   │   └── impls selected_colors = colors[indices]
│   ├── impls selected_labels = None
│   ├── if labels is not None
│   │   └── impls selected_labels = labels[indices]
│   ├── impls selection = (points[indices], selected_colors, selected_labels)
│   └── return selection
├── def create_dash_points_display(point_cloud: PointCloud, point_size: Optional[float] = None, point_color: Optional[str] = None) -> dcc.Graph
│   ├── # Renders a Dash point-cloud display element; the point_size and point_color overrides are opt-in.
│   ├── assert point_cloud is a PointCloud    # reporting its type
│   ├── assert point_size is None or numeric  # reporting its type
│   ├── assert point_color is None or a str   # reporting its type
│   ├── calls create_dash_points_scene(point_cloud=point_cloud, point_size=point_size, point_color=point_color)  # -> scene
│   ├── impls controls = create_dash_trackball_camera_controls
│   ├── calls create_dash_points_component(scene=scene, controls=controls)
│   └── return
├── def create_dash_points_scene(point_cloud: PointCloud, point_size: Optional[float] = None, point_color: Optional[str] = None) -> go.Scatter3d
│   ├── # Sync-builds the Plotly Scatter3d marker trace from the point cloud.
│   ├── assert point_cloud is a PointCloud    # reporting its type
│   ├── assert point_size is None or numeric  # reporting its type
│   ├── assert point_color is None or a str   # reporting its type
│   ├── impls points_np = point_cloud.xyz.detach().cpu().numpy()
│   ├── impls center = points_np.mean(axis=0)
│   ├── impls bounding_radius = the largest distance from center to a point
│   ├── if point_size is not None
│   │   └── impls effective_size = point_size
│   ├── else
│   │   └── impls effective_size = max(DEFAULT_POINT_SIZE_FLOOR, bounding_radius * DEFAULT_POINT_SIZE_RATIO)
│   ├── if point_color is not None
│   │   └── impls effective_color = point_color
│   ├── calls point_cloud.field_names()
│   ├── elif the point cloud carries 'rgb'
│   │   ├── impls point_cloud.rgb.detach()
│   │   ├── impls that tensor on the cpu
│   │   └── impls effective_color = that tensor as a numpy array
│   ├── else
│   │   └── impls effective_color = DEFAULT_POINT_COLOR
│   ├── impls trace = the points' x, y and z as markers, sized effective_size and colored effective_color  # impls-node-one-step:skip — one trace's arguments
│   └── return trace
├── def create_dash_points_component(scene: go.Scatter3d, controls: Any) -> dcc.Graph
│   ├── # Wraps the point-cloud scene into a Dash graph over a one-trace figure.
│   ├── assert scene is a go.Scatter3d  # reporting its type
│   ├── impls component = the dcc.Graph over the figure holding that scene
│   └── return component
└── def get_point_cloud_display_stats(point_cloud: PointCloud, change_map: Optional[torch.Tensor] = None, class_names: Optional[Dict[int, str]] = None) -> Dict[str, Any]
    ├── # Reports one point cloud's display statistics, with its change classes' distribution when a change map is given.
    ├── assert point_cloud is a PointCloud  # reporting its type
    ├── impls points = point_cloud.xyz
    ├── assert points is a torch.Tensor            # reporting its type
    ├── assert points is 2D                        # reporting its shape
    ├── assert points holds 3 coordinates or more  # reporting its second dimension
    ├── assert points is non-empty
    ├── if change_map is not None
    │   ├── assert change_map is a torch.Tensor        # reporting its type
    │   └── assert change_map has one entry per point  # reporting both lengths
    ├── if class_names is not None
    │   └── assert class_names is a dict  # reporting its type
    ├── impls points_np = points.detach().cpu().numpy()
    ├── calls point_cloud.field_names()
    ├── impls stats = those field names, the point count, the dimension count, the per-axis ranges, and the center  # impls-node-one-step:skip — one dict's fields
    ├── if change_map is not None
    │   ├── impls (unique_classes, class_counts) = the change map's unique classes and their counts  # impls-node-one-step:skip — one call's two returns
    │   ├── impls unique_classes = unique_classes.cpu().numpy()
    │   ├── impls class_counts = class_counts.cpu().numpy()
    │   ├── impls total_points = change_map.numel()
    │   ├── impls class_distribution = {}
    │   ├── for (cls, count) in zip(unique_classes, class_counts, strict=True)
    │   │   ├── impls percentage = count over total_points, as a percentage
    │   │   ├── if cls carries an 'item'
    │   │   │   └── impls cls.item()
    │   │   ├── else
    │   │   │   └── impls cls
    │   │   ├── impls cls_key = that value
    │   │   ├── if class_names is non-empty and carries cls_key
    │   │   │   └── impls class_names[cls_key]
    │   │   ├── else
    │   │   │   └── impls the "Class {cls_key}" fallback name
    │   │   ├── impls class_name = that name
    │   │   └── impls class_distribution[class_name] = that count and percentage  # impls-node-one-step:skip — one entry's two values
    │   └── impls stats['class_distribution'] = class_distribution
    └── return stats
```

### Backend schemas

`data/viewer/utils/displays/points/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PointDisplayResponse(DisplayResponse)
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorPCDisplayResponse(PointDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_pc"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class SegmentationPCDisplayResponse(PointDisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_pc"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/points/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, Optional, Tuple
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.displays.points.ts.backend.core_points_display import create_points_display_response_core
├── from data.viewer.utils.displays.points.ts.backend.schemas.display_response import SegmentationPCDisplayResponse
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display_response
│   ├── # Creates a color point-cloud response from an already colorized point resource.
│   ├── impls point-display meta_info is empty metadata
│   ├── calls create_points_display_response_core
│   └── return
├── def create_segmentation_pc_display_response(segmentation_pc_path: str, slot_id: str, title: str, class_id_to_rgb: Optional[Dict[int, Tuple[int, int, int]]] = None) -> SegmentationPCDisplayResponse
│   ├── # Creates a segmentation point-cloud response from a class-labeled point resource; the caller may override the class-id → rgb mapping, otherwise the lib computes the default mapping via map_class_ids_to_rgb.
│   ├── calls load_point_cloud
│   ├── impls effective_class_id_to_rgb = class_id_to_rgb if class_id_to_rgb is not None else map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
│   ├── calls _map_segmentation_pc_to_rgb
│   ├── calls _build_segmentation_pc_meta_info
│   ├── calls create_points_display_response_core
│   └── return
├── def _map_segmentation_pc_to_rgb(segmentation_pc_path: str, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> str
│   ├── # Writes a backend-colorized point-cloud resource using the class-to-RGB mapping.
│   ├── impls assert isinstance(segmentation_pc_path, str)
│   ├── impls assert isinstance(class_id_to_rgb, dict)
│   ├── calls load_point_cloud(filepath=segmentation_pc_path, device="cpu")
│   ├── impls label = the loaded cloud's own class ids, cast to torch.int64
│   ├── impls rgb = a float32 zeros tensor of shape (segmentation_pc.num_points, 3) on the cloud's device
│   ├── for each class_id, color in class_id_to_rgb.items()
│   │   └── impls rgb[label == int(class_id)] = color  # a label with no mapping entry keeps rgb 0
│   ├── impls colorized_data = the cloud's fields other than xyz / rgb / colors, carrying rgb under the "rgb" key
│   ├── impls colorized_pc = PointCloud(xyz=segmentation_pc.xyz, data=colorized_data)
│   ├── impls output_path = the deterministic colorized display path derived from segmentation_pc_path
│   ├── calls save_point_cloud(pc=colorized_pc, output_filepath=str(output_path))
│   ├── impls colorized_pc_path = str(output_path)
│   └── return colorized_pc_path  # the colorized point-cloud path the response serves
└── def _build_segmentation_pc_meta_info(class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> Dict[str, Any]
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`data/viewer/utils/displays/points/ts/backend/core_points_display.py`

```text
core_points_display.py
└── def create_points_display_response_core
    ├── # Creates a point display response from the loadable point resource path and caller-provided display metadata.
    ├── impls builds frontend resource url from point_cloud_path
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/points/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── interface PointDisplayResponse extends DisplayResponse
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorPCDisplayResponse extends PointDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_pc"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface SegmentationPCDisplayResponse extends PointDisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_pc"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`data/viewer/utils/displays/points/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { ColorPCDisplayResponse, SegmentationPCDisplayResponse } from "./types/display_response";
├── import { renderPointsDisplay, createPointsObject } from "./core_points_display";
├── import { registerSpatialLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── function renderColorPCDisplay({ displayResponse, initialCameraState, pointSize, pointColor }: { displayResponse: ColorPCDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string }): LeafVNode
│   ├── # Renders a color point-cloud display with opt-in pointSize and pointColor overrides.
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor })
│   └── return
├── function renderSegmentationPCDisplay({ displayResponse, initialCameraState, pointSize }: { displayResponse: SegmentationPCDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number }): LeafVNode
│   ├── # Renders the backend-colorized segmentation display and legend derived from meta_info; per-point colors are already baked in by the backend's class-id → rgb mapping, so no color override is exposed here.
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize })
│   └── return
└── impls registerSpatialLayerRenderer({ displayKind: "color_pc", layerRenderer: createPointsObject })  # module-load self-registration of the spatial color-pc layer renderer
```

`data/viewer/utils/displays/points/ts/frontend/core_points_display.ts`

```text
core_points_display.ts
├── import * as THREE from "three";
├── import { createTrackballCameraControls, type ThreeTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { PointDisplayResponse } from "./types/display_response";
├── export const DEFAULT_POINT_SIZE_FLOOR = 0.005  # smallest world-space point size the display falls back to, and the floor the bounding-sphere-relative auto-size is clamped against
├── export const DEFAULT_POINT_SIZE_RATIO = 0.002  # fraction of the geometry bounding-sphere radius used as the heuristic default size; lib-owned default, overridable
├── export const DEFAULT_POINT_COLOR = "#cccccc"   # uniform fallback used when the geometry has no per-point colors and the caller supplies no pointColor; lib-owned default, overridable
├── interface PlyProperty
│   ├── # One scalar property declared on the PLY vertex element.
│   ├── type: string
│   └── name: string
├── interface PlyHeader
│   ├── # The PLY header as this parser reads it: its format, its vertex count, and the vertex element's scalar properties.
│   ├── format: string
│   ├── vertexCount: number
│   └── properties: PlyProperty[]
├── interface PlyPropertyIndices
│   ├── # Column index of each consumed vertex property in an ASCII PLY row, -1 where the header declares none.
│   ├── x: number
│   ├── y: number
│   ├── z: number
│   ├── red: number
│   ├── green: number
│   └── blue: number
├── interface PlyPropertyOffset
│   ├── # One binary vertex property's byte offset within a vertex record, and its scalar type.
│   ├── offset: number
│   └── type: string
├── interface PlyPropertyOffsets
│   ├── # Byte layout of a binary PLY vertex record: its stride and each consumed property's own offset.
│   ├── stride: number
│   ├── x: PlyPropertyOffset
│   ├── y: PlyPropertyOffset
│   ├── z: PlyPropertyOffset
│   ├── red?: PlyPropertyOffset
│   ├── green?: PlyPropertyOffset
│   └── blue?: PlyPropertyOffset
├── export function renderPointsDisplay({ displayResponse, initialCameraState = null, pointSize, pointColor, }: { displayResponse: PointDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string; }): LeafVNode
│   ├── # Renders a self-contained point-cloud display element initialized at initialCameraState.
│   ├── () => [local]
│   │   ├── # The leaf's render: mounts the points display and returns its container.
│   │   ├── calls createSpatialDisplayScene({ initialCameraState })               # -> { container, scene, camera, renderer }
│   │   ├── calls createPointsObject({ displayResponse, pointSize, pointColor })  # -> object
│   │   ├── impls scene.add(object)
│   │   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })  # -> controls
│   │   ├── calls renderPointsScene({ scene, camera, renderer, controls })
│   │   └── return container
│   ├── impls leaf = the LeafVNode keyed by displayResponse.url or `points:${displayResponse.slot_id}`, with empty props and that render  # impls-node-one-step:skip — one constructor's fields
│   └── return leaf
├── export function createPointsObject({ displayResponse, pointSize, pointColor, }: { displayResponse: PointDisplayResponse; pointSize?: number; pointColor?: string; }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the point cloud, filled with the THREE.Points once the async geometry load resolves.
│   ├── impls group = new THREE.Group()
│   ├── calls loadPointGeometry({ displayResponse })
│   ├── (geometry) => [local]
│   │   ├── # On resolve: builds the points from the loaded geometry into the already-returned group.
│   │   ├── calls createThreePoints({ geometry, pointSize, pointColor })
│   │   └── impls group.add(those points)
│   ├── (error) => [local]
│   │   ├── # On rejection: throws a new Error carrying the underlying message.
│   │   ├── if error is an Error
│   │   │   └── impls error.message
│   │   ├── else
│   │   │   └── impls String(error)
│   │   ├── impls message = that text
│   │   └── throw unable to load point cloud: ${message}
│   ├── impls the geometry load, chained through that resolve step and that rejection step  # impls-node-one-step:skip — both steps have their own nodes
│   └── return group
├── export async function loadPointGeometry({ displayResponse, }: { displayResponse: PointDisplayResponse; }): Promise<THREE.BufferGeometry>
│   ├── # Async-loads the point resource served at displayResponse.url and parses it into a BufferGeometry.
│   ├── if displayResponse.url is null
│   │   └── throw point display response url is null
│   ├── impls response = await fetch(displayResponse.url)
│   ├── if the response is not ok
│   │   └── throw unable to load point cloud: HTTP ${response.status}
│   ├── impls buffer = await response.arrayBuffer()
│   ├── calls parsePlyBuffer({ buffer })
│   └── return  # that geometry
├── export function createThreePoints({ geometry, pointSize, pointColor, }: { geometry: THREE.BufferGeometry; pointSize?: number; pointColor?: string; }): THREE.Points
│   ├── # Sync-builds the points material and THREE.Points from the loaded geometry.
│   ├── impls geometry.computeBoundingSphere()
│   ├── impls boundingRadius = geometry.boundingSphere?.radius ?? 0
│   ├── impls effectiveSize = pointSize ?? max(DEFAULT_POINT_SIZE_FLOOR, boundingRadius * DEFAULT_POINT_SIZE_RATIO)
│   ├── impls let useVertexColors: boolean
│   ├── impls let effectiveColor: string | undefined
│   ├── if pointColor is supplied
│   │   ├── impls useVertexColors = false
│   │   └── impls effectiveColor = pointColor
│   ├── else if the geometry carries a "color" attribute
│   │   ├── impls useVertexColors = true
│   │   └── impls effectiveColor = undefined
│   ├── else
│   │   ├── impls useVertexColors = false
│   │   └── impls effectiveColor = DEFAULT_POINT_COLOR
│   ├── if effectiveColor is supplied
│   │   └── impls { color: effectiveColor }
│   ├── else
│   │   └── impls {}
│   ├── impls material = new THREE.PointsMaterial(vertexColors useVertexColors, size effectiveSize, spread with that color entry)
│   ├── impls points = new THREE.Points(geometry, material)
│   └── return points
├── function parsePlyBuffer({ buffer, }: { buffer: ArrayBuffer; }): THREE.BufferGeometry
│   ├── # Parses a PLY buffer, ASCII or binary little-endian, into a BufferGeometry carrying position and color attributes.
│   ├── impls headerBytes = the buffer's leading 1048576 bytes at most
│   ├── impls headerText = headerBytes decoded as utf-8
│   ├── impls endIndex = headerText.indexOf("end_header")
│   ├── if endIndex < 0
│   │   └── throw new Error("PLY header is missing end_header")
│   ├── impls headerPrefix = headerText up to endIndex
│   ├── impls rawHeader = headerText through the end of "end_header"
│   ├── impls encoder = new TextEncoder()
│   ├── impls bytes = new Uint8Array(buffer)
│   ├── impls dataOffset = the encoded byte length of rawHeader
│   ├── while dataOffset is inside bytes and bytes[dataOffset] is a newline byte, 10 or 13
│   │   └── impls dataOffset += 1
│   ├── calls readPlyHeader({ headerText: headerPrefix })  # -> header
│   ├── if header.format is "ascii"
│   │   ├── calls parseAsciiPlyGeometry({ buffer, dataOffset, header })
│   │   └── return
│   ├── if header.format is "binary_little_endian"
│   │   ├── calls parseBinaryLittleEndianPlyGeometry({ buffer, dataOffset, header })
│   │   └── return
│   └── throw new Error(`unsupported PLY format ${header.format}`)
├── function readPlyHeader({ headerText }: { headerText: string }): PlyHeader
│   ├── # Reads the PLY format, the vertex count, and the vertex element's scalar properties out of the header text.
│   ├── impls lines = headerText split on newlines
│   ├── impls format = ""
│   ├── impls vertexCount = 0
│   ├── impls inVertex = false
│   ├── impls properties = []
│   ├── for each line in lines
│   │   ├── impls parts = the trimmed line split on whitespace
│   │   ├── if parts is empty or its first entry is empty
│   │   │   └── continue
│   │   ├── if parts[0] is "format"
│   │   │   ├── impls format = parts[1]
│   │   │   └── continue
│   │   ├── if parts[0] is "element"
│   │   │   ├── impls inVertex = whether parts[1] is "vertex"
│   │   │   ├── if inVertex
│   │   │   │   └── impls vertexCount = Number(parts[2])
│   │   │   └── continue
│   │   └── if parts[0] is "property" and inVertex
│   │       ├── if parts[1] is "list"
│   │       │   └── throw vertex list properties are not supported
│   │       └── impls properties.push({ type: parts[1], name: parts[2] })
│   ├── if format is empty
│   │   └── throw PLY format is missing
│   ├── if vertexCount is not finite or below 1
│   │   └── throw PLY vertex count is invalid: ${vertexCount}
│   ├── impls header = { format, vertexCount, properties }
│   └── return header
├── function parseAsciiPlyGeometry({ buffer, dataOffset, header, }: { buffer: ArrayBuffer; dataOffset: number; header: PlyHeader; }): THREE.BufferGeometry
│   ├── # Reads an ASCII PLY body, one whitespace-split row per vertex, into the position and color attributes.
│   ├── impls dataText = the buffer past dataOffset decoded as utf-8
│   ├── impls lines = the trimmed dataText split on newlines
│   ├── calls plyPropertyIndices({ properties: header.properties })  # -> indices
│   ├── impls positions = new Float32Array(header.vertexCount * 3)
│   ├── impls colors = new Float32Array(header.vertexCount * 3)
│   ├── for index over every vertex of header.vertexCount
│   │   ├── impls parts = the trimmed line at index split on whitespace
│   │   ├── if parts is undefined or shorter than the header's property count
│   │   │   └── throw ASCII PLY row is missing vertex data: ${index}
│   │   ├── calls readAsciiColorComponent({ parts, index: indices.red })
│   │   ├── calls readAsciiColorComponent({ parts, index: indices.green })
│   │   ├── calls readAsciiColorComponent({ parts, index: indices.blue })
│   │   └── calls writeGeometryVertex({ positions, colors, index, x, y, z read as numbers at indices.x, indices.y, indices.z, red, green and blue read through that step })
│   ├── calls createPointBufferGeometry({ positions, colors })
│   └── return  # that geometry
├── function parseBinaryLittleEndianPlyGeometry({ buffer, dataOffset, header, }: { buffer: ArrayBuffer; dataOffset: number; header: PlyHeader; }): THREE.BufferGeometry
│   ├── # Reads a binary little-endian PLY body, one fixed-stride record per vertex, into the position and color attributes.
│   ├── impls view = new DataView(buffer)
│   ├── calls plyPropertyOffsets({ properties: header.properties })  # -> offsets
│   ├── impls positions = new Float32Array(header.vertexCount * 3)
│   ├── impls colors = new Float32Array(header.vertexCount * 3)
│   ├── for index over every vertex of header.vertexCount
│   │   ├── impls base = dataOffset + index * offsets.stride
│   │   ├── calls readBinaryScalar({ view, offset: base + offsets.x.offset, type: offsets.x.type })
│   │   ├── calls readBinaryScalar({ view, offset: base + offsets.y.offset, type: offsets.y.type })
│   │   ├── calls readBinaryScalar({ view, offset: base + offsets.z.offset, type: offsets.z.type })
│   │   ├── calls readBinaryColorComponent({ view, base, offset: offsets.red })
│   │   ├── calls readBinaryColorComponent({ view, base, offset: offsets.green })
│   │   ├── calls readBinaryColorComponent({ view, base, offset: offsets.blue })
│   │   └── calls writeGeometryVertex({ positions, colors, index, x, y, z read through those scalar steps, red, green and blue read through those color steps })
│   ├── calls createPointBufferGeometry({ positions, colors })
│   └── return  # that geometry
├── function plyPropertyIndices({ properties, }: { properties: PlyProperty[]; }): PlyPropertyIndices
│   ├── # Resolves each consumed vertex property to its column index in an ASCII PLY row.
│   ├── (property) => [local]
│   │   ├── # Per property: its name.
│   │   └── impls property.name
│   ├── impls names = properties mapped through that step
│   ├── impls x = names.indexOf("x")
│   ├── impls y = names.indexOf("y")
│   ├── impls z = names.indexOf("z")
│   ├── if any of x, y and z is below 0
│   │   └── throw PLY vertex coordinates are missing
│   ├── impls indices = { x, y, z, and the red, green and blue indices, -1 where the header declares none }  # impls-node-one-step:skip — one record's fields
│   └── return indices
├── function plyPropertyOffsets({ properties, }: { properties: PlyProperty[]; }): PlyPropertyOffsets
│   ├── # Lays every declared vertex property out at its byte offset within one binary vertex record.
│   ├── impls offsets = {}
│   ├── impls offset = 0
│   ├── for each property in properties
│   │   ├── impls offsets[property.name] = { offset, type: property.type }
│   │   ├── calls plyScalarTypeSize({ type: property.type })
│   │   └── impls offset += that size
│   ├── if any of offsets.x, offsets.y and offsets.z is undefined
│   │   └── throw PLY vertex coordinates are missing
│   ├── impls propertyOffsets = { stride: offset, and the x, y, z, red, green and blue offsets }  # impls-node-one-step:skip — one record's fields
│   └── return propertyOffsets
├── function plyScalarTypeSize({ type }: { type: string }): number
│   ├── # Maps a PLY scalar type name to its size in bytes.
│   ├── impls scalarTypeSizes = the table from each PLY scalar type name to its byte size
│   ├── impls size = scalarTypeSizes[type]
│   ├── if size is undefined
│   │   └── throw unsupported PLY scalar type ${type}
│   └── return size
├── function writeGeometryVertex({ positions, colors, index, x, y, z, red, green, blue, }: { positions: Float32Array; colors: Float32Array; index: number; x: number; y: number; z: number; red: number; green: number; blue: number; }): void
│   ├── # Writes one vertex's coordinates and normalized colors into the attribute arrays at its own offset.
│   ├── impls positionOffset = index * 3
│   ├── impls positions[positionOffset] = x
│   ├── impls positions[positionOffset + 1] = y
│   ├── impls positions[positionOffset + 2] = z
│   ├── calls normalizeColorComponent({ value: red })    # -> colors[positionOffset]
│   ├── calls normalizeColorComponent({ value: green })  # -> colors[positionOffset + 1]
│   └── calls normalizeColorComponent({ value: blue })   # -> colors[positionOffset + 2]
├── function normalizeColorComponent({ value }: { value: number }): number
│   ├── # Normalizes one raw color component into the unit range, reading a non-finite one as 0.7.
│   ├── if value is not finite
│   │   └── return 0.7
│   ├── if value <= 1
│   │   ├── impls component = value clamped into [0, 1]
│   │   └── return component
│   ├── impls component = value / 255 clamped into [0, 1]
│   └── return component
├── function createPointBufferGeometry({ positions, colors, }: { positions: Float32Array; colors: Float32Array; }): THREE.BufferGeometry
│   ├── # Builds the BufferGeometry over the filled position and color arrays and computes its bounds.
│   ├── impls geometry = new THREE.BufferGeometry()
│   ├── impls geometry.setAttribute("position", the 3-component buffer over positions)
│   ├── impls geometry.setAttribute("color", the 3-component buffer over colors)
│   ├── impls geometry.computeBoundingSphere()
│   ├── impls geometry.computeBoundingBox()
│   └── return geometry
├── function readAsciiColorComponent({ parts, index, }: { parts: string[]; index: number; }): number
│   ├── # Reads one ASCII color component, standing in 180 where the header declares that channel none.
│   ├── if index < 0
│   │   └── return 180
│   ├── impls component = Number(parts[index])
│   └── return component
├── function readBinaryColorComponent({ view, base, offset, }: { view: DataView; base: number; offset: PlyPropertyOffset | undefined; }): number
│   ├── # Reads one binary color component, standing in 180 where the header declares that channel none.
│   ├── if offset is undefined
│   │   └── return 180
│   ├── calls readBinaryScalar({ view, offset: base + offset.offset, type: offset.type })
│   └── return  # that scalar
├── function readBinaryScalar({ view, offset, type, }: { view: DataView; offset: number; type: string; }): number
│   ├── # Reads one little-endian scalar of the named PLY type at a byte offset.
│   ├── if type is "char" or "int8"
│   │   ├── impls scalar = the signed byte at offset
│   │   └── return scalar
│   ├── if type is "uchar" or "uint8"
│   │   ├── impls scalar = the unsigned byte at offset
│   │   └── return scalar
│   ├── if type is "short" or "int16"
│   │   ├── impls scalar = the signed 16-bit value at offset
│   │   └── return scalar
│   ├── if type is "ushort" or "uint16"
│   │   ├── impls scalar = the unsigned 16-bit value at offset
│   │   └── return scalar
│   ├── if type is "int" or "int32"
│   │   ├── impls scalar = the signed 32-bit value at offset
│   │   └── return scalar
│   ├── if type is "uint" or "uint32"
│   │   ├── impls scalar = the unsigned 32-bit value at offset
│   │   └── return scalar
│   ├── if type is "float" or "float32"
│   │   ├── impls scalar = the 32-bit float at offset
│   │   └── return scalar
│   ├── if type is "double" or "float64"
│   │   ├── impls scalar = the 64-bit float at offset
│   │   └── return scalar
│   └── throw unsupported PLY scalar type ${type}
└── function renderPointsScene({ scene, camera, renderer, controls, }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ThreeTrackballCameraControls; }): void
    ├── # Drives the point-cloud render loop with the supplied trackball controls.
    └── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
```
