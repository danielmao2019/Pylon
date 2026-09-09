# PCR Datasets Code Structure

## Code structure trees

`data/datasets/pcr_datasets/base_pcr_dataset.py`

```text
base_pcr_dataset.py
├── from typing import Any, Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import plotly.graph_objects as go
├── import torch
├── from dash import html
├── from data.datasets.base_dataset import BaseDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.display_utils import DisplayStyles, ParallelFigureCreator, create_figure_grid
├── from data.viewer.utils.displays.points.dash.core_points_display import build_point_cloud_id, create_point_cloud_display, get_point_cloud_display_stats
├── from data.viewer.utils.structure_validation import validate_pcr_structure
├── from models.three_d.point_cloud.ops import apply_transform
├── from models.three_d.point_cloud.ops.apply_transform import _normalize_transform
├── from models.three_d.point_cloud.ops.set_ops import pc_symmetric_difference
├── from models.three_d.point_cloud.ops.set_ops.symmetric_difference import _normalize_points
└── class BasePCRDataset(BaseDataset)
    ├── # The registration-pair base every PCR dataset inherits, which owns the viewer display for one pair rather than any loading path of its own.
    ├── INPUT_NAMES  # the two point-cloud keys a registration pair's inputs carry
    ├── LABEL_NAMES  # the one key its label carries, the ground-truth transform
    ├── @staticmethod def display_datapoint(datapoint: Dict[str, Any], class_labels: Optional[Dict[str, List[str]]] = None, camera_state: Optional[Dict[str, Any]] = None, settings_3d: Optional[Dict[str, Any]] = None) -> html.Div [override]
    │   ├── # The viewer's whole page for one pair, which is a figure grid over the two clouds and the union and difference views, under the transform, statistics and meta-info sections.
    │   ├── assert datapoint is not None
    │   ├── assert datapoint is a dict
    │   ├── calls validate_pcr_structure(datapoint)
    │   ├── impls inputs = datapoint['inputs']
    │   ├── impls src_pc = inputs['src_pc']
    │   ├── assert src_pc is a PointCloud
    │   ├── impls tgt_pc = inputs['tgt_pc']
    │   ├── assert tgt_pc is a PointCloud
    │   ├── impls point_size = 2
    │   ├── impls point_opacity = 0.8
    │   ├── impls sym_diff_radius = 0.05
    │   ├── impls lod_type = "continuous"
    │   ├── impls density_percentage = 100
    │   ├── if settings_3d is not None
    │   │   ├── assert settings_3d is a dict
    │   │   ├── impls point_size = what settings_3d holds under 'point_size', else the default just set
    │   │   ├── impls point_opacity = what settings_3d holds under 'point_opacity', else the default just set
    │   │   ├── impls sym_diff_radius = what settings_3d holds under 'sym_diff_radius', else the default just set
    │   │   ├── impls lod_type = what settings_3d holds under 'lod_type', else the default just set
    │   │   └── impls density_percentage = what settings_3d holds under 'density_percentage', else the default just set
    │   ├── impls src_xyz = src_pc.xyz
    │   ├── impls tgt_xyz = tgt_pc.xyz
    │   ├── impls transform = what datapoint['labels'] holds under 'transform'
    │   ├── if transform is None
    │   │   └── impls transform = the 4x4 identity  # the identity stands in when the label carries no pose
    │   ├── calls apply_transform(src_xyz, transform)
    │   ├── impls src_pc_transformed = the posed source coordinates
    │   ├── impls all_points = src_xyz, tgt_xyz, src_pc_transformed
    │   ├── impls x_coords, y_coords, z_coords = each axis column of all_points concatenated
    │   ├── impls padding = 0.05
    │   ├── impls x_range_unified = the smallest x_coords value to the largest
    │   ├── impls y_range_unified = the smallest y_coords value to the largest
    │   ├── impls z_range_unified = the smallest z_coords value to the largest
    │   ├── impls x_pad = the x_range_unified span scaled by padding
    │   ├── impls y_pad = the y_range_unified span scaled by padding
    │   ├── impls z_pad = the z_range_unified span scaled by padding
    │   ├── impls unified_axis_ranges = the three ranges, each widened by its own pad at both ends  # one range shared by every figure, so the four views stay comparable
    │   ├── impls figure_tasks = one zero-argument lambda per view, deferring each build below to the pool
    │   ├── calls create_point_cloud_display(pc=src_pc, color_key=None, highlight_indices=None, title="Source Point Cloud", point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=build_point_cloud_id(datapoint, "source"), axis_ranges=unified_axis_ranges)
    │   ├── calls create_point_cloud_display(pc=tgt_pc, color_key=None, highlight_indices=None, title="Target Point Cloud", point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=build_point_cloud_id(datapoint, "target"), axis_ranges=unified_axis_ranges)
    │   ├── calls BasePCRDataset.create_union_visualization(src_pc_transformed, tgt_xyz, point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, point_cloud_id=build_point_cloud_id(datapoint, "union"), density_percentage=density_percentage, axis_ranges=unified_axis_ranges)
    │   ├── calls BasePCRDataset.create_symmetric_difference_visualization(src_pc_transformed, tgt_xyz, radius=sym_diff_radius, point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, point_cloud_id=build_point_cloud_id(datapoint, "sym_diff"), density_percentage=density_percentage, axis_ranges=unified_axis_ranges)
    │   ├── if 'correspondences' in inputs
    │   │   ├── impls correspondences = inputs['correspondences']
    │   │   ├── calls BasePCRDataset.create_correspondence_visualization(src_pc_transformed, tgt_xyz, correspondences=correspondences, point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=build_point_cloud_id(datapoint, "correspondences"), title="Point Cloud Correspondences")
    │   │   └── impls figure_tasks gains that view's lambda as a fifth task
    │   ├── calls ParallelFigureCreator(max_workers=4, enable_timing=False)
    │   ├── impls figure_creator = the four-worker pool it built
    │   ├── calls figure_creator.create_figures_parallel(figure_tasks)
    │   ├── impls figures = the figures the pool produced
    │   ├── calls BasePCRDataset._compute_transform_info(transform)
    │   ├── impls transform_info = the pose summary it built
    │   ├── calls get_point_cloud_display_stats(inputs['src_pc'])
    │   ├── calls get_point_cloud_display_stats(inputs['tgt_pc'])
    │   ├── impls src_stats_dict, tgt_stats_dict = the two statistics mappings it measured
    │   ├── calls BasePCRDataset._dict_to_html_list(src_stats_dict)
    │   ├── calls BasePCRDataset._dict_to_html_list(tgt_stats_dict)
    │   ├── impls src_stats_children, tgt_stats_children = the two rendered statistics lists
    │   ├── calls create_figure_grid(figures, width_style="50%", height_style="520px")
    │   ├── impls grid_items = the half-width, 520-pixel-tall grid cells it built
    │   ├── calls BasePCRDataset._create_transform_info_section(transform_info)
    │   ├── calls BasePCRDataset._create_statistics_section(src_stats_children, tgt_stats_children)
    │   ├── impls layout_sections = a "Point Cloud Registration Visualization" heading, grid_items under DisplayStyles.FLEX_WRAP, the transform section, the statistics section
    │   ├── if 'correspondences' in inputs
    │   │   ├── impls correspondences = inputs['correspondences']
    │   │   ├── calls BasePCRDataset._create_correspondence_stats_section(correspondences)
    │   │   └── impls layout_sections gains that correspondence-statistics section
    │   ├── calls BasePCRDataset._create_meta_info_section(datapoint.get('meta_info', {}))
    │   ├── impls layout_sections gains that meta-info section last
    │   └── return  # layout_sections in one html.Div
    ├── @staticmethod def create_union_visualization(src_points: torch.Tensor, tgt_points: torch.Tensor, point_size: float = 2, point_opacity: float = 0.8, camera_state: Optional[Dict[str, Any]] = None, lod_type: str = "continuous", point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None, density_percentage: int = 100, axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None, title: str = "Union (Transformed Source + Target)") -> go.Figure
    │   ├── # Draws both clouds in one view, red against blue, which is how a viewer reads whether the pose aligns them.
    │   ├── calls _normalize_points(src_points)
    │   ├── calls _normalize_points(tgt_points)
    │   ├── impls src_points_normalized, tgt_points_normalized = the two unbatched blocks it returned
    │   ├── impls union_points = the two blocks concatenated
    │   ├── impls src_colors = a zeros block of one row per source point with its red channel set to one
    │   ├── impls tgt_colors = a zeros block of one row per target point with its blue channel set to one
    │   ├── impls union_colors = the two color blocks concatenated
    │   ├── calls PointCloud(xyz=union_points, data={'rgb': union_colors})
    │   ├── impls union_pc = the display cloud it built
    │   ├── calls create_point_cloud_display(pc=union_pc, color_key=None, highlight_indices=None, title=title, point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=point_cloud_id, axis_ranges=axis_ranges)
    │   └── return  # the figure it built
    ├── @staticmethod def create_symmetric_difference_visualization(src_points: torch.Tensor, tgt_points: torch.Tensor, radius: float = 0.05, point_size: float = 2, point_opacity: float = 0.8, camera_state: Optional[Dict[str, Any]] = None, lod_type: str = "continuous", point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None, density_percentage: int = 100, axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None, title: str = "Symmetric Difference") -> go.Figure
    │   ├── # Draws only the points each cloud holds beyond the other's radius, which is how a viewer reads what the pose left unexplained.
    │   ├── calls _normalize_points(src_points)
    │   ├── calls _normalize_points(tgt_points)
    │   ├── impls src_points_normalized, tgt_points_normalized = the two unbatched blocks it returned
    │   ├── calls pc_symmetric_difference(src_points_normalized, tgt_points_normalized, radius)
    │   ├── impls src_indices, tgt_indices = the two index blocks it returned
    │   ├── if either index block is non-empty
    │   │   ├── impls src_diff = the source rows src_indices names
    │   │   ├── impls tgt_diff = the target rows tgt_indices names
    │   │   ├── impls sym_diff_points = the two row blocks concatenated
    │   │   ├── impls src_colors = a zeros block of one row per src_diff point with its red channel set to one
    │   │   ├── impls tgt_colors = a zeros block of one row per tgt_diff point with its blue channel set to one
    │   │   ├── impls sym_diff_colors = the two color blocks concatenated
    │   │   ├── calls PointCloud(xyz=sym_diff_points, data={'rgb': sym_diff_colors})
    │   │   ├── impls sym_diff_pc = the display cloud it built
    │   │   ├── calls create_point_cloud_display(pc=sym_diff_pc, color_key=None, highlight_indices=None, title=title, point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=point_cloud_id, axis_ranges=axis_ranges)
    │   │   └── return  # the figure it built
    │   └── else
    │       ├── calls PointCloud(xyz=a single zero row on the normalized source device)
    │       ├── impls empty_pc = the one-point stand-in cloud it built
    │       ├── calls create_point_cloud_display(pc=empty_pc, color_key=None, highlight_indices=None, title=f"{title} (Empty)", point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=point_cloud_id, axis_ranges=axis_ranges)
    │       └── return  # the stand-in figure it built
    ├── @staticmethod def _compute_transform_info(transform: torch.Tensor) -> Dict[str, Any]
    │   ├── # Reduces a pose to the three things the transform panel states: its matrix text, its rotation angle, its translation magnitude.
    │   ├── calls _normalize_transform(transform, torch.Tensor, target_device=transform.device, target_dtype=transform.dtype)
    │   ├── impls transform_normalized = the unbatched [4, 4] pose it returned
    │   ├── impls rotation_matrix = the leading 3x3 block of transform_normalized
    │   ├── impls translation_vector = the leading three entries of transform_normalized's last column
    │   ├── impls trace = the trace of rotation_matrix
    │   ├── impls rotation_angle = the arccos of (trace - 1) / 2, carried to degrees
    │   ├── impls translation_magnitude = the norm of translation_vector
    │   ├── impls transform_str = the line "Transform Matrix:"
    │   ├── for each row index i below 4
    │   │   ├── impls row = that row's four entries, each at four decimal places
    │   │   └── impls transform_str gains row joined by two spaces, closed by a newline
    │   └── return  # transform_str, rotation_angle, translation_magnitude keyed by name
    ├── @staticmethod def _create_transform_info_section(transform_info: Dict[str, Any]) -> html.Div
    │   ├── # Renders that pose summary as the transform panel, which is where a viewer reads the pair's stated registration.
    │   ├── impls section_children = a "Transform Information:" heading, transform_info["transform_str"] preformatted, the rotation angle at two decimals in degrees, the translation magnitude at four decimals
    │   ├── impls section = section_children in one html.Div under a twenty-pixel top margin
    │   └── return section
    ├── @staticmethod def _create_statistics_section(src_stats_children: Any, tgt_stats_children: Any) -> html.Div
    │   ├── # Puts the two clouds' statistics side by side, so a viewer compares source against target in one read.
    │   ├── impls src_column = a "Source Point Cloud Statistics:" heading over src_stats_children, under DisplayStyles.GRID_ITEM_48_MARGIN
    │   ├── impls tgt_column = a "Target Point Cloud Statistics:" heading over tgt_stats_children, under DisplayStyles.GRID_ITEM_48_NO_MARGIN
    │   ├── impls section = the two columns in one html.Div under a twenty-pixel top margin
    │   └── return section
    ├── @staticmethod def _create_correspondence_stats_section(correspondences: torch.Tensor) -> html.Div
    │   ├── # States how many correspondence pairs the datapoint carries, the one number the correspondence figure itself cannot show.
    │   ├── impls num_correspondences = the row count of correspondences
    │   ├── impls stats_list = one list item reading that count, indented twenty pixels, margined five above
    │   ├── impls section = a "Correspondence Statistics:" heading over stats_list, in one html.Div under a twenty-pixel top margin
    │   └── return section
    ├── @staticmethod def _create_meta_info_section(meta_info: Dict[str, Any]) -> html.Div
    │   ├── # Renders whatever metadata the concrete dataset attached to the datapoint, whose keys vary per dataset.
    │   ├── if meta_info is empty
    │   │   ├── impls empty_section = a "Datapoint Meta Information:" heading over a "No meta information available" paragraph, in one html.Div under a twenty-pixel top margin
    │   │   └── return empty_section
    │   ├── calls BasePCRDataset._dict_to_html_list(meta_info)
    │   ├── impls meta_children = the nested rendering of the whole mapping
    │   ├── impls section = a "Datapoint Meta Information:" heading over meta_children, in one html.Div under a twenty-pixel top margin
    │   └── return section
    ├── @staticmethod def _dict_to_html_list(data: Dict[str, Any], key_name: str = None) -> html.Div
    │   ├── # Renders a statistics or metadata mapping as nested HTML lists, recursing into whatever sub-mappings it holds.
    │   ├── impls items = an empty list
    │   ├── if key_name
    │   │   └── impls items gains a heading reading key_name, margined fifteen pixels above, five below
    │   ├── impls list_items = an empty list
    │   ├── for each key, value in data
    │   │   ├── if value is a dict
    │   │   │   ├── calls BasePCRDataset._dict_to_html_list(value, key)
    │   │   │   └── impls items gains that nested rendering
    │   │   └── else
    │   │       ├── calls BasePCRDataset._format_value(key, value)
    │   │       ├── impls formatted_value = the display text it produced
    │   │       ├── if key is 'overlap'
    │   │       │   └── impls list_items gains a "key: formatted_value" item, bold in '#2E86AB'  # overlap is the PCR metric a viewer looks for first
    │   │       └── else
    │   │           └── impls list_items gains a plain "key: formatted_value" item
    │   ├── if list_items is non-empty
    │   │   └── impls items gains list_items as a list indented twenty pixels, margined five above
    │   └── return  # items in one html.Div
    ├── @staticmethod def _format_value(key: str, value: Any) -> str
    │   ├── # Formats one statistics value for display, taking its precision from its type, its degree suffix from the key's name.
    │   ├── if value is a three-element list of ints or floats
    │   │   ├── if 'angle' occurs in the lowercased key
    │   │   │   ├── impls formatted = the three entries at two decimals, each suffixed with a degree sign, comma-separated inside square brackets
    │   │   │   └── return formatted
    │   │   └── else
    │   │       ├── impls formatted = the three entries at four decimals, comma-separated inside square brackets
    │   │       └── return formatted
    │   ├── elif value is a float
    │   │   ├── impls formatted = value at four decimals
    │   │   └── return formatted
    │   └── else
    │       ├── impls formatted = value as its plain string
    │       └── return formatted
    └── @staticmethod def create_correspondence_visualization(src_points: torch.Tensor, tgt_points: torch.Tensor, correspondences: torch.Tensor, point_size: float = 2, point_opacity: float = 0.8, camera_state: Optional[Dict[str, Any]] = None, lod_type: str = "continuous", density_percentage: int = 100, point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None, title: str = "Point Cloud Correspondences") -> go.Figure
        ├── # Draws the two clouds pushed apart side by side, with dashed lines joining a sample of the corresponding points.
        ├── calls _normalize_points(src_points)
        ├── calls _normalize_points(tgt_points)
        ├── impls src_points_normalized, tgt_points_normalized = the two unbatched blocks it returned
        ├── impls src_points_np, tgt_points_np = the two normalized blocks on cpu as numpy
        ├── impls correspondences_np = the correspondences argument itself on cpu as numpy
        ├── impls src_bounds = the per-axis smallest-to-largest extent of src_points_np
        ├── impls tgt_bounds = the per-axis smallest-to-largest extent of tgt_points_np
        ├── impls src_width = the x extent of src_bounds
        ├── impls tgt_width = the x extent of tgt_bounds
        ├── impls gap = three tenths of the wider of those two widths
        ├── impls x_offset = the shift putting tgt_bounds' left edge one gap past src_bounds' right edge
        ├── impls tgt_points_offset = a copy of tgt_points_np displaced along x by x_offset
        ├── impls fig = an empty plotly figure
        ├── impls fig gains a legended blue "Source Points" scatter3d over src_points_np at point_size, point_opacity
        ├── impls fig gains a legended red "Target Points" scatter3d over tgt_points_offset at point_size, point_opacity
        ├── if correspondences_np is non-empty
        │   ├── impls max_correspondences = 50
        │   ├── if correspondences_np holds more rows than max_correspondences
        │   │   ├── impls sample_indices = max_correspondences row indices drawn without replacement
        │   │   └── impls correspondences_display = the sampled rows  # only a sample is drawn, so the connecting lines stay readable
        │   ├── else
        │   │   └── impls correspondences_display = every row
        │   ├── impls src_corr_indices = the first column of correspondences_display as ints
        │   ├── impls tgt_corr_indices = the second column of correspondences_display as ints
        │   ├── impls src_corr_points = the src_points_np rows src_corr_indices names
        │   ├── impls tgt_corr_points_offset = the tgt_points_offset rows tgt_corr_indices names
        │   ├── impls fig gains a legended cyan scatter3d over src_corr_points named "Source Correspondences" with the drawn count, its markers half again point_size at full opacity
        │   ├── impls fig gains a legended yellow scatter3d over tgt_corr_points_offset named "Target Correspondences" with the drawn count, its markers half again point_size at full opacity
        │   └── for each i below the count of correspondences_display
        │       ├── impls src_point = src_corr_points[i]
        │       ├── impls tgt_point = tgt_corr_points_offset[i]
        │       └── impls fig gains a dashed green two-point line trace joining those, kept off the legend, kept off hover
        ├── impls fig's layout set to the title carrying the total correspondence count, x/y/z axis titles, data aspect mode, the legend shown, a 1000 by 600 size
        ├── if camera_state is not None
        │   └── impls fig's scene camera set to camera_state
        └── return  # fig
```

`data/datasets/pcr_datasets/kitti_dataset.py`

```text
kitti_dataset.py
├── from typing import Any, Dict, Tuple
├── import os
├── import numpy as np
├── import torch
├── import open3d as o3d
├── from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class KITTIDataset(BasePCRDataset)
    ├── # KITTI odometry scan pairs: two velodyne sweeps far enough apart to register, their ground truth refined by ICP.
    └── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]] [override]
        ├── # Loads one scan pair and its ground truth, refining that ground truth by ICP the first time the pair is asked for.
        ├── impls ann = self.annotations[idx]
        ├── impls seq = the sequence ann names
        ├── impls t0 = the first frame ann names
        ├── impls t1 = the second frame ann names
        ├── calls self.get_video_odometry(seq, indices=[t0, t1])
        ├── calls self.odometry_to_positions(two_odo)
        ├── impls two_pos = the two poses it built
        ├── impls fname0 = the velodyne file frame t0 names under this sequence
        ├── impls fname1 = the velodyne file frame t1 names under this sequence
        ├── impls xyzr0 = fname0 read as float32 and reshaped to four columns  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── impls xyzr1 = fname1 read as float32 and reshaped to four columns  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── impls xyz0 = the leading three columns of xyzr0
        ├── impls xyz1 = the leading three columns of xyzr1
        ├── impls icp_cache_file = the cache file this sequence and frame pair names  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── if icp_cache_file does not exist
        │   ├── impls M = the odometry-implied source-to-target pose, carried through self.velo2cam at both ends
        │   ├── calls self.apply_transform(xyz0, M)
        │   ├── impls xyz0_t = the seeded source coordinates
        │   ├── impls pcd0 = xyz0_t as a grey open3d cloud
        │   ├── impls pcd1 = xyz1 as a green open3d cloud
        │   ├── impls reg = point-to-point ICP over those two at a fifth-metre threshold, up to two hundred iterations
        │   ├── impls gt_transform = M composed with what ICP returned  # the odometry pose is only a seed, so the pair's stated ground truth is what ICP settles on
        │   └── impls gt_transform saved to icp_cache_file
        ├── else
        │   └── impls gt_transform = what that cache file holds
        ├── calls PointCloud(xyz=torch.from_numpy(xyz0).float().to(self.device), data={'reflectance': the fourth column of xyzr0 as a float32 [N, 1], 'feat': a ones column})
        ├── calls PointCloud(xyz=torch.from_numpy(xyz1).float().to(self.device), data={'reflectance': the fourth column of xyzr1 as a float32 [N, 1], 'feat': a ones column})
        ├── impls src_pc, tgt_pc = the two clouds it built
        └── return  # those two clouds, gt_transform as a float32 label, and the sequence and frames as meta info
```

`data/datasets/pcr_datasets/modelnet40_dataset.py`

```text
modelnet40_dataset.py
├── import os
├── import glob
├── from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
└── class ModelNet40Dataset(SyntheticTransformPCRDataset)
    ├── # ModelNet40 objects self-registered: one OFF file is loaded as both halves of a pair, and the synthetic base poses and crops them apart.
    └── def _init_annotations(self) -> None [override]
        ├── # Names one annotation per OFF file, both halves of the pair pointing at that same file.
        ├── impls split_dir = self.split
        ├── if self.split is val
        │   └── impls split_dir = test  # ModelNet40 has no val split of its own, so a val run reads the test files
        ├── impls off_files = an empty list
        ├── for each category in self.CATEGORIES
        │   ├── impls category_dir = that category's directory under the data root for split_dir
        │   ├── if category_dir does not exist
        │   │   └── continue
        │   └── impls off_files gains the sorted OFF files under category_dir
        ├── impls self.annotations = an empty list
        ├── for each file_path in off_files
        │   ├── calls self.get_category_from_path(file_path)
        │   └── impls self.annotations gains that file as both source and target, under the category it named  # impls-node-one-step:skip — one step; the "and" names what it is made of
        └── impls the count found printed for the split
```

`data/datasets/pcr_datasets/synthetic_transform_pcr_dataset.py`

```text
synthetic_transform_pcr_dataset.py
├── from abc import ABC
├── from typing import Any, Dict, Optional, Tuple
├── import torch
├── from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
├── from data.structures.three_d.point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.ops.set_ops.intersection import compute_registration_overlap
├── from utils.determinism.hash_utils import deterministic_hash
└── class SyntheticTransformPCRDataset(BasePCRDataset, ABC)
    ├── # Builds a registration pair out of one cloud by posing and cropping it, trying sampled transforms until the overlap lands in the range asked for.
    ├── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, Any], Dict[str, Any]] [override]
    │   ├── # Loads one synthetic pair, which is whichever trial of this index first produced an overlap in range.
    │   ├── impls annotation = self.annotations[idx]
    │   ├── assert annotation is to name both source paths
    │   ├── assert this subclass implements _apply_crop
    │   ├── impls t1_pc_filepath, t2_pc_filepath = the two paths that annotation names
    │   ├── calls self._search_or_generate(t1_pc_filepath=t1_pc_filepath, t2_pc_filepath=t2_pc_filepath, idx=idx)
    │   ├── impls src_pc, tgt_pc, overlap_ratio, transform_matrix, trial_idx = what it settled on
    │   ├── if src_pc carries no feat
    │   │   └── impls src_pc.feat = a float32 ones column of one entry per point
    │   ├── if tgt_pc carries no feat
    │   │   └── impls tgt_pc.feat = a float32 ones column of one entry per point
    │   └── return  # the two clouds, transform_matrix as the label, and the two paths, the trial index, the transform and the overlap as meta info
    ├── def _search_or_generate(self, t1_pc_filepath: str, t2_pc_filepath: str, idx: int) -> Tuple[PointCloud, PointCloud, float, torch.Tensor, int]
    │   ├── # Walks this index's trials in order and settles on the first transform whose overlap lands in the range asked for.
    │   ├── impls idx_key = idx as a string  # the trials cache is keyed by dataset index
    │   ├── with self.cache_lock
    │   │   ├── if idx_key is absent from self.trials_cache
    │   │   │   └── impls self.trials_cache[idx_key] = an empty list
    │   │   └── impls cached_overlaps = a copy of that list  # the loop reads the copy, so a concurrent append cannot move it underfoot
    │   ├── for each trial_idx below self.max_trials
    │   │   ├── calls deterministic_hash((idx, trial_idx))
    │   │   ├── impls trial_seed = the seed it derived
    │   │   ├── calls self._sample_transform(trial_seed)
    │   │   ├── impls transform_matrix = the pose that seed samples
    │   │   ├── if trial_idx is within cached_overlaps
    │   │   │   ├── impls cached_overlap = cached_overlaps[trial_idx]
    │   │   │   └── if cached_overlap is not None and falls in self.overlap_range, its low end exclusive
    │   │   │       ├── calls self._generate(t1_pc_filepath=t1_pc_filepath, t2_pc_filepath=t2_pc_filepath, transform_matrix=transform_matrix, idx=idx)
    │   │   │       ├── impls src_pc, tgt_pc, generated_overlap = the pair that trial rebuilds
    │   │   │       ├── assert generated_overlap is not None
    │   │   │       ├── assert generated_overlap matches cached_overlap to within a hundred-thousandth  # the cache stores overlaps, so replaying a trial has to reproduce the one it stored
    │   │   │       ├── impls the cached trial printed for this datapoint
    │   │   │       └── return  # that pair, generated_overlap, transform_matrix and trial_idx
    │   │   └── else
    │   │       ├── calls self._generate(t1_pc_filepath=t1_pc_filepath, t2_pc_filepath=t2_pc_filepath, transform_matrix=transform_matrix, idx=idx)
    │   │       ├── impls src_pc, tgt_pc, overlap_ratio = the pair that trial builds
    │   │       ├── with self.cache_lock
    │   │       │   ├── impls cache_list = self.trials_cache[idx_key]
    │   │       │   ├── assert cache_list holds exactly trial_idx entries
    │   │       │   ├── impls cache_list gains overlap_ratio
    │   │       │   └── if self.cache_filepath is not None
    │   │       │       └── calls self._save_trials_cache()
    │   │       └── if overlap_ratio is not None and falls in self.overlap_range, its low end exclusive
    │   │           ├── impls the newly generated trial printed for this datapoint
    │   │           └── return  # that pair, overlap_ratio, transform_matrix and trial_idx
    │   └── raise RuntimeError  # no trial under self.max_trials produced an overlap in range
    └── def _generate(self, t1_pc_filepath: str, t2_pc_filepath: str, transform_matrix: torch.Tensor, idx: int) -> Tuple[PointCloud, PointCloud, Optional[float]]
        ├── # Runs one trial of that search, which is this transform posing the pair apart, the crop, and the overlap that survives it.
        ├── calls load_point_cloud(t1_pc_filepath, device=self.device, dtype=torch.float32)
        ├── calls load_point_cloud(t2_pc_filepath, device=self.device, dtype=torch.float32)
        ├── impls t1_pc_data, t2_pc_data = the two clouds it loaded
        ├── calls self._apply_transform(t1_pc_data, t2_pc_data, transform_matrix)
        ├── impls src_pc_transformed, tgt_pc_original = the first cloud carried by the inverse pose and the second left where it was  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── calls self._apply_crop(idx, src_pc_transformed)
        ├── calls self._apply_crop(idx, tgt_pc_original)
        ├── impls src_pc, tgt_pc = the two cropped clouds
        ├── if either cropped cloud has no points
        │   └── impls overlap_ratio = None
        ├── else
        │   ├── calls compute_registration_overlap(ref_points=tgt_pc.xyz, src_points=src_pc.xyz, transform=transform_matrix, positive_radius=self.matching_radius * 2)
        │   └── impls overlap_ratio = the overlap it measured
        └── return  # the two cropped clouds and overlap_ratio
```

`data/datasets/pcr_datasets/threedmatch_dataset.py`

```text
threedmatch_dataset.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
├── from data.structures.three_d.point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── class _ThreeDMatchBaseDataset(BasePCRDataset)
│   ├── # The annotated fragment pairs both 3DMatch splits are drawn from, the split itself being an overlap band the subclass names.
│   └── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]] [override]
│       ├── # Loads one fragment pair and inverts the annotation's pose, since the metadata states target-to-source and a datapoint states source-to-target.
│       ├── impls annotation = self.annotations[idx]
│       ├── calls load_point_cloud(annotation['src_path'], device=self.device)
│       ├── calls load_point_cloud(annotation['tgt_path'], device=self.device)
│       ├── impls src_pc, tgt_pc = the two clouds it loaded
│       ├── impls each of them gains a float32 ones column as feat
│       ├── impls transform_tgt_to_src = the float32 [4, 4] the annotation's rotation and translation make  # impls-node-one-step:skip — one step; the "and" names what it is made of
│       ├── impls transform = the inverse of transform_tgt_to_src
│       └── return  # the two clouds, transform as the label, and the paths, scene name, overlap and two frame ids as meta info
├── class ThreeDMatchDataset(_ThreeDMatchBaseDataset)
│   ├── # The 3DMatch split of those pairs, which is the ones overlapping by more than three tenths.
│   ├── DATASET_SIZE  # split name -> the pair count that band leaves
│   └── def __init__(self, **kwargs) -> None
│       ├── # Builds the base over this split's overlap band.
│       └── calls super().__init__(overlap_min=0.3, overlap_max=1.0, **kwargs)
└── class ThreeDLoMatchDataset(_ThreeDMatchBaseDataset)
    ├── # The 3DLoMatch split of those same pairs, which is the harder band from a tenth to three tenths.
    ├── DATASET_SIZE  # split name -> the pair count that band leaves
    └── def __init__(self, **kwargs) -> None
        ├── # Builds the base over this split's overlap band.
        └── calls super().__init__(overlap_min=0.1, overlap_max=0.3, **kwargs)
```
