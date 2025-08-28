import os
import json
import xxhash
import random
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from dash import html
import plotly.graph_objects as go
from data.cache.combined_dataset_cache import CombinedDatasetCache
from data.dataloaders.base_dataloader import BaseDataLoader
from data.datasets.index_dataset import IndexDataset
from utils.ops.apply import apply_tensor_op


class PCRCachedCollator:
    """Picklable collator wrapper for PCR dataloader with caching functionality."""
    
    def __init__(self, original_dataset, collator, cache, device=torch.device('cuda')):
        self.original_dataset = original_dataset
        self.collator = collator
        self.cache = cache
        self.device = device
    
    def __call__(self, datapoints: List[int]):
        assert isinstance(datapoints, list)
        assert len(datapoints) == 1
        assert isinstance(datapoints[0], int)
        key = datapoints[0]
        assert self.cache is not None
        cached_result = self.cache.get(key)
        if cached_result is not None:
            return apply_tensor_op(func=lambda x: x.to(self.device), inputs=cached_result)
        else:
            actual_datapoints = [self.original_dataset[idx] for idx in datapoints]
            batched_datapoints = self.collator(actual_datapoints)
            self.cache.put(key, batched_datapoints)
            return batched_datapoints


class PCRDataloader(BaseDataLoader):

    def __init__(
        self,
        dataset,
        collator,
        use_cpu_cache: bool = True,
        use_disk_cache: bool = True,
        max_cache_memory_percent: float = 80.0,
        enable_cpu_validation: bool = False,
        enable_disk_validation: bool = False,
        device=torch.device('cuda'),
        **kwargs,
    ) -> None:
        self._init_cache(
            dataset=dataset,
            collator=collator,
            use_cpu_cache=use_cpu_cache,
            use_disk_cache=use_disk_cache,
            max_cache_memory_percent=max_cache_memory_percent,
            enable_cpu_validation=enable_cpu_validation,
            enable_disk_validation=enable_disk_validation,
        )
        if self.cache is not None:
            # Using index dataset avoids loading datapoint in case a batched datapoint is already cached
            index_dataset = IndexDataset(size=len(dataset))
            cached_collator = PCRCachedCollator(original_dataset=dataset, collator=collator, cache=self.cache, device=device)
            super().__init__(dataset=index_dataset, collate_fn=cached_collator, **kwargs)
        else:
            super().__init__(dataset=dataset, collate_fn=collator, **kwargs)

    def _init_cache(
        self,
        dataset,
        collator,
        use_cpu_cache: bool,
        use_disk_cache: bool,
        max_cache_memory_percent: float,
        enable_cpu_validation: bool,
        enable_disk_validation: bool,
    ) -> None:
        assert isinstance(use_cpu_cache, bool), f"{type(use_cpu_cache)=}"
        assert isinstance(use_disk_cache, bool), f"{type(use_disk_cache)=}"
        assert isinstance(max_cache_memory_percent, float), f"{type(max_cache_memory_percent)=}"
        assert 0.0 <= max_cache_memory_percent <= 100.0, f"{max_cache_memory_percent=}"
        
        if use_cpu_cache or use_disk_cache:
            # Generate version hash for this dataset configuration
            version_hash = self.get_cache_version_hash(dataset, collator)
            
            # For datasets without data_root (e.g., random datasets), use a default location
            # For datasets with soft links, resolve to real path to ensure cache is in target location (e.g., /pub not /home)
            if hasattr(dataset, 'data_root'):
                data_root_for_cache = dataset.data_root
                if os.path.islink(data_root_for_cache):
                    data_root_for_cache = os.path.realpath(data_root_for_cache)
            else:
                # Use dataset class name for default location when no data_root is provided
                data_root_for_cache = f'/tmp/cache/{dataset.__class__.__name__.lower()}'
            
            self.cache = CombinedDatasetCache(
                data_root=data_root_for_cache,
                version_hash=version_hash,
                use_cpu_cache=use_cpu_cache,
                use_disk_cache=use_disk_cache,
                max_cpu_memory_percent=max_cache_memory_percent,
                enable_cpu_validation=enable_cpu_validation,
                enable_disk_validation=enable_disk_validation,
                dataset_class_name=dataset.__class__.__name__,
                version_dict=self._get_cache_version_dict(dataset, collator),
            )
        else:
            self.cache = None
    
    def _get_cache_version_dict(self, dataset, collator) -> Dict[str, Any]:
        """Return parameters that affect dataloader cache content for cache versioning.
        
        Base implementation provides common fields. Subclasses should call super()
        and add their specific parameters.
        
        Args:
            dataset: The dataset being used
            collator: The collator being used
            
        Returns:
            Dict containing version parameters for this PCR dataloader configuration
        """
        return {
            'dataloader_class': self.__class__.__name__,
            'dataset_version': dataset.get_cache_version_hash(),
        }
    
    def get_cache_version_hash(self, dataset, collator):
        """Generate deterministic hash from dataloader configuration."""
        version_dict = self._get_cache_version_dict(dataset, collator)
        hash_str = json.dumps(version_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()[:16]

    @staticmethod
    def create_correspondence_visualization(
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        radius: float = 0.1,
        point_size: float = 2,
        point_opacity: float = 0.8,
        camera_state: Optional[Dict[str, Any]] = None,
        lod_type: str = "continuous",
        density_percentage: int = 100,
        point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    ) -> go.Figure:
        """Create a visualization of correspondences between transformed source and target point clouds.

        Args:
            src_points: Transformed source point cloud [N, 3] or [1, N, 3]
            tgt_points: Target point cloud [M, 3] or [1, M, 3]
            radius: Radius for finding correspondences
            point_size: Size of points in visualization
            point_opacity: Opacity of points in visualization
            camera_state: Optional dictionary containing camera position state
            lod_type: Type of LOD ("continuous", "discrete", or "none")
            density_percentage: Percentage of points to display when lod_type is "none" (1-100)
            point_cloud_id: Unique identifier for LOD caching

        Returns:
            Plotly figure showing the correspondence visualization
        """
        # Normalize points to unbatched format
        src_points_normalized = _normalize_points(src_points)
        tgt_points_normalized = _normalize_points(tgt_points)
        
        src_points_np = src_points_normalized.cpu().numpy()
        tgt_points_np = tgt_points_normalized.cpu().numpy()

        # Find correspondences based on radius
        correspondences = get_correspondences(src_points_normalized, tgt_points_normalized, None, radius)

        # Create figure with both point clouds
        corr_fig = create_point_cloud_display(
            points=src_points_normalized,
            title="Point Cloud Correspondences",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            density_percentage=density_percentage,
            point_cloud_id=point_cloud_id,
        )

        # Add target points
        corr_fig.add_trace(go.Scatter3d(
            x=tgt_points_np[:, 0],
            y=tgt_points_np[:, 1],
            z=tgt_points_np[:, 2],
            mode='markers',
            marker=dict(size=point_size, color='red', opacity=point_opacity),
            name='Target Points'
        ))

        # Create list of correspondence line traces
        correspondence_traces = []
        for src_idx, tgt_idx in correspondences:
            src_point = src_points_np[src_idx]
            tgt_point = tgt_points_np[tgt_idx]
            correspondence_traces.append(go.Scatter3d(
                x=[src_point[0], tgt_point[0]],
                y=[src_point[1], tgt_point[1]],
                z=[src_point[2], tgt_point[2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))

        if len(correspondence_traces) > 10:
            correspondence_traces = random.sample(correspondence_traces, 10)

        # Add all correspondence traces at once
        if correspondence_traces:
            corr_fig.add_traces(correspondence_traces)

        return corr_fig

    @staticmethod 
    def display_batched_datapoint(
        datapoint: Dict[str, Any],
        point_size: float = 2,
        point_opacity: float = 0.8,
        camera_state: Optional[Dict[str, Any]] = None,
        sym_diff_radius: float = 0.05,
        lod_type: str = "continuous",
        density_percentage: int = 100
    ) -> html.Div:
        """Display a batched point cloud registration datapoint.

        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info
            point_size: Size of points in visualization
            point_opacity: Opacity of points in visualization
            camera_state: Optional dictionary containing camera position state
            sym_diff_radius: Radius for computing symmetric difference
            lod_type: Type of LOD ("continuous", "discrete", or "none")

        Returns:
            html.Div containing the visualization
        """
        from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
        from data.viewer.utils.structure_validation import validate_pcr_structure
        from data.viewer.utils.atomic_displays.point_cloud_display import create_point_cloud_display, build_point_cloud_id
        from data.viewer.utils.display_utils import DisplayStyles, ParallelFigureCreator, create_figure_grid
        
        # Validate structure and inputs (includes all basic validation)
        validate_pcr_structure(datapoint)
        
        inputs = datapoint['inputs']
        all_figures = []

        # Process each level in the hierarchy
        for level in range(len(inputs['points'])):
            # Split points into source and target
            src_points, tgt_points = BasePCRDataset.split_points_by_lengths(
                inputs['points'][level], inputs.get('lengths', inputs['stack_lengths'])[level],
            )

            # For top level (level 0), show all visualizations
            if level == 0:
                figure_tasks = [
                    lambda src=src_points, lvl=level: create_point_cloud_display(
                        points=src,
                        title=f"Source Point Cloud (Level {lvl})",
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"source_batch_{lvl}"),
                    ),
                    lambda tgt=tgt_points, lvl=level: create_point_cloud_display(
                        points=tgt,
                        title=f"Target Point Cloud (Level {lvl})",
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"target_batch_{lvl}"),
                    ),
                    lambda src=src_points, tgt=tgt_points, lvl=level: BasePCRDataset._create_union_with_title(
                        src_points=src,
                        tgt_points=tgt,
                        title=f"Union (Level {lvl})",
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"union_batch_{lvl}")
                    ),
                    lambda src=src_points, tgt=tgt_points, lvl=level: BasePCRDataset._create_sym_diff_with_title(
                        src_points=src,
                        tgt_points=tgt,
                        title=f"Symmetric Difference (Level {lvl})",
                        radius=sym_diff_radius,
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"sym_diff_batch_{lvl}")
                    ),
                ]

                # Create figures in parallel using centralized utility
                figure_creator = ParallelFigureCreator(max_workers=4, enable_timing=False)
                level_figures = figure_creator.create_figures_parallel(figure_tasks)
                all_figures.extend(level_figures)

                # TODO: Add correspondence visualization
                # corr_fig = BasePCRDataset.create_correspondence_visualization(
                #     src_points, tgt_points, radius=corr_radius, point_size=point_size,
                #     point_opacity=point_opacity, camera_state=camera_state,
                # )
                # corr_fig.update_layout(title=f"Point Cloud Correspondences (Level {level})")
                # all_figures.append(corr_fig)
            else:
                # For lower levels, only show source and target
                all_figures.extend([
                    create_point_cloud_display(
                        points=src_points,
                        title=f"Source Point Cloud (Level {level})",
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"source_batch_{level}"),
                    ),
                    create_point_cloud_display(
                        points=tgt_points,
                        title=f"Target Point Cloud (Level {level})",
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"target_batch_{level}"),
                    )
                ])

        # Create grid layout using centralized utilities
        grid_items = create_figure_grid(all_figures, width_style="50%", height_style="520px")
        
        return html.Div([
            html.H3("Point Cloud Registration Visualization (Hierarchical)"),
            html.Div(grid_items, style=DisplayStyles.FLEX_WRAP),
            BasePCRDataset._create_meta_info_section(datapoint.get('meta_info', {}))
        ])
