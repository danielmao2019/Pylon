import os
import json
import xxhash
import torch
from typing import List, Dict, Any, Optional, Tuple
from dash import html
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
    def split_points_by_lengths(points: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split concatenated points into source and target using lengths.

        Args:
            points: Concatenated points tensor [src_points, tgt_points]
            lengths: Lengths tensor indicating split point

        Returns:
            Tuple of (source_points, target_points)
        """
        total_length = lengths[0]
        src_points = points[:total_length//2]
        tgt_points = points[total_length//2:total_length]
        return src_points, tgt_points

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
            src_points, tgt_points = PCRDataloader.split_points_by_lengths(
                inputs['points'][level], inputs.get('lengths', inputs['stack_lengths'])[level],
            )

            # For top level (level 0), show all visualizations
            if level == 0:
                # Get transform and apply it to source points
                assert 'labels' in datapoint, "datapoint must have 'labels' for transformation"
                assert 'transform' in datapoint['labels'], "datapoint['labels'] must have 'transform' for transformation"
                transform = datapoint['labels']['transform']
                assert isinstance(transform, torch.Tensor), f"transform must be torch.Tensor, got {type(transform)}"
                
                # Transform source points
                from utils.point_cloud_ops import apply_transform
                src_points_transformed = apply_transform(src_points, transform)
                
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
                    lambda src_transformed=src_points_transformed, tgt=tgt_points, lvl=level: BasePCRDataset.create_union_visualization(
                        src_points=src_transformed,  # Use transformed source points
                        tgt_points=tgt,
                        title=f"Union (Level {lvl})",
                        point_size=point_size,
                        point_opacity=point_opacity,
                        camera_state=camera_state,
                        lod_type=lod_type,
                        density_percentage=density_percentage,
                        point_cloud_id=build_point_cloud_id(datapoint, f"union_batch_{lvl}")
                    ),
                    lambda src_transformed=src_points_transformed, tgt=tgt_points, lvl=level: BasePCRDataset.create_symmetric_difference_visualization(
                        src_points=src_transformed,  # Use transformed source points
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

                # Add correspondence visualization if correspondences are available (only for level 0)
                if 'correspondences' in inputs:
                    correspondences = inputs['correspondences']
                    # Validate correspondences format
                    assert isinstance(correspondences, torch.Tensor), f"correspondences must be torch.Tensor, got {type(correspondences)}"
                    assert correspondences.ndim == 2, f"correspondences must be 2D tensor, got {correspondences.ndim}D"
                    assert correspondences.shape[1] == 2, f"correspondences must have shape (N, 2), got shape {correspondences.shape}"
                    
                    figure_tasks.append(
                        lambda src_transformed=src_points_transformed, tgt=tgt_points, lvl=level, corr=correspondences: BasePCRDataset.create_correspondence_visualization(
                            src_points=src_transformed,  # Use transformed source points
                            tgt_points=tgt,
                            correspondences=corr,
                            point_size=point_size,
                            point_opacity=point_opacity,
                            camera_state=camera_state,
                            lod_type=lod_type,
                            density_percentage=density_percentage,
                            point_cloud_id=build_point_cloud_id(datapoint, f"correspondences_batch_{lvl}"),
                            title=f"Point Cloud Correspondences (Level {lvl})",
                        )
                    )

                # Create figures in parallel using centralized utility
                figure_creator = ParallelFigureCreator(max_workers=4, enable_timing=False)
                level_figures = figure_creator.create_figures_parallel(figure_tasks)
                all_figures.extend(level_figures)
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
