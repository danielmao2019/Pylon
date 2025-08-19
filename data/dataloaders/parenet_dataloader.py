from typing import Any, Optional, List
from functools import partial
from data.collators.parenet.parenet_collator_wrapper import parenet_collate_fn
from data.dataloaders.base_dataloader import BaseDataLoader


class PARENetDataloader(BaseDataLoader):
    """PARENet dataloader following Pylon patterns.
    
    This dataloader sets up the PARENet collate function with proper parameters
    and inherits from BaseDataLoader like other PCR models (GeoTransformer, OverlapPredator).
    
    The collation logic uses the parenet_collate_fn wrapper which handles:
    - Pylon format to PARENet format conversion
    - Multi-stage hierarchical point cloud processing
    - Neighbor computation with proper device handling
    
    Unlike GeoTransformer which calibrates neighbors dynamically, PARENet uses fixed
    neighbor counts specified in num_neighbors.
    """

    def __init__(
        self,
        dataset: Any,
        num_stages: int = 4,
        voxel_size: float = 0.05,
        subsample_ratio: float = 4.0,
        num_neighbors: Optional[List[int]] = None,
        precompute_data: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PARENet dataloader.
        
        Args:
            dataset: The dataset instance
            num_stages: Number of hierarchical stages for multi-scale processing
            voxel_size: Base voxel size for grid subsampling  
            subsample_ratio: Ratio between consecutive stages
            num_neighbors: List of neighbor counts for each stage (defaults to [32, 32, 32, 32])
            precompute_data: Whether to precompute multi-scale data and neighbors
            **kwargs: Additional arguments passed to BaseDataLoader
        """
        assert 'collate_fn' not in kwargs, 'collate_fn is not allowed to be set'
        
        # Initialize default neighbor counts if not provided
        if num_neighbors is None:
            num_neighbors = [32, 32, 32, 32][:num_stages]
        assert len(num_neighbors) == num_stages, \
            f"num_neighbors length ({len(num_neighbors)}) must match num_stages ({num_stages})"
        
        # Store neighbor counts (no calibration needed for PARENet)
        self.num_neighbors = num_neighbors
        
        # Initialize base dataloader with partial collate_fn using the wrapper
        super(PARENetDataloader, self).__init__(
            dataset=dataset,
            collate_fn=partial(
                parenet_collate_fn,
                num_stages=num_stages,
                voxel_size=voxel_size,
                num_neighbors=num_neighbors,
                subsample_ratio=subsample_ratio,
                precompute_data=precompute_data,
            ),
            **kwargs,
        )