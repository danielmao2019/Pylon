from typing import Any, Dict, Tuple
import os
import pickle
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
from utils.point_cloud_ops.correspondences import get_correspondences
from utils.io.point_clouds.load_point_cloud import load_point_cloud


class _ThreeDMatchBaseDataset(BasePCRDataset):
    """Base dataset for 3DMatch family of datasets for point cloud registration.

    This dataset contains RGB-D scans of real-world indoor scenes from the 3DMatch benchmark.
    It is commonly used for evaluating point cloud registration algorithms.

    Paper:
        3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions
        https://arxiv.org/abs/1603.08182
    """

    SPLIT_OPTIONS = ['train', 'val', 'test']
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self,
        matching_radius: float = 0.1,
        overlap_min: float = 0.0,
        overlap_max: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the base dataset.

        Args:
            matching_radius: Radius for finding correspondences (default: 0.1)
            overlap_min: Minimum overlap ratio between point cloud pairs (default: 0.0)
            overlap_max: Maximum overlap ratio between point cloud pairs (default: 1.0)
            **kwargs: Additional arguments passed to BaseDataset
        """
        self.matching_radius = matching_radius
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max

        # Initialize base class
        super(_ThreeDMatchBaseDataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        """Initialize dataset annotations from metadata files."""
        # Metadata paths
        metadata_dir = os.path.join(self.data_root, 'metadata')

        # Load metadata based on split and dataset type
        if self.split in ['train', 'val']:
            metadata_file = os.path.join(metadata_dir, f'{self.split}.pkl')
        else:  # test split
            # Use appropriate test file based on dataset type
            if isinstance(self, ThreeDLoMatchDataset):
                metadata_file = os.path.join(metadata_dir, '3DLoMatch.pkl')
            else:
                metadata_file = os.path.join(metadata_dir, '3DMatch.pkl')

        # Assert metadata file exists
        assert os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}"

        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata_list = pickle.load(f)

        # GeoTransformer format: list of dicts
        assert isinstance(metadata_list, list), f"Expected list format, got {type(metadata_list)}"
        assert len(metadata_list) > 0, "Metadata list is empty"

        # Check required keys in first item
        expected_keys = ['pcd0', 'pcd1', 'rotation', 'translation', 'overlap']
        first_item = metadata_list[0]
        assert all(key in first_item for key in expected_keys), f"Missing keys in metadata: {list(first_item.keys())}"

        # Filter by overlap threshold and convert to annotations format
        self.annotations = []
        for item in metadata_list:
            overlap = item['overlap']
            if self.overlap_min < overlap <= self.overlap_max:
                # Extract scene names and ensure they match
                src_scene = item['pcd0'].split('/')[0]
                tgt_scene = item['pcd1'].split('/')[0]
                assert src_scene == tgt_scene, f"Scene names must match: src={src_scene}, tgt={tgt_scene}"

                annotation = {
                    'src_path': os.path.join(self.data_root, item['pcd0']),
                    'tgt_path': os.path.join(self.data_root, item['pcd1']),
                    'rotation': item['rotation'],  # (3, 3) numpy array
                    'translation': item['translation'],  # (3,) numpy array
                    'overlap': overlap,
                    'scene_name': item.get('scene_name', src_scene),  # Use provided or extracted scene name
                    'frag_id0': item.get('frag_id0', int(item['pcd0'].split('/')[-1].split('_')[-1].split('.')[0])),  # Use provided or extract fragment ID
                    'frag_id1': item.get('frag_id1', int(item['pcd1'].split('/')[-1].split('_')[-1].split('.')[0])),  # Use provided or extract fragment ID
                }
                self.annotations.append(annotation)

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        version_dict.update({
            'matching_radius': self.matching_radius,
            'overlap_min': self.overlap_min,
            'overlap_max': self.overlap_max,
        })
        return version_dict

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a single datapoint from the dataset.

        Args:
            idx: Index of the datapoint to load

        Returns:
            Tuple of (inputs, labels, meta_info) dictionaries
        """
        # Get annotation
        annotation = self.annotations[idx]

        # Load point clouds (returns Dict[str, torch.Tensor])
        src_pc_dict = load_point_cloud(annotation['src_path'], device=self.device)
        tgt_pc_dict = load_point_cloud(annotation['tgt_path'], device=self.device)

        # Create point cloud dictionaries
        src_pc = {
            'pos': src_pc_dict['pos'],
            'feat': torch.ones((src_pc_dict['pos'].shape[0], 1), dtype=torch.float32, device=self.device)
        }
        tgt_pc = {
            'pos': tgt_pc_dict['pos'],
            'feat': torch.ones((tgt_pc_dict['pos'].shape[0], 1), dtype=torch.float32, device=self.device)
        }

        # Create transformation matrix
        # NOTE: The metadata contains target→source transform, but we need source→target
        # So we create the matrix and then invert it
        rotation = torch.tensor(annotation['rotation'], dtype=torch.float32, device=self.device)
        translation = torch.tensor(annotation['translation'], dtype=torch.float32, device=self.device)
        transform_tgt_to_src = torch.eye(4, dtype=torch.float32, device=self.device)
        transform_tgt_to_src[:3, :3] = rotation
        transform_tgt_to_src[:3, 3] = translation
        
        # Invert to get source→target transformation
        transform = torch.inverse(transform_tgt_to_src)

        # Get or compute correspondences with caching
        correspondences = self._get_cached_correspondences(annotation, src_pc['pos'], tgt_pc['pos'], transform)

        # Prepare inputs
        inputs = {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': correspondences,
        }

        # Prepare labels
        labels = {
            'transform': transform,
        }

        # Prepare meta_info (BaseDataset automatically adds 'idx')
        meta_info = {
            'src_path': annotation['src_path'],
            'tgt_path': annotation['tgt_path'],
            'scene_name': annotation.get('scene_name', 'unknown'),
            'overlap': annotation.get('overlap', 0.0),
            'src_frame': annotation.get('frag_id0', 0),
            'tgt_frame': annotation.get('frag_id1', 0),
        }

        return inputs, labels, meta_info

    def _get_cached_correspondences(
        self,
        annotation: Dict[str, Any],
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """Get correspondences with caching mechanism.

        Args:
            annotation: Annotation dictionary containing paths and metadata
            src_points: Source point cloud positions [M, 3]
            tgt_points: Target point cloud positions [N, 3]
            transform: Transformation matrix [4, 4]

        Returns:
            Correspondences tensor [K, 2]
        """
        # Create cache directory (sibling to data_root)
        cache_dir = os.path.join(os.path.dirname(self.data_root), f'{os.path.basename(self.data_root)}_correspondences_cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Create simple cache key from file basenames and radius
        src_name = os.path.basename(annotation['src_path']).split('.')[0]
        tgt_name = os.path.basename(annotation['tgt_path']).split('.')[0]
        cache_key = f"{src_name}_{tgt_name}_{self.matching_radius}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    correspondences = pickle.load(f)
                return torch.tensor(correspondences, dtype=torch.int64, device=self.device)
            except:
                # Cache file corrupted, recompute
                pass

        # Compute correspondences
        correspondences = get_correspondences(
            src_points=src_points,
            tgt_points=tgt_points,
            transform=transform,
            radius=self.matching_radius,
        )

        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(correspondences.cpu().numpy(), f)
        except:
            # Cache write failed, but continue
            pass

        return correspondences


class ThreeDMatchDataset(_ThreeDMatchBaseDataset):
    """3DMatch dataset for point cloud registration.

    Official 3DMatch dataset considers only scan pairs with >30% overlap.

    Paper:
        3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions
        https://arxiv.org/abs/1603.08182
    """

    DATASET_SIZE = {
        'train': 14313,  # 3DMatch train (overlap > 0.3)
        'val': 915,      # 3DMatch val (overlap > 0.3)
        'test': 1520,    # 3DMatch test (uses 3DMatch.pkl, filtered for overlap > 0.3)
    }

    def __init__(self, **kwargs) -> None:
        """Initialize 3DMatch dataset with overlap > 0.3."""
        super(ThreeDMatchDataset, self).__init__(
            overlap_min=0.3,
            overlap_max=1.0,
            **kwargs
        )


class ThreeDLoMatchDataset(_ThreeDMatchBaseDataset):
    """3DLoMatch dataset for point cloud registration.

    3DLoMatch considers only scan pairs with overlaps between 10% and 30%.
    This is a more challenging variant of 3DMatch for low-overlap scenarios.

    Paper:
        3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions
        https://arxiv.org/abs/1603.08182
    """

    DATASET_SIZE = {
        'train': 6225,   # 3DLoMatch train (0.1 < overlap <= 0.3)
        'val': 414,      # 3DLoMatch val (0.1 < overlap <= 0.3)
        'test': 1772,    # 3DLoMatch test (uses 3DLoMatch.pkl, filtered for 0.1 < overlap <= 0.3)
    }

    def __init__(self, **kwargs) -> None:
        """Initialize 3DLoMatch dataset with overlap between 0.1 and 0.3."""
        super(ThreeDLoMatchDataset, self).__init__(
            overlap_min=0.1,
            overlap_max=0.3,
            **kwargs
        )
