import json
import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.pcr_translation import PCRTranslation


class LiDARCameraPosePCRDataset(SyntheticTransformPCRDataset):
    """LiDAR camera pose PCR dataset that samples sensor poses from transforms.json files.
    
    This dataset extends SyntheticTransformPCRDataset to use camera poses from
    transforms.json files when performing LiDAR simulation crops. Point cloud
    filepaths and corresponding transforms.json filepaths are provided directly.
    
    Key Features:
    - Accepts lists of point cloud and transforms.json filepaths directly
    - Samples sensor positions from the union of all camera poses
    - Uses camera extrinsics matrices for LiDAR simulation sensor poses
    - Maintains deterministic sampling with caching support
    """
    
    # Required BaseDataset attributes - inherit from parent but override if needed
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None  # Dynamic based on dataset_size parameter
    
    def __init__(
        self,
        pc_filepaths: List[str],
        transforms_json_filepaths: List[str],
        dataset_size: int,
        camera_count: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize LiDAR camera pose PCR dataset.
        
        Args:
            pc_filepaths: List of point cloud file paths
            transforms_json_filepaths: List of transforms.json file paths (must correspond to pc_filepaths)
            dataset_size: Total number of synthetic pairs to generate
            camera_count: Optional number of camera poses to randomly sample from the union. 
                         If None, use all available camera poses.
            **kwargs: Additional arguments passed to parent class
        """
        # Validate inputs
        assert isinstance(pc_filepaths, list), f"pc_filepaths must be list, got {type(pc_filepaths)}"
        assert isinstance(transforms_json_filepaths, list), f"transforms_json_filepaths must be list, got {type(transforms_json_filepaths)}"
        assert len(pc_filepaths) == len(transforms_json_filepaths), (
            f"Number of point clouds ({len(pc_filepaths)}) must match number of transforms.json files ({len(transforms_json_filepaths)})"
        )
        assert len(pc_filepaths) > 0, "Must provide at least one point cloud file"
        
        # Validate camera_count if provided
        if camera_count is not None:
            assert isinstance(camera_count, int), f"camera_count must be int, got {type(camera_count)}"
            assert camera_count > 0, f"camera_count must be positive, got {camera_count}"
        
        # Store file paths and camera_count
        self.pc_filepaths = pc_filepaths
        self.transforms_json_filepaths = transforms_json_filepaths
        self.camera_count = camera_count
        
        # Initialize list to store all camera poses from all transforms.json files
        self.all_camera_poses: List[np.ndarray] = []
        
        # Load all camera poses before calling parent constructor
        self._load_all_camera_poses()
        
        # Call parent constructor with dummy data_root (not used)
        # Use temp directory as data_root to satisfy parent class validation
        import tempfile
        temp_data_root = tempfile.mkdtemp()
        super().__init__(
            data_root=temp_data_root,  # Temporary directory to satisfy parent validation
            dataset_size=dataset_size,
            crop_method='lidar',  # Force LiDAR cropping
            **kwargs,
        )
        
        print(f"Loaded {len(self.all_camera_poses)} camera poses from {len(self.pc_filepaths)} point cloud files")

    # =========================================================================
    # Dataset-specific initialization methods (called during __init__)
    # =========================================================================
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations using provided filepaths."""
        # Create file pair annotations using provided filepaths
        self.file_pair_annotations = []
        
        for pc_filepath in self.pc_filepaths:
            # For single-temporal dataset, src and tgt are the same
            annotation = {
                'src_filepath': pc_filepath,
                'tgt_filepath': pc_filepath,  # Same file for self-registration
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Initialized {len(self.file_pair_annotations)} file pair annotations")
    
    def _load_all_camera_poses(self) -> None:
        """Load camera poses from all transforms.json files into union."""
        self.all_camera_poses = []
        
        # Load camera poses from each transforms.json file and add to union
        for json_path in self.transforms_json_filepaths:
            camera_poses = self._load_camera_poses_from_json(json_path)
            
            # Validate we have camera poses for this file (maintain 1-1 correspondence validation)
            assert len(camera_poses) > 0, (
                f"No camera poses found in transforms.json file {json_path}"
            )
            
            # Add all poses from this file to the union
            self.all_camera_poses.extend(camera_poses)
        
        # Validate we have camera poses in total
        assert len(self.all_camera_poses) > 0, (
            f"No camera poses found in any transforms.json files."
        )
        
        # Apply subsampling if camera_count is specified
        if self.camera_count is not None:
            total_poses = len(self.all_camera_poses)
            
            # Ensure we don't request more poses than available
            if self.camera_count > total_poses:
                print(f"Warning: Requested {self.camera_count} camera poses but only {total_poses} available. Using all poses.")
            else:
                # Randomly sample camera_count poses from the union
                # Use a fixed seed for reproducibility
                rng = np.random.RandomState(seed=42)
                selected_indices = rng.choice(total_poses, size=self.camera_count, replace=False)
                selected_indices = sorted(selected_indices)  # Sort for deterministic ordering
                
                # Keep only the selected poses
                self.all_camera_poses = [self.all_camera_poses[i] for i in selected_indices]
                
                print(f"Subsampled {self.camera_count} camera poses from {total_poses} total poses")
    
    # =========================================================================
    # Camera pose loading utilities
    # =========================================================================
    
    def _load_camera_poses_from_json(self, json_path: str) -> List[np.ndarray]:
        """Load camera poses from a transforms.json file (nerfstudio format only).
        
        Args:
            json_path: Path to transforms.json file
            
        Returns:
            List of 4x4 camera extrinsics matrices as numpy arrays
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Assert nerfstudio format
        assert isinstance(data, dict), f"transforms.json must be a dictionary, got {type(data)}"
        assert 'frames' in data, f"transforms.json must have 'frames' key, got keys: {list(data.keys())}"
        assert isinstance(data['frames'], list), f"'frames' must be a list, got {type(data['frames'])}"
        assert len(data['frames']) > 0, f"'frames' must not be empty"
        
        camera_poses = []
        
        for i, frame in enumerate(data['frames']):
            assert isinstance(frame, dict), f"Frame {i} must be a dictionary, got {type(frame)}"
            assert 'transform_matrix' in frame, f"Frame {i} must have 'transform_matrix' key, got keys: {list(frame.keys())}"
            
            transform_matrix = frame['transform_matrix']
            assert isinstance(transform_matrix, list), f"Frame {i} transform_matrix must be a list, got {type(transform_matrix)}"
            assert len(transform_matrix) == 4, f"Frame {i} transform_matrix must have 4 rows, got {len(transform_matrix)}"
            
            # Convert to numpy array and validate
            pose_matrix = np.array(transform_matrix, dtype=np.float32)
            assert pose_matrix.shape == (4, 4), f"Frame {i} invalid pose matrix shape: {pose_matrix.shape}"
            
            # Basic sanity check: last row should be [0, 0, 0, 1]
            expected_last_row = np.array([0, 0, 0, 1], dtype=np.float32)
            last_row_close = np.allclose(pose_matrix[3, :], expected_last_row, atol=1e-5)
            assert last_row_close, f"Frame {i} invalid last row: {pose_matrix[3, :]} (expected [0, 0, 0, 1])"
            
            camera_poses.append(pose_matrix)
        
        print(f"Loaded {len(camera_poses)} camera poses from {json_path}")
        return camera_poses

    # =========================================================================
    # Transform sampling methods (override parent behavior)
    # =========================================================================
    
    def _sample_transform(self, seed: int) -> Dict[str, Any]:
        """Sample transform parameters using camera poses for sensor positions.
        
        This method overrides the parent's _sample_transform to use camera poses
        from transforms.json files instead of randomly sampling sensor positions.
        Samples from the union of all camera poses across all transforms.json files.
        
        Args:
            seed: Random seed for deterministic sampling
            
        Returns:
            Dictionary containing all transform parameters including camera-based sensor pose
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Sample SE(3) transformation parameters for synthetic misalignment
        rotation_angles = torch.rand(3, generator=generator) * (2 * self.rotation_mag) - self.rotation_mag
        translation = torch.rand(3, generator=generator) * (2 * self.translation_mag) - self.translation_mag
        
        # Sample a camera pose from the union of all camera poses
        camera_pose_idx = int(torch.randint(0, len(self.all_camera_poses), (1,), generator=generator).item())
        camera_extrinsics = self.all_camera_poses[camera_pose_idx]
        
        # Extract position and rotation from the 4x4 extrinsics matrix
        sensor_position = camera_extrinsics[:3, 3]  # Translation component
        rotation_matrix = camera_extrinsics[:3, :3]  # Rotation component
        
        # Convert rotation matrix to Euler angles for storage
        # Using ZYX Euler angle convention
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        euler_angles = np.array([x, y, z])
        
        # Generate crop seed for deterministic cropping (derived from main seed)
        crop_seed = (seed * 31 + 42) % (2**32)  # Deterministic derivation from main seed
        
        # Build transform configuration with camera-based sensor pose
        config = {
            'rotation_angles': rotation_angles.tolist(),
            'translation': translation.tolist(),
            'crop_method': 'lidar',
            'sensor_position': sensor_position.tolist(),
            'sensor_euler_angles': euler_angles.tolist(),
            'lidar_max_range': self.lidar_max_range,
            'lidar_horizontal_fov': self.lidar_horizontal_fov,
            'lidar_vertical_fov': list(self.lidar_vertical_fov),
            'seed': seed,
            'crop_seed': crop_seed,
            'camera_pose_idx': camera_pose_idx,  # Store which camera pose was used (from union)
        }
        
        return config

    # =========================================================================
    # Data loading methods (override parent behavior)
    # =========================================================================
    
    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load point cloud data with PCRTranslation centering applied.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_data, tgt_pc_data) centered point cloud dictionaries with all attributes (pos, rgb, etc.)
        """
        # Load raw point clouds using parent method (now returns dictionaries)
        src_pc_data, tgt_pc_data = super()._load_file_pair_data(file_pair_annotation)
        
        # Create identity transform (PCRTranslation will adjust this appropriately)
        identity_transform = torch.eye(4, dtype=torch.float32, device=self.device)
        
        # Apply PCRTranslation to center both point clouds
        pcr_translation = PCRTranslation()
        centered_src_pc, centered_tgt_pc, _ = pcr_translation(
            src_pc=src_pc_data,
            tgt_pc=tgt_pc_data,
            transform=identity_transform
        )
        
        # Return centered point cloud dictionaries with all attributes preserved
        return centered_src_pc, centered_tgt_pc
