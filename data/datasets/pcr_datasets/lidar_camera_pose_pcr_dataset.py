import os
import json
import glob
import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.pcr_translation import PCRTranslation


class LiDARCameraPosePCRDataset(SyntheticTransformPCRDataset):
    """LiDAR camera pose PCR dataset that samples sensor poses from transforms.json files.
    
    This dataset extends SyntheticTransformPCRDataset to use camera poses from
    transforms.json files when performing LiDAR simulation crops. Each point cloud
    file is expected to have a corresponding transforms.json file containing camera
    poses, and the dataset samples from the union of all camera poses across all files.
    
    Key Features:
    - Loads camera poses from transforms.json files alongside point cloud files
    - Samples sensor positions from the union of all camera poses
    - Uses camera extrinsics matrices for LiDAR simulation sensor poses
    - Maintains deterministic sampling with caching support
    """
    
    # Required BaseDataset attributes - inherit from parent but override if needed
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None  # Dynamic based on dataset_size parameter
    
    def __init__(
        self,
        data_root: str,
        dataset_size: int,
        rotation_mag: float = 45.0,
        translation_mag: float = 0.5,
        matching_radius: float = 0.05,
        overlap_range: Tuple[float, float] = (0.3, 1.0),
        min_points: int = 512,
        max_trials: int = 1000,
        cache_filepath: Optional[str] = None,
        transforms_json_pattern: str = "transforms.json",
        **kwargs,
    ) -> None:
        """Initialize LiDAR camera pose PCR dataset.
        
        Args:
            data_root: Path to dataset root directory
            dataset_size: Total number of synthetic pairs to generate
            rotation_mag: Maximum rotation magnitude in degrees for synthetic transforms
            translation_mag: Maximum translation magnitude for synthetic transforms
            matching_radius: Radius for correspondence finding
            overlap_range: Overlap range (overlap_min, overlap_max] for filtering
            min_points: Minimum number of points filter for cache generation
            max_trials: Maximum number of trials to generate valid transforms
            cache_filepath: Path to cache file (if None, no caching is used)
            transforms_json_pattern: Pattern for finding transforms.json files relative to point clouds
            **kwargs: Additional arguments passed to parent class
        """
        # Store transforms.json pattern before calling parent init
        self.transforms_json_pattern = transforms_json_pattern
        
        # Initialize list to store all camera poses from all transforms.json files
        self.all_camera_poses: List[np.ndarray] = []
        self.camera_pose_to_file_idx: Dict[int, int] = {}  # Maps camera pose index to file index
        
        # Call parent constructor which will call _init_annotations
        super().__init__(
            data_root=data_root,
            dataset_size=dataset_size,
            rotation_mag=rotation_mag,
            translation_mag=translation_mag,
            matching_radius=matching_radius,
            overlap_range=overlap_range,
            min_points=min_points,
            max_trials=max_trials,
            cache_filepath=cache_filepath,
            crop_method='lidar',  # Force LiDAR cropping
            **kwargs,
        )
        
        print(f"Loaded {len(self.all_camera_poses)} camera poses from {len(self.file_pair_annotations)} point cloud files")

    def _init_annotations(self) -> None:
        """Initialize file pair annotations and load camera poses from transforms.json files."""
        # Find all point cloud files
        all_files = []
        for pattern in ['*.ply', '*.las', '*.laz', '*.txt', '*.pth', '*.off']:
            all_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        pc_files = sorted(all_files)
        
        # Create file pair annotations and load camera poses
        self.file_pair_annotations = []
        self.all_camera_poses = []
        self.camera_pose_to_file_idx = {}
        
        for file_idx, pc_filepath in enumerate(pc_files):
            # For single-temporal dataset, src and tgt are the same
            annotation = {
                'src_filepath': pc_filepath,
                'tgt_filepath': pc_filepath,
            }
            
            # Look for corresponding transforms.json file
            # Try different locations relative to the point cloud file
            base_dir = os.path.dirname(pc_filepath)
            base_name = os.path.splitext(os.path.basename(pc_filepath))[0]
            
            # Possible locations for transforms.json
            possible_json_paths = [
                os.path.join(base_dir, self.transforms_json_pattern),  # Same directory
                os.path.join(base_dir, base_name, self.transforms_json_pattern),  # Subdirectory with same name
                os.path.join(base_dir, '..', self.transforms_json_pattern),  # Parent directory
                pc_filepath.replace(os.path.splitext(pc_filepath)[1], '_transforms.json'),  # Replace extension
            ]
            
            # Find the first existing transforms.json file
            transforms_json_path = None
            for json_path in possible_json_paths:
                if os.path.exists(json_path):
                    transforms_json_path = json_path
                    break
            
            if transforms_json_path is not None:
                # Load camera poses from transforms.json
                camera_poses = self._load_camera_poses_from_json(transforms_json_path)
                annotation['transforms_json_path'] = transforms_json_path
                annotation['num_camera_poses'] = len(camera_poses)
                
                # Add to global list of camera poses
                for pose in camera_poses:
                    self.all_camera_poses.append(pose)
                    self.camera_pose_to_file_idx[len(self.all_camera_poses) - 1] = file_idx
            else:
                print(f"Warning: No transforms.json found for {pc_filepath}")
                annotation['transforms_json_path'] = None
                annotation['num_camera_poses'] = 0
            
            self.file_pair_annotations.append(annotation)
        
        # Validate we have camera poses
        assert len(self.all_camera_poses) > 0, (
            f"No camera poses found in any transforms.json files. "
            f"Searched for '{self.transforms_json_pattern}' files relative to point clouds."
        )
        
        print(f"Found {len(self.file_pair_annotations)} point cloud files")
        print(f"Total camera poses collected: {len(self.all_camera_poses)}")

    def _load_camera_poses_from_json(self, json_path: str) -> List[np.ndarray]:
        """Load camera poses from a transforms.json file.
        
        Args:
            json_path: Path to transforms.json file
            
        Returns:
            List of 4x4 camera extrinsics matrices as numpy arrays
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        camera_poses = []
        
        # Handle different formats of transforms.json
        if 'frames' in data:
            # Standard NeRF/Nerfstudio format
            for frame in data['frames']:
                if 'transform_matrix' in frame:
                    # Convert list of lists to numpy array
                    pose_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
                    assert pose_matrix.shape == (4, 4), f"Invalid pose matrix shape: {pose_matrix.shape}"
                    camera_poses.append(pose_matrix)
        elif 'transforms' in data:
            # Alternative format
            for transform in data['transforms']:
                if 'matrix' in transform:
                    pose_matrix = np.array(transform['matrix'], dtype=np.float32)
                    assert pose_matrix.shape == (4, 4), f"Invalid pose matrix shape: {pose_matrix.shape}"
                    camera_poses.append(pose_matrix)
        else:
            # Try to directly parse if it's a list of transforms
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, np.ndarray)) and len(item) == 4:
                        pose_matrix = np.array(item, dtype=np.float32)
                        if pose_matrix.shape == (4, 4):
                            camera_poses.append(pose_matrix)
        
        return camera_poses

    def _sample_transform(self, seed: int) -> Dict[str, Any]:
        """Sample transform parameters using camera poses for sensor positions.
        
        This method overrides the parent's _sample_transform to use camera poses
        from transforms.json files instead of randomly sampling sensor positions.
        
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
        camera_idx = int(torch.randint(0, len(self.all_camera_poses), (1,), generator=generator).item())
        camera_extrinsics = self.all_camera_poses[camera_idx]
        
        # Extract position and rotation from the 4x4 extrinsics matrix
        sensor_position = camera_extrinsics[:3, 3]  # Translation component
        rotation_matrix = camera_extrinsics[:3, :3]  # Rotation component
        
        # Convert rotation matrix to Euler angles for storage
        # Using ZYX Euler angle convention
        # Extract Euler angles from rotation matrix
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
            'lidar_apply_range_filter': self.lidar_apply_range_filter,
            'lidar_apply_fov_filter': self.lidar_apply_fov_filter,
            'lidar_apply_occlusion_filter': self.lidar_apply_occlusion_filter,
            'seed': seed,
            'camera_pose_idx': camera_idx,  # Store which camera pose was used
            'camera_pose_file_idx': self.camera_pose_to_file_idx.get(camera_idx, -1),
        }
        
        return config

    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud data with PCRTranslation centering applied.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_raw, tgt_pc_raw) centered point cloud position tensors
        """
        # Load raw point clouds using parent method
        src_pc_raw, tgt_pc_raw = super()._load_file_pair_data(file_pair_annotation)
        
        # Create point cloud dictionaries for PCRTranslation
        src_pc_dict = {'pos': src_pc_raw}
        tgt_pc_dict = {'pos': tgt_pc_raw}
        
        # Create identity transform (PCRTranslation will adjust this appropriately)
        identity_transform = torch.eye(4, dtype=torch.float32, device=self.device)
        
        # Apply PCRTranslation to center both point clouds
        pcr_translation = PCRTranslation()
        centered_src_pc, centered_tgt_pc, _ = pcr_translation(
            src_pc=src_pc_dict,
            tgt_pc=tgt_pc_dict,
            transform=identity_transform
        )
        
        # Return centered position tensors
        return centered_src_pc['pos'], centered_tgt_pc['pos']
