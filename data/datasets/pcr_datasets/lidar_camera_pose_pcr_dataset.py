import json
import os
import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.pcr_translation import PCRTranslation
from data.transforms.vision_3d import LiDARSimulationCrop


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
        lidar_max_range: float = 6.0,
        lidar_fov: tuple = (120.0, 60.0),
        lidar_fov_crop_mode: str = "frustum",
        lidar_apply_range_filter: bool = False,
        lidar_apply_fov_filter: bool = True,
        lidar_apply_occlusion_filter: bool = False,
        **kwargs,
    ) -> None:
        """Initialize LiDAR camera pose PCR dataset.
        
        Args:
            pc_filepaths: List of point cloud file paths
            transforms_json_filepaths: List of transforms.json file paths (must correspond to pc_filepaths)
            dataset_size: Total number of synthetic pairs to generate
            camera_count: Optional number of camera poses to randomly sample from the union. 
                         If None, use all available camera poses.
            lidar_max_range: Maximum LiDAR sensor range in meters
            lidar_fov: Tuple of (horizontal_fov, vertical_fov) in degrees
            lidar_fov_crop_mode: Cropping mode - "ellipsoid" or "frustum"
            lidar_apply_range_filter: Whether to apply range-based filtering
            lidar_apply_fov_filter: Whether to apply field-of-view filtering
            lidar_apply_occlusion_filter: Whether to apply occlusion simulation
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
        
        # Store LiDAR parameters
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_fov = lidar_fov
        self.lidar_fov_crop_mode = lidar_fov_crop_mode
        self.lidar_apply_range_filter = lidar_apply_range_filter
        self.lidar_apply_fov_filter = lidar_apply_fov_filter
        self.lidar_apply_occlusion_filter = lidar_apply_occlusion_filter
        
        # Initialize per-scene camera pose storage
        self.scene_camera_poses: Dict[str, List[np.ndarray]] = {}  # camera poses organized by scene filepath
        self.scene_centering_translations: Dict[str, torch.Tensor] = {}  # centering translation per scene
        self.scene_poses_transformed: Dict[str, bool] = {}  # track transformation status per scene
        
        # Load all camera poses before calling parent constructor
        self._load_all_camera_poses()
        
        # Call parent constructor with dummy data_root (not used)
        # Use temp directory as data_root to satisfy parent class validation
        import tempfile
        temp_data_root = tempfile.mkdtemp()
        super().__init__(
            data_root=temp_data_root,  # Temporary directory to satisfy parent validation
            dataset_size=dataset_size,
            **kwargs,
        )
        
        total_poses = sum(len(poses) for poses in self.scene_camera_poses.values())
        print(f"Loaded {total_poses} camera poses from {len(self.pc_filepaths)} point cloud files")

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
        """Load camera poses from all transforms.json files, organized by scene."""
        self.scene_camera_poses = {}
        
        # Load camera poses from each transforms.json file, keeping them organized by scene
        for i, json_path in enumerate(self.transforms_json_filepaths):
            pc_filepath = self.pc_filepaths[i]  # Corresponding point cloud file
            
            camera_poses = self._load_camera_poses_from_json(json_path)
            
            # Validate we have camera poses for this file
            assert len(camera_poses) > 0, (
                f"No camera poses found in transforms.json file {json_path}"
            )
            
            # Store poses organized by point cloud filepath (this is the key for scene identification)
            self.scene_camera_poses[pc_filepath] = camera_poses
            self.scene_poses_transformed[pc_filepath] = False  # Not yet transformed
        
        # Apply per-scene subsampling if camera_count is specified
        if self.camera_count is not None:
            total_before = sum(len(poses) for poses in self.scene_camera_poses.values())
            
            # Subsample camera_count poses from each scene independently using scene-specific seeds
            for pc_filepath in list(self.scene_camera_poses.keys()):
                scene_poses = self.scene_camera_poses[pc_filepath]
                num_poses = len(scene_poses)
                
                if self.camera_count < num_poses:
                    # Create scene-specific random generator using pc_filepath as seed
                    scene_seed = hash(pc_filepath) % (2**32)  # Ensure positive 32-bit integer
                    scene_rng = np.random.RandomState(seed=scene_seed)
                    
                    # Randomly sample camera_count poses from this scene
                    selected_indices = scene_rng.choice(num_poses, size=self.camera_count, replace=False)
                    selected_indices = sorted(selected_indices)
                    self.scene_camera_poses[pc_filepath] = [scene_poses[i] for i in selected_indices]
                    print(f"Scene {os.path.basename(pc_filepath)}: {num_poses} → {self.camera_count} poses (seed: {scene_seed})")
                else:
                    print(f"Scene {os.path.basename(pc_filepath)}: keeping all {num_poses} poses (< {self.camera_count})")
            
            total_after = sum(len(poses) for poses in self.scene_camera_poses.values())
            print(f"Total camera poses: {total_before} → {total_after} after per-scene subsampling")
    
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
    
    def _get_cache_param_key(self) -> Tuple:
        """Generate cache parameter key including LiDAR and camera parameters.
        
        Extends parent class to include camera_count and LiDAR parameters
        in cache key, ensuring cache invalidation when these change.
        
        Returns:
            Cache parameter key tuple
        """
        parent_key = super()._get_cache_param_key()
        return parent_key + (
            self.camera_count,
            self.lidar_max_range,
            self.lidar_fov,
            self.lidar_fov_crop_mode,
            self.lidar_apply_range_filter,
            self.lidar_apply_fov_filter,
            self.lidar_apply_occlusion_filter
        )
    
    def _sample_crop(self, seed: int, file_idx: int) -> Dict[str, Any]:
        """Sample LiDAR crop parameters using scene-specific camera poses.
        
        Args:
            seed: Random seed for deterministic sampling
            file_idx: Index of file pair for scene-specific camera pose sampling
        
        Returns:
            Dictionary containing LiDAR crop parameters
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Get scene-specific camera poses using file_idx
        file_pair_annotation = self.file_pair_annotations[file_idx]
        pc_filepath = file_pair_annotation['src_filepath']
        scene_camera_poses = self.scene_camera_poses[pc_filepath]
        
        assert len(scene_camera_poses) > 0, f"No camera poses found for scene {pc_filepath}"
        
        # Sample a camera pose from scene-specific poses
        camera_pose_idx = int(torch.randint(0, len(scene_camera_poses), (1,), generator=generator).item())
        camera_extrinsics = scene_camera_poses[camera_pose_idx]
        
        # Extract position and rotation from the 4x4 extrinsics matrix
        sensor_position = camera_extrinsics[:3, 3]
        rotation_matrix = camera_extrinsics[:3, :3]
        
        # Convert rotation matrix to Euler angles
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
        
        # Generate crop seed for deterministic cropping
        crop_seed = (seed * 31 + 42) % (2**32)
        
        return {
            'sensor_position': sensor_position.tolist(),
            'sensor_euler_angles': euler_angles.tolist(),
            'lidar_max_range': self.lidar_max_range,
            'lidar_fov': self.lidar_fov,
            'crop_seed': crop_seed,
            'camera_pose_idx': camera_pose_idx,
            'scene_filepath': pc_filepath,
        }
    
    def _build_crop(self, crop_params: dict) -> LiDARSimulationCrop:
        """Build LiDAR crop transform from parameters.
        
        Args:
            crop_params: Crop configuration dictionary
            
        Returns:
            LiDAR crop transform object with sensor extrinsics
        """
        # Build 4x4 extrinsics matrix from sampled pose parameters
        sensor_position = torch.tensor(crop_params['sensor_position'], dtype=torch.float32, device=self.device)
        euler_angles = torch.tensor(crop_params['sensor_euler_angles'], dtype=torch.float32, device=self.device)
        
        # Convert Euler angles to rotation matrix
        cos_vals = torch.cos(euler_angles)
        sin_vals = torch.sin(euler_angles)
        
        # ZYX Euler angle convention (yaw, pitch, roll)
        R_z = torch.tensor([
            [cos_vals[2], -sin_vals[2], 0],
            [sin_vals[2], cos_vals[2], 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        R_y = torch.tensor([
            [cos_vals[1], 0, sin_vals[1]],
            [0, 1, 0],
            [-sin_vals[1], 0, cos_vals[1]]
        ], dtype=torch.float32, device=self.device)
        
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_vals[0], -sin_vals[0]],
            [0, sin_vals[0], cos_vals[0]]
        ], dtype=torch.float32, device=self.device)
        
        rotation_matrix = R_z @ R_y @ R_x
        
        # Build 4x4 extrinsics matrix
        sensor_extrinsics = torch.eye(4, dtype=torch.float32, device=self.device)
        sensor_extrinsics[:3, :3] = rotation_matrix
        sensor_extrinsics[:3, 3] = sensor_position
        
        # Create LiDAR crop transform with configurable filter settings
        crop_transform = LiDARSimulationCrop(
            max_range=crop_params['lidar_max_range'],
            fov=crop_params['lidar_fov'],
            ray_density_factor=0.8,  # Default value
            apply_range_filter=self.lidar_apply_range_filter,
            apply_fov_filter=self.lidar_apply_fov_filter,
            apply_occlusion_filter=self.lidar_apply_occlusion_filter,
            fov_crop_mode=self.lidar_fov_crop_mode
        )
        
        # Store sensor extrinsics for LiDAR cropping
        crop_transform._sensor_extrinsics = sensor_extrinsics
        
        return crop_transform
    
    def _apply_crop(self, crop_transform: LiDARSimulationCrop, pc_data: dict, crop_params: dict) -> dict:
        """Apply LiDAR crop transform with sensor extrinsics.
        
        Override parent to handle LiDAR-specific signature.
        
        Args:
            crop_transform: LiDAR crop transform object
            pc_data: Point cloud dictionary
            crop_params: Crop configuration parameters
            
        Returns:
            Cropped point cloud dictionary
        """
        # Use crop_seed for deterministic cropping
        crop_seed = crop_params['crop_seed']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(crop_seed)
        
        # LiDAR cropping requires sensor extrinsics parameter
        sensor_extrinsics = crop_transform._sensor_extrinsics
        return crop_transform._call_single(pc_data, sensor_extrinsics, generator=generator)

    # =========================================================================
    # Data loading methods (override parent behavior)
    # =========================================================================
    
    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load point cloud data with PCRTranslation centering applied.
        
        Camera poses for the specific scene are automatically transformed to match
        that scene's centered coordinate frame.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_data, tgt_pc_data) centered point cloud dictionaries with all attributes (pos, rgb, etc.)
        """
        pc_filepath = file_pair_annotation['src_filepath']  # Scene identifier
        
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
        
        # Transform camera poses for this specific scene if not already done
        if not self.scene_poses_transformed.get(pc_filepath, False):
            # Extract centering translation from PCRTranslation computation
            # Following the same logic as PCRTranslation.__call__
            src_pos = src_pc_data['pos']
            tgt_pos = tgt_pc_data['pos']
            union_points = torch.cat([src_pos, tgt_pos], dim=0)
            centering_translation = union_points.mean(dim=0)
            
            # Transform camera poses for this specific scene
            self._transform_scene_camera_poses(pc_filepath, centering_translation)
        
        # Return centered point cloud dictionaries with all attributes preserved
        return centered_src_pc, centered_tgt_pc

    def _transform_scene_camera_poses(self, pc_filepath: str, centering_translation: torch.Tensor) -> None:
        """Transform camera poses for a specific scene to match its centered coordinate frame.
        
        When PCRTranslation centers a point cloud, the camera poses for that specific scene
        must be transformed using that scene's specific centering translation.
        
        Args:
            pc_filepath: Point cloud file path (scene identifier)
            centering_translation: Translation vector used to center this scene's point cloud
        """
        if self.scene_poses_transformed.get(pc_filepath, False):
            return  # Already transformed for this scene
            
        assert isinstance(centering_translation, torch.Tensor), f"centering_translation must be torch.Tensor, got {type(centering_translation)}"
        assert centering_translation.shape == (3,), f"centering_translation must be shape (3,), got {centering_translation.shape}"
        assert pc_filepath in self.scene_camera_poses, f"Scene {pc_filepath} not found in scene_camera_poses"
        
        # Convert centering translation to numpy for consistency with camera poses
        centering_translation_np = centering_translation.detach().cpu().numpy()
        
        # Transform camera poses for this specific scene
        scene_poses = self.scene_camera_poses[pc_filepath]
        transformed_poses = []
        for pose in scene_poses:
            transformed_pose = pose.copy()
            # Subtract centering translation from camera position
            transformed_pose[:3, 3] -= centering_translation_np
            transformed_poses.append(transformed_pose)
        
        # Store the transformed poses for this scene
        self.scene_camera_poses[pc_filepath] = transformed_poses
        self.scene_poses_transformed[pc_filepath] = True
        
        print(f"Transformed {len(transformed_poses)} camera poses for scene {os.path.basename(pc_filepath)}")
        print(f"Applied centering translation: {centering_translation_np}")
