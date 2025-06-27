# Reference: https://teaser.readthedocs.io/en/latest/quickstart.html#usage-in-python-projects
from typing import Dict, Optional, Tuple
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class TeaserPlusPlus(torch.nn.Module):
    """TEASER++ robust point cloud registration algorithm.
    
    TEASER++ (Truncated least squares Estimation And SEmidefinite Relaxation) 
    is a certifiably optimal robust registration method that can handle 
    high outlier rates in correspondences.
    
    Reference:
        Yang et al., "TEASER: Fast and Certifiable Point Cloud Registration"
        https://teaser.readthedocs.io/
    
    Args:
        estimate_rotation: Whether to estimate rotation
        estimate_scaling: Whether to estimate scaling
        correspondences: Method to establish correspondences ('fpfh' or None)
        voxel_size: Voxel size for downsampling and FPFH feature extraction
    """
    
    def __init__(
        self,
        estimate_rotation: bool = True,
        estimate_scaling: bool = True,
        correspondences: Optional[str] = None,
        voxel_size: float = 0.05,
    ) -> None:
        """
        Initialize TeaserPlusPlus model.

        Args:
            estimate_rotation: Whether to estimate rotation
            estimate_scaling: Whether to estimate scaling
            correspondences: Method to establish correspondences. Options:
                - None: Use all points (default behavior)
                - 'fpfh': Use FPFH features to establish correspondences
            voxel_size: Voxel size for downsampling and FPFH feature extraction
        """
        super(TeaserPlusPlus, self).__init__()
        self.estimate_rotation = estimate_rotation
        self.estimate_scaling = estimate_scaling
        if correspondences is not None:
            assert correspondences in ['fpfh'], f"{correspondences=}"
        self.correspondences = correspondences
        assert isinstance(voxel_size, (float, int)) and voxel_size > 0, f"{voxel_size=}"
        self.voxel_size = voxel_size

    def _extract_fpfh(self, points: np.ndarray) -> np.ndarray:
        """Extract FPFH features from point cloud.
        
        Args:
            points: Point cloud as numpy array (N, 3)
            
        Returns:
            FPFH features as numpy array (N, 33)
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute FPFH features
        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return np.array(fpfh.data).T

    def _find_correspondences(self, feats0: np.ndarray, feats1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find correspondences between two sets of features using mutual nearest neighbor.
        
        Args:
            feats0: Features of first point cloud (N, 33)
            feats1: Features of second point cloud (M, 33)
            
        Returns:
            Tuple of (indices in first cloud, indices in second cloud)
        """
        # Find nearest neighbors from feats0 to feats1
        feat1tree = cKDTree(feats1)
        nns01 = feat1tree.query(feats0, k=1)[1]
        corres01_idx0 = np.arange(len(nns01))
        corres01_idx1 = nns01

        # Find nearest neighbors from feats1 to feats0
        feat0tree = cKDTree(feats0)
        nns10 = feat0tree.query(feats1, k=1)[1]
        corres10_idx1 = np.arange(len(nns10))
        corres10_idx0 = nns10

        # Apply mutual filter
        mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
        corres_idx0 = corres01_idx0[mutual_filter]
        corres_idx1 = corres01_idx1[mutual_filter]

        return corres_idx0, corres_idx1

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Perform TEASER++ registration on source and target point clouds.
        
        Args:
            inputs: Dictionary containing:
                - 'src_pc': Dict with 'pos' key containing source points (B, N, 3)
                - 'tgt_pc': Dict with 'pos' key containing target points (B, M, 3)
                
        Returns:
            Transformation matrix (B, 4, 4) that aligns source to target
        """
        import teaserpp_python
        
        # Input validation
        assert isinstance(inputs, dict), "inputs must be a dictionary"
        assert 'src_pc' in inputs, "inputs must contain 'src_pc' key"
        assert 'tgt_pc' in inputs, "inputs must contain 'tgt_pc' key"
        assert isinstance(inputs['src_pc'], dict), "inputs['src_pc'] must be a dictionary"
        assert isinstance(inputs['tgt_pc'], dict), "inputs['tgt_pc'] must be a dictionary"
        assert 'pos' in inputs['src_pc'], "inputs['src_pc'] must contain 'pos' key"
        assert 'pos' in inputs['tgt_pc'], "inputs['tgt_pc'] must contain 'pos' key"
        
        # Validate tensor properties
        assert isinstance(inputs['src_pc']['pos'], torch.Tensor), "Source positions must be a tensor"
        assert isinstance(inputs['tgt_pc']['pos'], torch.Tensor), "Target positions must be a tensor"
        assert inputs['src_pc']['pos'].dim() == 3, f"Source positions must be 3D tensor, got {inputs['src_pc']['pos'].dim()}D"
        assert inputs['tgt_pc']['pos'].dim() == 3, f"Target positions must be 3D tensor, got {inputs['tgt_pc']['pos'].dim()}D"
        assert inputs['src_pc']['pos'].shape[-1] == 3, f"Source positions must have 3 coordinates, got {inputs['src_pc']['pos'].shape[-1]}"
        assert inputs['tgt_pc']['pos'].shape[-1] == 3, f"Target positions must have 3 coordinates, got {inputs['tgt_pc']['pos'].shape[-1]}"
        assert inputs['src_pc']['pos'].shape[0] == inputs['tgt_pc']['pos'].shape[0], \
            f"Batch sizes must match: source={inputs['src_pc']['pos'].shape[0]}, target={inputs['tgt_pc']['pos'].shape[0]}"
        assert inputs['src_pc']['pos'].shape[1] > 0, "Source point cloud cannot be empty"
        assert inputs['tgt_pc']['pos'].shape[1] > 0, "Target point cloud cannot be empty"
        
        batch_size = inputs['src_pc']['pos'].shape[0]
        device = inputs['src_pc']['pos'].device

        # Convert to numpy for TEASER++
        source_np = inputs['src_pc']['pos'].detach().cpu().numpy()
        target_np = inputs['tgt_pc']['pos'].detach().cpu().numpy()

        # Process each batch
        transformations = []
        for i in range(batch_size):
            src_points = source_np[i]
            tgt_points = target_np[i]

            # If using FPFH correspondences, extract features and find correspondences
            if self.correspondences == 'fpfh':
                # Extract FPFH features
                src_feats = self._extract_fpfh(src_points)
                tgt_feats = self._extract_fpfh(tgt_points)

                # Find correspondences
                src_idx, tgt_idx = self._find_correspondences(src_feats, tgt_feats)

                # Extract corresponding points
                src_points = src_points[src_idx]
                tgt_points = tgt_points[tgt_idx]

                print(f'FPFH generates {len(src_idx)} putative correspondences.')

            # Set up TEASER++ parameters
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = 0.01
            if self.estimate_rotation:
                solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
                solver_params.rotation_gnc_factor = 1.4
                solver_params.rotation_max_iterations = 100
                solver_params.rotation_cost_threshold = 1e-12
            solver_params.estimate_scaling = self.estimate_scaling

            # Create solver and solve
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(src_points.T.astype(np.float64), tgt_points.T.astype(np.float64))

            # Get solution
            solution = solver.getSolution()
            rot = solution.rotation if self.estimate_rotation else np.eye(3)
            trans = solution.translation
            solution = np.eye(4)
            solution[:3, :3] = rot
            solution[:3, 3] = trans
            transformations.append(solution)

        # Convert back to tensor
        return torch.tensor(np.stack(transformations), dtype=torch.float32, device=device)
