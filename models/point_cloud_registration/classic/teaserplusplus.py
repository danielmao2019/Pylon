# Reference: https://teaser.readthedocs.io/en/latest/quickstart.html#usage-in-python-projects
from typing import Dict, Optional, Tuple
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class TeaserPlusPlus(torch.nn.Module):

    def __init__(self, voxel_size: float = 0.05, correspondences: Optional[str] = None):
        """
        Initialize TeaserPlusPlus model.
        
        Args:
            voxel_size: Voxel size for downsampling and FPFH feature extraction
            correspondences: Method to establish correspondences. Options:
                - None: Use all points (default behavior)
                - 'fpfh': Use FPFH features to establish correspondences
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.correspondences = correspondences

    def _extract_fpfh(self, points: np.ndarray) -> np.ndarray:
        """
        Extract FPFH features from point cloud.
        
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

    def _find_correspondences(self, feats0: np.ndarray, feats1: np.ndarray, mutual_filter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find correspondences between two sets of features using nearest neighbor search.
        
        Args:
            feats0: Features of first point cloud (N, 33)
            feats1: Features of second point cloud (M, 33)
            mutual_filter: Whether to apply mutual filter
            
        Returns:
            Tuple of (indices in first cloud, indices in second cloud)
        """
        # Find nearest neighbors from feats0 to feats1
        feat1tree = cKDTree(feats1)
        nns01 = feat1tree.query(feats0, k=1, n_jobs=-1)[1]
        corres01_idx0 = np.arange(len(nns01))
        corres01_idx1 = nns01
        
        if not mutual_filter:
            return corres01_idx0, corres01_idx1
        
        # Find nearest neighbors from feats1 to feats0
        feat0tree = cKDTree(feats0)
        nns10 = feat0tree.query(feats1, k=1, n_jobs=-1)[1]
        corres10_idx1 = np.arange(len(nns10))
        corres10_idx0 = nns10
        
        # Apply mutual filter
        mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
        corres_idx0 = corres01_idx0[mutual_filter]
        corres_idx1 = corres01_idx1[mutual_filter]
        
        return corres_idx0, corres_idx1

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        TEASER++ registration.

        Args:
            inputs: Dictionary containing source and target point clouds
            inputs['src_pc']['pos']: Source point cloud (B, N, 3)
            inputs['tgt_pc']['pos']: Target point cloud (B, M, 3)

        Returns:
            Transformation matrix (B, 4, 4)
        """
        import teaserpp_python
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
            solver_params.estimate_scaling = False
            solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12

            # Create solver and solve
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(src_points.T.astype(np.float64), tgt_points.T.astype(np.float64))

            # Get solution
            solution = solver.getSolution()
            rot = solution.rotation
            trans = solution.translation
            solution = np.eye(4)
            solution[:3, :3] = rot
            solution[:3, 3] = trans
            transformations.append(solution)

        # Convert back to tensor
        return torch.tensor(np.stack(transformations), dtype=torch.float32, device=device)
