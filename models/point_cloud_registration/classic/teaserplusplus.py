# Reference: https://teaser.readthedocs.io/en/latest/quickstart.html#usage-in-python-projects
import torch
import torch.nn as nn
import numpy as np
import teaserpp_python


class TeaserPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        TEASER++ registration.
        
        Args:
            source: Source point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            Transformation matrix (B, 4, 4)
        """
        batch_size = source.shape[0]
        device = source.device
        
        # Convert to numpy for TEASER++
        source_np = source.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Process each batch
        transformations = []
        for i in range(batch_size):
            # Set up TEASER++ parameters
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = 0.01
            solver_params.estimate_scaling = True
            solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12
            
            # Create solver and solve
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(source_np[i], target_np[i])
            
            # Get solution
            solution = solver.getSolution()
            transformations.append(solution)
        
        # Convert back to tensor
        return torch.tensor(np.stack(transformations), device=device)
