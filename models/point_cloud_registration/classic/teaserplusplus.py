# Reference: https://teaser.readthedocs.io/en/latest/quickstart.html#usage-in-python-projects
from typing import Dict
import torch
import numpy as np
import teaserpp_python


class TeaserPlusPlus(torch.nn.Module):

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
        batch_size = inputs['src_pc']['pos'].shape[0]
        device = inputs['src_pc']['pos'].device

        # Convert to numpy for TEASER++
        source_np = inputs['src_pc']['pos'].detach().cpu().numpy()
        target_np = inputs['tgt_pc']['pos'].detach().cpu().numpy()

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
        return torch.tensor(np.stack(transformations), dtype=torch.float32, device=device)
