# Reference: https://teaser.readthedocs.io/en/latest/quickstart.html#usage-in-python-projects
import numpy as np
import teaserpp_python


def teaserplusplus(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.01
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("TEASER++ Parameters are:", solver_params)

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(source, target)

    solution = solver.getSolution()
    return solution
