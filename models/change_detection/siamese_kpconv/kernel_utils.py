"""
Kernel utilities for KPConv

This module provides utilities for creating and optimizing kernel points
for the KPConv convolution operations.
"""
import numpy as np
import torch


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    
    Args:
        sq_r: input radiuses [dn, ..., d1, d0]
        sig: extents of gaussians [d1, d0] or [d0] or float
        eps: small value to avoid division by zero
        
    Returns:
        gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3, fixed='center', 
                                    ratio=1.0, verbose=False):
    """
    Creation of kernel point via optimization of potentials.
    
    Args:
        radius: Radius of the kernels
        num_points: Number of points in the kernel
        num_kernels: Number of kernels to generate
        dimension: dimension of the space
        fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
        ratio: Ratio of the radius where you want the maximum kernel density
        verbose: Display optimization steps
        
    Returns:
        points [num_kernels, num_points, dimension]
    """
    # Parameter generation
    # ====================

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0
    diameter0 = 2 * radius0

    # Create points for kernel generations
    K_points_numpy = np.zeros((num_kernels, num_points, dimension))

    for i in range(num_kernels):
        # Random generation of initial positions
        # =====================================

        # Generate random point in the sphere with r<1
        randN = np.random.rand(num_points - 1, dimension) * 2 - 1
        normN = np.sqrt(np.sum(np.square(randN), axis=1))
        randN = randN / normN[:, np.newaxis] * np.cbrt(np.random.rand(num_points - 1))

        # Center is fixed
        K_points_numpy[i, 0, :] = 0
        K_points_numpy[i, 1:, :] = randN

    # Rescale kernels with real radius
    return radius * K_points_numpy


def load_kernels(radius, num_points, num_kernels=1, dimension=3, fixed='center'):
    """
    Load pre-computed kernel points.
    
    Args:
        radius: Radius of the kernels
        num_points: Number of points in the kernel
        num_kernels: Number of kernels to generate
        dimension: dimension of the space
        fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
        
    Returns:
        numpy array of kernel points
    """
    K_points_numpy = kernel_point_optimization_debug(
        radius,
        num_points,
        num_kernels=num_kernels,
        dimension=dimension,
        fixed=fixed
    )
    return K_points_numpy 