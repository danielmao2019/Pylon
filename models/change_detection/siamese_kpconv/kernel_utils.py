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
        sq_r: input squared distances [dn, ..., d1, d0]
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
    # Create points for kernel generations
    K_points_numpy = np.zeros((num_kernels, num_points, dimension))

    for k in range(num_kernels):
        # Initialize points
        kernel_points = np.zeros((num_points, dimension))
        
        # Create center point
        if fixed == "center" or fixed == "verticals":
            kernel_points[0, :] = 0
            num_free = num_points - 1
        else:
            num_free = num_points
        
        # Initialize with random points
        init_radius = 1.5
        
        # Optimizing parameters
        repulsion_strength = 1.0
        n_iter = 300
        min_dist = np.inf
        
        # Discrete gradient parameters
        epsilon = 1e-5
        gradient = np.zeros((num_free, dimension))
        
        # Use numpy batch operations for efficiency
        indices = np.arange(num_points)
        
        # Initialize free points
        if num_free > 0:
            # Random generation of initial positions
            # Generate random point in the sphere with r<1
            free_points = np.random.rand(num_free, dimension) * 2 * init_radius - init_radius
            if fixed == "center":
                # Generate points randomly in a sphere
                n_non_fixed = num_points - 1  # Assuming 'center' is fixed
                randN = np.random.rand(n_non_fixed, dimension) * 2 - 1
                normN = np.sqrt(np.sum(np.square(randN), axis=1))
                
                # Avoid division by zero
                normN = np.maximum(normN, 1e-10)
                
                # Use proper broadcasting
                scale_factor = np.cbrt(np.random.rand(n_non_fixed))
                randN = randN / normN.reshape(-1, 1) * scale_factor.reshape(-1, 1)
                kernel_points[1:, :] = randN
            elif fixed == "verticals":
                vertical_indices = np.zeros((num_points,), dtype=np.int32)
                for i in range(dimension):
                    vertical_indices[i+1:i+1+num_points//dimension] = i+1
                
                for i in range(num_free):
                    if vertical_indices[i+1] == 0:
                        kernel_points[i+1, :] = free_points[i, :]
                    else:
                        axis = vertical_indices[i+1] - 1
                        kernel_points[i+1, axis] = free_points[i, 0] * 2 - 1
                        for j in range(dimension):
                            if j != axis:
                                kernel_points[i+1, j] = 0
            else:
                kernel_points = free_points
        
        # Optimization loop - simulate repulsive forces between points
        for iter_count in range(n_iter):
            # Compute pair-wise distances between kernel points
            diff = kernel_points[:, np.newaxis, :] - kernel_points[np.newaxis, :, :]
            squared_dist = np.sum(diff * diff, axis=2)
            np.fill_diagonal(squared_dist, np.inf)
            closest_dist = np.min(squared_dist, axis=1)
            min_dist = np.min(closest_dist)
            
            # Update gradient directions for free points
            for i in range(num_free):
                point_idx = i + 1 if fixed in ["center", "verticals"] else i
                
                # Compute repulsive forces
                point_i = kernel_points[point_idx, :]
                other_indices = indices != point_idx
                if np.any(other_indices):
                    other_points = kernel_points[other_indices, :]
                    vectors = point_i[np.newaxis, :] - other_points
                    dist = np.sqrt(np.sum(vectors * vectors, axis=1))
                    direction = vectors / (dist[:, np.newaxis] + 1e-6)  # Small epsilon for stability
                    repulsion = (direction.T * repulsion_strength / (dist + 1e-6)).T  # Small epsilon for stability
                
                    # Apply constraint for points on vertical lines
                    if fixed == "verticals" and vertical_indices[point_idx] > 0:
                        axis = vertical_indices[point_idx] - 1
                        repulsion[:, (np.arange(dimension) != axis)] = 0
                    
                    gradient[i, :] = np.sum(repulsion, axis=0)
            
            # Apply gradient
            gradient_norm = np.sqrt(np.sum(gradient * gradient))
            if gradient_norm > 1e-6:  # Only apply if gradient is significant
                gradient = gradient / gradient_norm
                kernel_points_new = np.copy(kernel_points)
                
                if fixed == "center":
                    kernel_points_new[1:, :] = kernel_points[1:, :] + gradient * min_dist
                elif fixed == "verticals":
                    for i in range(num_free):
                        point_idx = i + 1
                        if vertical_indices[point_idx] > 0:
                            axis = vertical_indices[point_idx] - 1
                            kernel_points_new[point_idx, axis] = kernel_points[point_idx, axis] + gradient[i, axis] * min_dist
                        else:
                            kernel_points_new[point_idx, :] = kernel_points[point_idx, :] + gradient[i, :] * min_dist
                else:
                    kernel_points_new = kernel_points + gradient * min_dist
                
                # Update if we got an improvement
                diff_new = kernel_points_new[:, np.newaxis, :] - kernel_points_new[np.newaxis, :, :]
                squared_dist_new = np.sum(diff_new * diff_new, axis=2)
                np.fill_diagonal(squared_dist_new, np.inf)
                min_dist_new = np.min(np.min(squared_dist_new, axis=1))
                
                if min_dist_new > min_dist:
                    kernel_points = kernel_points_new
                    min_dist = min_dist_new
                else:
                    break
        
        # Scale kernel points to fit into a sphere of defined radius
        radius_fit = np.max(np.linalg.norm(kernel_points, axis=1))
        kernel_points = kernel_points * radius / radius_fit
        
        # Add dimension to ensure all points are within the unit sphere
        for i in range(num_points):
            norm = np.linalg.norm(kernel_points[i, :])
            if norm > radius:
                kernel_points[i, :] = kernel_points[i, :] * radius / norm
        
        # Store the kernel points
        K_points_numpy[k, ...] = kernel_points

    return K_points_numpy


def load_kernels(radius, num_kpoints=15, dimension=3, fixed="center", num_kernels=1):
    """
    Load kernels for KPConv
    
    Args:
        radius: radius of the kernel
        num_kpoints: number of points in the kernel
        dimension: dimension of the space (2D or 3D)
        fixed: policy for fixing kernel points ('center', 'verticals', or None)
        num_kernels: number of kernels to load
    
    Returns:
        Kernel points
    """
    return kernel_point_optimization_debug(
        radius=radius,
        num_points=num_kpoints,
        num_kernels=num_kernels,
        dimension=dimension,
        fixed=fixed
    ) 