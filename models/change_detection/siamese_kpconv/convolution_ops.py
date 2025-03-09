"""
KPConv convolution operations

This module contains all operations related to KPConv convolutions for 3D point clouds,
organized as follows:

1. Convolution Formats - Enum for different convolution approaches (PARTIAL_DENSE, MESSAGE_PASSING)
2. Base Classes - Base classes for convolution operations
3. Batch Normalization - FastBatchNorm1d implementation
4. KPConv Layer - Core KPConv implementation
5. Convolution Blocks - Wrapper blocks for KPConv (SimpleBlock, ResnetBBlock, KPDualBlock)

All convolution operations follow the PARTIAL_DENSE format from the original 
torch-points3d-SiameseKPConv implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union, Optional

from enum import Enum, auto

from models.change_detection.siamese_kpconv.kernel_utils import radius_gaussian, load_kernels
from models.change_detection.siamese_kpconv.utils import add_ones, knn, gather


#-------------------------------------------------------------------------------
# 1. Convolution Formats
#-------------------------------------------------------------------------------

class ConvolutionFormat(Enum):
    """Different types of convolution formats"""
    PARTIAL_DENSE = auto()  # Dense tensor operations with shadow points
    MESSAGE_PASSING = auto() # Graph-based message passing


#-------------------------------------------------------------------------------
# 2. Base Classes
#-------------------------------------------------------------------------------

class BasePartialDenseConvolution(nn.Module):
    """
    Base class for PARTIAL_DENSE convolution format.
    
    This format uses dense tensor operations with shadow points for non-existent neighbors.
    It's the core format used in KPConv.
    
    Attributes:
        CONV_TYPE: String identifier for the convolution type
    """
    CONV_TYPE = ConvolutionFormat.PARTIAL_DENSE.value
    
    def __init__(self):
        super(BasePartialDenseConvolution, self).__init__()
        
    def conv(self, x, pos, x_neighbour, pos_centered_neighbour, idx_neighbour, idx_sampler):
        """Generic convolution for partial dense data - to be implemented by subclasses"""
        raise NotImplementedError(
            "BasePartialDenseConvolution is an abstract class. Implement conv in a subclass."
        )
    
    def forward(self, x, pos, batch_x, pos_target, batch_target, k=16):
        """Forward pass for partial dense convolution
        
        Args:
            x: Input features
            pos: Input positions
            batch_x: Batch indices
            pos_target: Target positions
            batch_target: Target batch indices
            k: Number of neighbors
        
        Returns:
            Convolved features
        """
        # Get neighbors
        row_idx, col_idx = knn(pos, pos_target, k, batch_x, batch_target)
        
        # Apply convolution
        x_out = self.conv(pos_target, pos, col_idx, x)
        
        return x_out


#-------------------------------------------------------------------------------
# 3. Batch Normalization
#-------------------------------------------------------------------------------

class FastBatchNorm1d(nn.BatchNorm1d):
    """
    Batch normalization for 1D data (B, C) or (B, C, L)
    """
    
    def __init__(self, num_features, momentum=0.1, affine=True, track_running_stats=True):
        super(FastBatchNorm1d, self).__init__(
            num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )
    
    def forward(self, x):
        if x.dim() == 2:
            return super(FastBatchNorm1d, self).forward(x)
        else:
            # For 3D data (B, C, L), flatten to (B*L, C) for BN
            shape = x.shape
            flatten_input = x.transpose(1, 2).reshape(-1, shape[1])
            flatten_output = super(FastBatchNorm1d, self).forward(flatten_input)
            return flatten_output.reshape(shape[0], shape[2], shape[1]).transpose(1, 2)


#-------------------------------------------------------------------------------
# 4. KPConv Layer
#-------------------------------------------------------------------------------

def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    Args:
        sq_r: input squared distances [dn, ..., d1, d0]
        sig: extents of gaussians [d1, d0] or [d0] or float
        eps: small constant for numerical stability
    Returns:
        gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def kernel_point_optimization_debug(
    n_points: int = 40, 
    dim: int = 3, 
    fixed: str = "center", 
    ratio: float = 1.0, 
    verbose: bool = False
) -> np.ndarray:
    """
    Optimize the position of kernel points by simulating repulsive forces between them.

    Args:
        n_points: Number of points to optimize
        dim: Dimension of the space
        fixed: Strategy for fixing points ('center', 'verticals', or None)
        ratio: Ratio between kernel extent and influence range
        verbose: Whether to print optimization information

    Returns:
        Optimized kernel points positions [n_points, dim]
    """
    # Initialize points
    kernel_points = np.zeros((n_points, dim))
    
    # Create center point
    if fixed == "center" or fixed == "verticals":
        kernel_points[0, :] = 0
        num_free = n_points - 1
    else:
        num_free = n_points
    
    # Initialize with random points
    # From -1 to 1 with a repulsion radius of 1.5
    init_radius = 1.5
    
    # Optimizing parameters
    repulsion_strength = 1.0
    n_iter = 300
    min_dist = np.inf
    
    # Discrete gradient parameters
    epsilon = 1e-5
    gradient = np.zeros((num_free, dim))
    
    # Use numpy batch operations for efficiency
    indices = np.arange(n_points)
    
    # Initialize free points
    if num_free > 0:
        free_points = np.random.rand(num_free, dim) * 2 * init_radius - init_radius
        if fixed == "center":
            # Add the center point
            kernel_points[1:, :] = free_points
        elif fixed == "verticals":
            # Replace with vertical lines
            vertical_indices = np.zeros((n_points,), dtype=np.int32)
            for i in range(dim):
                vertical_indices[i+1:i+1+n_points//dim] = i+1
            
            for i in range(num_free):
                if vertical_indices[i+1] == 0:
                    # Random point not on a specific axis
                    kernel_points[i+1, :] = free_points[i, :]
                else:
                    # Point on a specific axis
                    axis = vertical_indices[i+1] - 1
                    kernel_points[i+1, axis] = free_points[i, 0] * 2 - 1
                    for j in range(dim):
                        if j != axis:
                            kernel_points[i+1, j] = 0
        else:
            # All points are free
            kernel_points = free_points
    
    for iter_count in range(n_iter):
        # Compute pair-wise distances between kernel points
        squared_dist = np.sum(np.square(kernel_points[:, np.newaxis, :] - kernel_points[np.newaxis, :, :]), axis=2)
        
        # Get distance to closest point for each free point
        # Ensure we don't compute the minimum to itself by setting the diagonal to a large value
        np.fill_diagonal(squared_dist, np.inf)
        closest_dist = np.min(squared_dist, axis=1)
        
        # Normalize the distances
        # Make sure min_dist is not 0 to prevent division by zero
        min_dist = np.min(closest_dist)
        if min_dist < epsilon:
            min_dist = epsilon
        
        # Update gradient directions for free points
        for i in range(num_free):
            if fixed == "center":
                point_idx = i + 1
            elif fixed == "verticals":
                point_idx = i + 1
            else:
                point_idx = i
            
            # Compute repulsive forces
            point_i = kernel_points[point_idx, :]
            other_indices = indices != point_idx
            if np.any(other_indices):  # Make sure there are other points
                other_points = kernel_points[other_indices, :]
                vectors = point_i[np.newaxis, :] - other_points
                
                # Compute squared distances with numeric stability
                squared = np.sum(vectors * vectors, axis=1)  # More stable than np.square()
                
                # Avoid sqrt of zero or negative values
                squared = np.maximum(squared, 1e-10)
                dist = np.sqrt(squared)
                
                # Calculate and normalize direction vectors
                direction = vectors / dist[:, np.newaxis]
                
                # Weight by inverse distance, with a minimum to avoid numeric issues
                # Use a smaller repulsion strength to avoid overflow
                safe_repulsion = min(repulsion_strength, 1e3)
                repulsion = (direction.T * safe_repulsion / dist).T
            
            # Apply constraint for points on vertical lines
            if fixed == "verticals" and vertical_indices[point_idx] > 0:
                axis = vertical_indices[point_idx] - 1
                repulsion[:, (np.arange(dim) != axis)] = 0
            
            # Compute gradient - sum if repulsion exists
            if 'repulsion' in locals() and repulsion.size > 0:
                gradient[i, :] = np.sum(repulsion, axis=0)
        
        # Check if gradient has valid values
        if np.all(np.isfinite(gradient)) and np.linalg.norm(gradient) > 0:
            # Normalize gradient with a safe norm calculation
            norm = np.linalg.norm(gradient)
            if norm > 1e-10:
                gradient = gradient / norm
            
            # Apply gradient to free points with a safe step size
            step_size = min(min_dist, 0.1)  # Limit step size for stability
            kernel_points_new = np.copy(kernel_points)
            
            if fixed == "center":
                kernel_points_new[1:, :] = kernel_points[1:, :] + gradient * step_size
            elif fixed == "verticals":
                for i in range(num_free):
                    point_idx = i + 1
                    if vertical_indices[point_idx] > 0:
                        axis = vertical_indices[point_idx] - 1
                        kernel_points_new[point_idx, axis] = kernel_points[point_idx, axis] + gradient[i, axis] * step_size
                    else:
                        kernel_points_new[point_idx, :] = kernel_points[point_idx, :] + gradient[i, :] * step_size
            else:
                kernel_points_new = kernel_points + gradient * step_size
        else:
            # If gradient has problems, just keep the current points
            kernel_points_new = np.copy(kernel_points)
        
        # Only update if we get an improvement - use a safer distance calculation
        try:
            # Compute pairwise differences avoiding squaring large numbers
            differences = kernel_points_new[:, np.newaxis, :] - kernel_points_new[np.newaxis, :, :]
            # Use a safer method to compute squared distances
            squared_dist_new = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:  # Skip diagonal
                        # Use a more numerically stable method
                        diff = kernel_points_new[i] - kernel_points_new[j]
                        squared_dist_new[i, j] = np.sum(diff * diff)
                    else:
                        squared_dist_new[i, j] = np.inf
            
            closest_dist_new = np.min(squared_dist_new, axis=1)
            min_dist_new = np.min(closest_dist_new)
            
            # Update if we got an improvement or if no previous min_dist has been set yet
            if min_dist_new > min_dist:
                kernel_points = kernel_points_new
                min_dist = min_dist_new
            else:
                # No improvement, stop the optimization
                break
        except Exception as e:
            # In case of numeric issues, just keep current points and stop
            print(f"Warning in kernel optimization: {e}")
            break
    
    # Scale kernel points to fit into a sphere of defined ratio
    radius = np.max(np.linalg.norm(kernel_points, axis=1))
    kernel_points = kernel_points * ratio / radius
    
    # Add dimension to ensure all points are within the unit sphere
    for i in range(n_points):
        if np.linalg.norm(kernel_points[i, :]) > ratio:
            kernel_points[i, :] = kernel_points[i, :] * ratio / np.linalg.norm(kernel_points[i, :])
    
    return kernel_points


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
    K_points_numpy = np.zeros((num_kernels, num_kpoints, dimension))
    
    for i in range(num_kernels):
        K_points_numpy[i, ...] = kernel_point_optimization_debug(
            num_kpoints, dimension, fixed, verbose=False
        )
    
    return K_points_numpy


class KPConvLayer(BasePartialDenseConvolution):
    """
    Apply kernel point convolution on a point cloud.
    
    This implements the core KPConv operation using the PARTIAL_DENSE format.
    It defines convolution kernels as sets of points in space and uses these
    to compute convolution operations on point clouds.
    
    Attributes:
        _INFLUENCE_TO_RADIUS: Ratio between point influence and kernel radius
    """
    _INFLUENCE_TO_RADIUS = 1.5

    def __init__(
        self,
        num_inputs,
        num_outputs,
        point_influence,
        n_kernel_points=15,
        fixed="center",
        KP_influence="linear",
        aggregation_mode="sum",
        dimension=3,
        add_one=False,
    ):
        super(KPConvLayer, self).__init__()
        self.kernel_radius = self._INFLUENCE_TO_RADIUS * point_influence
        self.point_influence = point_influence
        self.add_one = add_one
        self.num_inputs = num_inputs + self.add_one * 1
        self.num_outputs = num_outputs

        self.KP_influence = KP_influence
        self.n_kernel_points = n_kernel_points
        self.aggregation_mode = aggregation_mode

        # Initial kernel extent for this layer
        K_points_numpy = load_kernels(
            self.kernel_radius,
            n_kernel_points,
            num_kernels=1,
            dimension=dimension,
            fixed=fixed,
        )

        self.K_points = nn.Parameter(
            torch.from_numpy(K_points_numpy.reshape((n_kernel_points, dimension))).to(torch.float),
            requires_grad=False,
        )

        weights = torch.empty([n_kernel_points, self.num_inputs, num_outputs], dtype=torch.float)
        nn.init.xavier_normal_(weights)
        self.weight = nn.Parameter(weights)

    def conv(self, query_points, support_points, neighbors, x):
        """
        Kernel Point Convolution - implementation of BasePartialDenseConvolution's conv method
        
        Args:
            query_points: Target points where features will be computed [N, 3]
            support_points: Source points providing features [N0, 3]
            neighbors: Indices of neighbors from support to query [N, M]
            x: Features of support points [N0, C]
            
        Returns:
            Output features of size [N, C']
        """
        x = add_ones(support_points, x, self.add_one)

        # Add a fake point in the last row for shadow neighbors
        shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
        support_points = torch.cat([support_points, shadow_point], dim=0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors_points = support_points[neighbors]

        # Center every neighborhood
        neighbors_points = neighbors_points - query_points.unsqueeze(1)

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors_points = neighbors_points.unsqueeze(2)
        differences = neighbors_points - self.K_points.unsqueeze(0).unsqueeze(0)

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences**2, dim=3)

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == "constant":
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = all_weights.transpose(2, 1)
        elif self.KP_influence == "linear":
            # Influence decrease linearly with the distance, and get to zero when d = point_influence.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.point_influence, min=0.0)
            all_weights = all_weights.transpose(2, 1)
        elif self.KP_influence == "gaussian":
            # Influence in gaussian of the distance.
            sigma = self.point_influence * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = all_weights.transpose(2, 1)
        else:
            raise ValueError("Unknown influence function type")

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == "closest":
            neighbors_1nn = torch.argmin(sq_distances, dim=-1)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K_points.shape[0]), 1, 2)
        elif self.aggregation_mode != "sum":
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        features = torch.cat([x, torch.zeros_like(x[:1, :])], dim=0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighborhood_features = features[neighbors]

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighborhood_features)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute(1, 0, 2)
        kernel_outputs = torch.matmul(weighted_features, self.weight)

        # Convolution sum to get [n_points, out_fdim]
        output_features = torch.sum(kernel_outputs, dim=0)

        return output_features
        
    # For backward compatibility with direct calls
    forward = conv


class SimpleBlock(BasePartialDenseConvolution):
    """
    Simple block with KPConv convolution -> activation -> BN
    
    This block implements a simple KPConv layer followed by batch normalization
    and activation, using the PARTIAL_DENSE convolution format.
    """

    def __init__(
        self,
        down_conv_nn=None,
        point_influence=0.05,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        bn_momentum=0.02,
        bn=FastBatchNorm1d,
        deformable=False,
        add_one=False,
        **kwargs,
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn

        if deformable:
            # Not implemented deformable yet
            raise NotImplementedError("Deformable KPConv not implemented yet")
        else:
            self.kp_conv = KPConvLayer(
                num_inputs, num_outputs, point_influence=point_influence, add_one=add_one, **kwargs
            )

        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation

    def conv(self, query_points, support_points, neighbors, x):
        """
        Convolution method implementation for BasePartialDenseConvolution
        
        Args:
            query_points: Target points where features will be computed
            support_points: Source points providing features
            neighbors: Indices of neighbors from support to query points
            x: Features of support points
            
        Returns:
            Processed features with convolution, BN and activation
        """
        # Apply KPConv
        x_out = self.kp_conv(query_points, support_points, neighbors, x)
        
        # Apply batch norm and activation
        if self.bn:
            x_out = self.bn(x_out)
        x_out = self.activation(x_out)
        
        return x_out


class ResnetBBlock(BasePartialDenseConvolution):
    """
    Bottleneck Resnet block for KPConv
    
    This implements a ResNet bottleneck block with KPConv as the convolution operation.
    It follows the PARTIAL_DENSE convolution format.
    """

    def __init__(
        self,
        down_conv_nn=None,
        point_influence=0.05,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        has_bottleneck=True,
        bn_momentum=0.02,
        bn=FastBatchNorm1d,
        deformable=False,
        add_one=False,
        **kwargs,
    ):
        super(ResnetBBlock, self).__init__()
        assert len(down_conv_nn) == 2
        
        num_inputs, num_outputs = down_conv_nn
        
        if has_bottleneck:
            # Create a bottleneck architecture
            bottleneck_features = num_outputs // 4
            self.mlp_in = nn.Sequential(
                nn.Linear(num_inputs, bottleneck_features, bias=False),
                bn(bottleneck_features, momentum=bn_momentum),
                activation
            )
            self.kp_conv = KPConvLayer(
                bottleneck_features, bottleneck_features, 
                point_influence=point_influence, 
                add_one=add_one, 
                **kwargs
            )
            self.mlp_out = nn.Sequential(
                nn.Linear(bottleneck_features, num_outputs, bias=False),
                bn(num_outputs, momentum=bn_momentum)
            )
        else:
            # No bottleneck
            self.mlp_in = nn.Identity()
            self.kp_conv = KPConvLayer(
                num_inputs, num_outputs, 
                point_influence=point_influence, 
                add_one=add_one, 
                **kwargs
            )
            self.mlp_out = nn.Sequential(
                bn(num_outputs, momentum=bn_momentum)
            )
        
        # Shortcut
        if num_inputs != num_outputs:
            self.shortcut = nn.Sequential(
                nn.Linear(num_inputs, num_outputs, bias=False),
                bn(num_outputs, momentum=bn_momentum)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.activation = activation

    def conv(self, query_points, support_points, neighbors, x):
        """
        Convolution method implementation for BasePartialDenseConvolution
        
        Args:
            query_points: Target points where features will be computed
            support_points: Source points providing features
            neighbors: Indices of neighbors from support to query points
            x: Features of support points
            
        Returns:
            Processed features with bottleneck convolution and residual connection
        """
        # Shortcut
        shortcut = self.shortcut(x)
        
        # Apply MLP in
        x = self.mlp_in(x)
        
        # Apply KPConv
        x = self.kp_conv(query_points, support_points, neighbors, x)
        
        # Apply MLP out
        x = self.mlp_out(x)
        
        # Add shortcut
        x = x + shortcut
        
        # Apply activation
        x = self.activation(x)
        
        return x


class KPDualBlock(BasePartialDenseConvolution):
    """
    Dual KPConv block (typically a combination of blocks, e.g., SimpleBlock + ResnetBBlock)
    
    This implements a sequence of KPConv blocks that are applied one after another.
    It follows the PARTIAL_DENSE convolution format.
    """

    def __init__(
        self,
        block_names=None,
        down_conv_nn=None,
        point_influence=0.05,
        has_bottleneck=None,
        max_num_neighbors=None,
        bn_momentum=0.02,
        deformable=False,
        add_one=False,
        **kwargs,
    ):
        super(KPDualBlock, self).__init__()
        
        assert len(block_names) == len(down_conv_nn)
        self.blocks = nn.ModuleList()
        
        for i, class_name in enumerate(block_names):
            # Get the appropriate block class
            if class_name == "SimpleBlock":
                block_cls = SimpleBlock
            elif class_name == "ResnetBBlock":
                block_cls = ResnetBBlock
            else:
                raise ValueError(f"Unknown block name: {class_name}")
            
            # Create the block
            has_bn = has_bottleneck[i] if has_bottleneck is not None else True
            deform = deformable[i] if isinstance(deformable, list) else deformable
            add_one_i = add_one[i] if isinstance(add_one, list) else add_one
            
            block = block_cls(
                down_conv_nn=down_conv_nn[i],
                point_influence=point_influence,
                has_bottleneck=has_bn,
                bn_momentum=bn_momentum,
                deformable=deform,
                add_one=add_one_i,
                **kwargs,
            )
            self.blocks.append(block)

    def conv(self, query_points, support_points, neighbors, x):
        """
        Convolution method implementation for BasePartialDenseConvolution
        
        Args:
            query_points: Target points where features will be computed
            support_points: Source points providing features
            neighbors: Indices of neighbors from support to query points
            x: Features of support points
            
        Returns:
            Processed features after passing through the sequence of blocks
        """
        # Pass through each block in sequence
        for block in self.blocks:
            x = block.conv(query_points, support_points, neighbors, x)
        return x
