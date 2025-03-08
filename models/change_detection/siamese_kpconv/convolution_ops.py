"""
KPConv convolution operations

This module contains operations for KPConv, including the main KPConvLayer
and the SimpleBlock that uses it.
"""
import torch
import torch.nn as nn
from models.change_detection.siamese_kpconv.kernel_utils import radius_gaussian, load_kernels
from models.change_detection.siamese_kpconv.utils import add_ones


class FastBatchNorm1d(nn.BatchNorm1d):
    """
    Batch norm implementation with improved memory usage
    """
    def __init__(self, num_features, momentum=0.1, affine=True):
        super(FastBatchNorm1d, self).__init__(num_features, momentum=momentum, affine=affine)

    def forward(self, x):
        if x.dim() > 2:
            x = x.transpose(1, 2)
            y = super(FastBatchNorm1d, self).forward(x)
            y = y.transpose(1, 2)
        else:
            y = super(FastBatchNorm1d, self).forward(x)
        return y


class KPConvLayer(nn.Module):
    """
    Apply kernel point convolution on a point cloud
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

    def forward(self, query_points, support_points, neighbors, x):
        """
        Kernel Point Convolution
        
        Args:
            query_points(torch Tensor): query of size N x 3
            support_points(torch Tensor): support points of size N0 x 3
            neighbors(torch Tensor): neighbors of size N x M
            x : feature of size N0 x d (d is the number of inputs)
            
        Returns:
            output features of size [n_points, out_fdim]
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


class SimpleBlock(nn.Module):
    """
    Simple layer with KPConv convolution -> activation -> BN
    """
    def __init__(
        self,
        down_conv_nn=None,
        sigma=1.0,
        point_influence=0.025,  # This replaces prev_grid_size * sigma
        max_num_neighbors=16,
        activation=nn.LeakyReLU(negative_slope=0.1),
        bn_momentum=0.02,
        bn=FastBatchNorm1d,
        deformable=False,
        add_one=False,
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn
        
        self.kp_conv = KPConvLayer(
            num_inputs, num_outputs, point_influence, add_one=add_one
        )
        
        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x, pos, batch_x, pos_target, batch_target, k=16):
        """
        Forward pass of the SimpleBlock
        
        Args:
            x: input features [N, C]
            pos: input positions [N, 3]
            batch_x: batch indices [N]
            pos_target: target positions [M, 3]
            batch_target: target batch indices [M]
            k: number of neighbors to use
            
        Returns:
            output features of size [M, C']
        """
        # Find neighbors
        from torch_geometric.nn import knn
        if pos.shape[0] == pos_target.shape[0]:
            idx_neighbors = knn(pos, pos, k, batch_x, batch_x)
        else:
            idx_neighbors = knn(pos, pos_target, k, batch_x, batch_target)
        
        # Apply KPConv
        x_out = self.kp_conv(pos_target, pos, idx_neighbors[1], x)
        
        # Apply batch norm and activation
        if self.bn:
            x_out = self.bn(x_out)
        x_out = self.activation(x_out)
        
        return x_out 