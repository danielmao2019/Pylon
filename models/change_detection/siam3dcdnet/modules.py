"""
Core modules for the 3DCDNet model implementation.

This file includes the modules from the original 3DCDNet implementation:
- LocalFeatureAggregation (LFA) module and its components (SPE, LFE)
- Convolution modules (Conv1d, Conv2d)
- Utility functions (gather_neighbour)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any


class _BNBase(nn.Sequential):
    """Base class for BatchNorm modules."""
    
    def __init__(self, in_size: int, batch_norm: Any = None, name: str = ""):
        """Initialize _BNBase.
        
        Args:
            in_size: Input dimension
            batch_norm: Batch normalization module to use
            name: Optional name prefix for the layers
        """
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))
        
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    """1D batch normalization module."""
    
    def __init__(self, in_size: int, *, name: str = ""):
        """Initialize BatchNorm1d.
        
        Args:
            in_size: Input dimension
            name: Optional name prefix for the layers
        """
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    """2D batch normalization module."""
    
    def __init__(self, in_size: int, name: str = ""):
        """Initialize BatchNorm2d.
        
        Args:
            in_size: Input dimension
            name: Optional name prefix for the layers
        """
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):
    """Base class for convolution modules."""
    
    def __init__(
            self,
            in_size: int,
            out_size: int,
            kernel_size: Any,
            stride: Any,
            padding: Any,
            activation: Any,
            bn: bool,
            init: Any,
            conv: Any = None,
            batch_norm: Any = None,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm: bool = False,
            instance_norm_func: Any = None
    ):
        """Initialize _ConvBase.
        
        Args:
            in_size: Input dimension
            out_size: Output dimension
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            activation: Activation function to use
            bn: Whether to use batch normalization
            init: Initialization function for the convolution weights
            conv: Convolution module to use
            batch_norm: Batch normalization module to use
            bias: Whether to use bias in the convolution
            preact: Whether to apply batch norm and activation before convolution
            name: Optional name prefix for the layers
            instance_norm: Whether to use instance normalization
            instance_norm_func: Instance normalization module to use
        """
        super().__init__()
        
        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
            
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)
                
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
                
            if activation is not None:
                self.add_module(name + 'activation', activation)
                
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
                
        self.add_module(name + 'conv', conv_unit)
        
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
                
            if activation is not None:
                self.add_module(name + 'activation', activation)
                
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class Conv1d(_ConvBase):
    """1D convolution with batch normalization and activation."""
    
    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        """Initialize Conv1d.
        
        Args:
            in_size: Input dimension
            out_size: Output dimension
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            activation: Activation function to use
            bn: Whether to use batch normalization
            init: Initialization function for the convolution weights
            bias: Whether to use bias in the convolution
            preact: Whether to apply batch norm and activation before convolution
            name: Optional name prefix for the layers
            instance_norm: Whether to use instance normalization
        """
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):
    """2D convolution with batch normalization and activation."""
    
    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        """Initialize Conv2d.
        
        Args:
            in_size: Input dimension
            out_size: Output dimension
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            activation: Activation function to use
            bn: Whether to use batch normalization
            init: Initialization function for the convolution weights
            bias: Whether to use bias in the convolution
            preact: Whether to apply batch norm and activation before convolution
            name: Optional name prefix for the layers
            instance_norm: Whether to use instance normalization
        """
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


def gather_neighbour(pc: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather the coordinates or features of neighboring points.
    
    Args:
        pc: Point cloud features of shape [B, C, N, 1]
        neighbor_idx: Neighbor indices of shape [B, N, K]
        
    Returns:
        Gathered features of shape [B, C, N, K]
    """
    pc = pc.transpose(2, 1).squeeze(-1)  # B, N, C
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)  # B, N*K
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))  # B, N*K, C
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # B, N, K, C
    features = features.permute(0, 3, 1, 2)  # B, C, N, K
    return features


class SPE(nn.Module):
    """Spatial Points Encoding module from 3DCDNet."""
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize SPE module.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super().__init__()
        self.mlp2 = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
    def forward(self, feature: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPE module.
        
        Args:
            feature: Feature tensor of shape [B, C, N, 1]
            neigh_idx: Neighbor indices of shape [B, N, K]
            
        Returns:
            Output features of shape [B, C_out, N, 1]
        """
        f_neigh = gather_neighbour(feature, neigh_idx)  # B, C, N, K
        f_neigh = torch.cat((feature, f_neigh), -1)  # B, C, N, K+1
        f_agg2 = self.mlp2(f_neigh)  # B, C_out, N, K+1
        f_agg2 = torch.sum(f_agg2, -1, keepdim=True)  # B, C_out, N, 1
        return f_agg2


class LFE(nn.Module):
    """Local Feature Extraction module from 3DCDNet."""
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize LFE module.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super().__init__()
        self.mlp1 = Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.mlp2 = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp3 = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
   
    def forward(self, feature: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of LFE module.
        
        Args:
            feature: Feature tensor of shape [B, C, N, 1]
            neigh_idx: Neighbor indices of shape [B, N, K]
            
        Returns:
            Output features of shape [B, C_out, N, 1]
        """
        f_neigh = gather_neighbour(feature, neigh_idx)  # B, C, N, K
        f_neigh = self.mlp1(f_neigh)  # B, C, N, K
        f_neigh = torch.sum(f_neigh, dim=-1, keepdim=True)  # B, C, N, 1
        f_neigh = self.mlp2(f_neigh)  # B, C_out, N, 1
        feature = self.mlp3(feature)  # B, C_out, N, 1
        f_agg = f_neigh + feature  # B, C_out, N, 1
        return f_agg


class LFA(nn.Module):
    """Local Feature Aggregation module from 3DCDNet."""
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize LFA module.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super().__init__()
        self.spe = SPE(d_in, d_out)
        self.lfe = LFE(d_in, d_out)
        self.mlp = Conv2d(d_out, d_out, kernel_size=(1, 1), bn=True)
     
    def forward(self, feature: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of LFA module.
        
        Args:
            feature: Feature tensor of shape [B, C, N, 1]
            neigh_idx: Neighbor indices of shape [B, N, K]
            
        Returns:
            Output features of shape [B, C_out, N, 1]
        """
        spe = self.spe(feature, neigh_idx)  # B, C_out, N, 1
        lfe = self.lfe(feature, neigh_idx)  # B, C_out, N, 1
        f_agg = spe + lfe  # B, C_out, N, 1
        f_agg = self.mlp(f_agg)  # B, C_out, N, 1
        return f_agg
