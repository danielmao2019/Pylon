# HANet: High-Resolution Attention Networks for Change Detection

## Overview
HANet is a deep learning model designed for change detection in satellite imagery. It employs a high-resolution attention mechanism to effectively capture spatial dependencies and changes between bi-temporal images.

## Architecture

### Key Components
1. **Backbone Network**
   - ResNet-based feature extraction
   - Multi-scale feature representation

2. **High-Resolution Attention Module**
   - Row and Column attention mechanisms (CAM_Module)
   - Position-sensitive feature aggregation
   - Scale-adaptive processing

3. **Change Detection Head**
   - Feature fusion
   - Binary change prediction

## Implementation Details

### Attention Mechanism
The core of HANet is its attention mechanism, implemented in the `CAM_Module` class. This module:
1. Projects input features into query and key spaces
2. Computes attention weights through matrix multiplication
3. Applies attention to value features
4. Uses residual connections for stable training

```python
class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
```

## Numerical Stability Issue

### Problem Description
During training at 256x256 resolution, the model encountered NaN values in its outputs. The issue was traced to numerical instability in the attention computation within the `CAM_Module`.

### Root Cause Analysis
The issue stemmed from the attention computation in `CAM_Module`:
- Large feature maps (256x256) led to high magnitude values in query-key matrix multiplication
- These large values caused numerical overflow in the subsequent softmax operation
- The overflow resulted in NaN values propagating through the network

### Solution
The fix involved applying proper scaling in the attention computation:

1. **Scale Factor Implementation**
   ```python
   # In CAM_Module.forward
   B, C, H, W = x.size()
   scale = 1.0 / math.sqrt(H * W)  # Scale factor
   
   # Apply scaling to query and key projections
   proj_query = self.query_conv(x) * math.sqrt(scale)
   proj_key = self.key_conv(x) * math.sqrt(scale)
   ```

2. **Attention Computation**
   ```python
   energy = torch.bmm(proj_query, proj_key)
   attention = F.softmax(energy, dim=-1)
   out = torch.bmm(attention, proj_value)
   ```

The scaling factor `1/sqrt(H * W)` is distributed across query and key projections to prevent numerical overflow while maintaining the mathematical relationships in the attention mechanism.
