# Semantic Scene Completion Methods

## Overview
Semantic Scene Completion (SSC) methods reconstruct complete 3D semantic volumes from partial observations. Unlike projection-based methods that only label visible points, SSC approaches generate dense voxel grids with both geometry and semantics, effectively "hallucinating" occluded regions based on learned priors.

## Core Concepts

### 1. Problem Formulation
```python
class SemanticSceneCompletion:
    """
    Input: Partial observations (RGB-D images, point clouds)
    Output: Dense 3D voxel grid with semantic labels
    
    Voxel Grid: [X, Y, Z, C] where C = num_classes + 1 (including empty space)
    """
    
    def __init__(
        self,
        voxel_size: float = 0.05,  # 5cm voxels
        scene_size: Tuple[float, float, float] = (10.0, 10.0, 3.0),  # meters
        num_classes: int = 20
    ):
        self.voxel_size = voxel_size
        self.grid_dims = tuple(int(s / voxel_size) for s in scene_size)
        self.num_classes = num_classes + 1  # +1 for empty space
```

### 2. Voxel Representation
```python
def pointcloud_to_voxel_grid(
    points: torch.Tensor,  # [N, 3]
    labels: Optional[torch.Tensor] = None,  # [N]
    voxel_size: float = 0.05,
    grid_dims: Tuple[int, int, int] = (200, 200, 60),
    origin: torch.Tensor = None  # [3] world origin of voxel grid
) -> torch.Tensor:
    """
    Convert point cloud to voxel occupancy grid.
    
    Returns:
        voxel_grid: [X, Y, Z] binary occupancy or [X, Y, Z, C] semantic
    """
    if origin is None:
        origin = points.min(dim=0)[0]
    
    # Compute voxel indices
    voxel_coords = ((points - origin) / voxel_size).long()
    
    # Filter out-of-bounds voxels
    valid_mask = (
        (voxel_coords >= 0).all(dim=1) &
        (voxel_coords[:, 0] < grid_dims[0]) &
        (voxel_coords[:, 1] < grid_dims[1]) &
        (voxel_coords[:, 2] < grid_dims[2])
    )
    voxel_coords = voxel_coords[valid_mask]
    
    if labels is None:
        # Binary occupancy grid
        voxel_grid = torch.zeros(grid_dims, dtype=torch.float32)
        voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1.0
    else:
        # Semantic grid
        labels = labels[valid_mask]
        voxel_grid = torch.zeros((*grid_dims, self.num_classes))
        voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], labels] = 1.0
    
    return voxel_grid
```

## MonoScene Architecture (CVPR 2022)

### 1. 2D-3D Feature Lifting
```python
class MonoScene(nn.Module):
    def __init__(
        self,
        backbone_2d: str = "efficientnet-b7",
        voxel_dims: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 20,
        feature_dim: int = 128
    ):
        super().__init__()
        
        # 2D Feature Extraction
        self.backbone_2d = EfficientNet.from_pretrained(backbone_2d)
        
        # 2D-to-3D Feature Projection
        self.feature_projector = FrustumFeatureProjection(
            image_dims=(480, 640),
            voxel_dims=voxel_dims,
            feature_dim=feature_dim
        )
        
        # 3D U-Net for completion
        self.unet_3d = UNet3D(
            in_channels=feature_dim,
            out_channels=num_classes,
            feature_dims=[128, 256, 256, 256]
        )
        
        # Semantic segmentation head
        self.semantic_head = nn.Conv3d(256, num_classes, kernel_size=1)
```

### 2. Frustum Feature Projection
```python
class FrustumFeatureProjection(nn.Module):
    """Project 2D features into 3D frustum voxels."""
    
    def __init__(
        self,
        image_dims: Tuple[int, int],
        voxel_dims: Tuple[int, int, int],
        feature_dim: int
    ):
        super().__init__()
        self.image_dims = image_dims
        self.voxel_dims = voxel_dims
        
        # Precompute voxel-to-pixel mappings
        self.voxel_to_pixel = self._compute_projection_matrix()
        
        # MLP for feature transformation
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def _compute_projection_matrix(self):
        """Precompute which pixels each voxel projects to."""
        X, Y, Z = self.voxel_dims
        H, W = self.image_dims
        
        # Create 3D grid of voxel centers
        xx, yy, zz = torch.meshgrid(
            torch.arange(X), torch.arange(Y), torch.arange(Z)
        )
        voxel_centers = torch.stack([xx, yy, zz], dim=-1).float()
        
        # Convert to world coordinates
        voxel_centers = voxel_centers * self.voxel_size + self.origin
        
        # Project to image plane
        pixel_coords = project_3d_to_2d(
            voxel_centers.reshape(-1, 3),
            self.camera_intrinsics,
            self.camera_extrinsics
        )
        
        return pixel_coords.reshape(X, Y, Z, 2)
    
    def forward(
        self,
        features_2d: torch.Tensor,  # [B, C, H, W]
        depth_map: Optional[torch.Tensor] = None  # [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Lift 2D features to 3D frustum.
        
        Returns:
            features_3d: [B, C, X, Y, Z]
        """
        B, C, H, W = features_2d.shape
        X, Y, Z = self.voxel_dims
        
        features_3d = torch.zeros(B, C, X, Y, Z, device=features_2d.device)
        
        # For each voxel, sample corresponding 2D feature
        for b in range(B):
            # Flatten for grid_sample
            pixel_coords_norm = self.voxel_to_pixel / torch.tensor([W, H]) * 2 - 1
            pixel_coords_norm = pixel_coords_norm.reshape(1, -1, 1, 2)  # [1, X*Y*Z, 1, 2]
            
            # Sample features
            sampled = F.grid_sample(
                features_2d[b:b+1],
                pixel_coords_norm,
                mode='bilinear',
                align_corners=False
            )  # [1, C, X*Y*Z, 1]
            
            # Reshape to 3D
            features_3d[b] = sampled.squeeze().reshape(C, X, Y, Z)
            
            # Apply depth weighting if available
            if depth_map is not None:
                depth_weights = self._compute_depth_weights(depth_map[b], pixel_coords_norm)
                features_3d[b] *= depth_weights.unsqueeze(0)
        
        return features_3d
```

### 3. 3D Completion Network
```python
class UNet3D(nn.Module):
    """3D U-Net for voxel completion and refinement."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_dims: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_dim = in_channels
        for out_dim in feature_dims:
            self.encoders.append(
                self._make_encoder_block(in_dim, out_dim)
            )
            in_dim = out_dim
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(feature_dims[-1], feature_dims[-1] * 2, 3, padding=1),
            nn.BatchNorm3d(feature_dims[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dims[-1] * 2, feature_dims[-1], 3, padding=1),
            nn.BatchNorm3d(feature_dims[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(feature_dims) - 1, 0, -1):
            self.decoders.append(
                self._make_decoder_block(
                    feature_dims[i] + feature_dims[i-1],
                    feature_dims[i-1]
                )
            )
        
        # Output layer
        self.output = nn.Conv3d(feature_dims[0], out_channels, 1)
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            x = torch.cat([x, encoder_features[-(i+2)]], dim=1)
            x = decoder(x)
        
        return self.output(x)
```

## VoxFormer Architecture (CVPR 2023)

### 1. Query-Based Voxel Generation
```python
class VoxFormer(nn.Module):
    """Transformer-based voxel completion using learnable queries."""
    
    def __init__(
        self,
        num_queries: int = 10000,
        hidden_dim: int = 256,
        num_classes: int = 20,
        voxel_dims: Tuple[int, int, int] = (200, 200, 16)
    ):
        super().__init__()
        
        # Learnable voxel queries
        self.voxel_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Image feature extractor
        self.backbone_2d = ResNet50FPN()
        
        # Deformable attention for 2D-3D interaction
        self.deformable_attention = DeformableAttention3D(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_levels=4,
            num_points=8
        )
        
        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=6
        )
        
        # Voxel prediction heads
        self.occupancy_head = nn.Linear(hidden_dim, 1)
        self.semantic_head = nn.Linear(hidden_dim, num_classes)
        self.location_head = nn.Linear(hidden_dim, 3)  # Predict 3D location
```

### 2. Deformable 3D Attention
```python
class DeformableAttention3D(nn.Module):
    """Deformable attention for sparse 3D voxel queries."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        
        # Sampling offset network
        self.sampling_offsets = nn.Linear(
            hidden_dim,
            num_heads * num_levels * num_points * 3
        )
        
        # Attention weight network
        self.attention_weights = nn.Linear(
            hidden_dim,
            num_heads * num_levels * num_points
        )
        
        # Value projection
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,  # [B, Q, C]
        reference_points: torch.Tensor,  # [B, Q, 3]
        input_features: List[torch.Tensor],  # Multi-scale features
        spatial_shapes: List[Tuple[int, int, int]]
    ):
        B, Q, C = query.shape
        
        # Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query)  # [B, Q, H*L*P*3]
        sampling_offsets = sampling_offsets.view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 3
        )
        
        # Predict attention weights
        attention_weights = self.attention_weights(query)  # [B, Q, H*L*P]
        attention_weights = attention_weights.view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, Q, self.num_heads, self.num_levels, self.num_points
        )
        
        # Sample features at offset locations
        sampled_features = []
        for lvl, (feat, shape) in enumerate(zip(input_features, spatial_shapes)):
            # Compute sampling locations
            sampling_locations = reference_points.unsqueeze(2).unsqueeze(3) + \
                                sampling_offsets[:, :, :, lvl]
            
            # Normalize to [-1, 1]
            sampling_locations = sampling_locations / torch.tensor(shape).to(sampling_locations.device)
            sampling_locations = sampling_locations * 2 - 1
            
            # Sample features
            sampled = F.grid_sample(
                feat, sampling_locations,
                mode='bilinear', align_corners=False
            )
            sampled_features.append(sampled)
        
        # Weighted aggregation
        output = torch.stack(sampled_features, dim=2)  # [B, Q, L, C]
        output = output * attention_weights.unsqueeze(-1)
        output = output.sum(dim=(2, 3))  # [B, Q, C]
        
        return self.output_proj(output)
```

### 3. Sparse-to-Dense Voxel Prediction
```python
def sparse_to_dense_voxels(
    voxel_queries: torch.Tensor,  # [Q, C]
    predicted_locations: torch.Tensor,  # [Q, 3]
    predicted_occupancy: torch.Tensor,  # [Q, 1]
    predicted_semantics: torch.Tensor,  # [Q, num_classes]
    voxel_dims: Tuple[int, int, int],
    occupancy_threshold: float = 0.5
) -> torch.Tensor:
    """
    Convert sparse query predictions to dense voxel grid.
    """
    X, Y, Z = voxel_dims
    dense_grid = torch.zeros(X, Y, Z, predicted_semantics.shape[1] + 1)
    
    # Filter by occupancy
    occupied_mask = predicted_occupancy.squeeze() > occupancy_threshold
    occupied_locations = predicted_locations[occupied_mask]
    occupied_semantics = predicted_semantics[occupied_mask]
    
    # Discretize locations to voxel indices
    voxel_indices = (occupied_locations * torch.tensor(voxel_dims)).long()
    
    # Clip to valid range
    voxel_indices = torch.clamp(voxel_indices, 0, torch.tensor(voxel_dims) - 1)
    
    # Fill dense grid
    dense_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], 1:] = occupied_semantics
    dense_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], 0] = 1  # Occupancy
    
    return dense_grid
```

## Training Strategies

### 1. Multi-Task Loss
```python
class SemanticCompletionLoss(nn.Module):
    def __init__(
        self,
        semantic_weight: float = 1.0,
        geometry_weight: float = 1.0,
        lovasz_weight: float = 0.5
    ):
        super().__init__()
        self.semantic_weight = semantic_weight
        self.geometry_weight = geometry_weight
        self.lovasz_weight = lovasz_weight
        
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.geometry_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszSoftmax()
    
    def forward(
        self,
        pred_semantics: torch.Tensor,  # [B, C, X, Y, Z]
        pred_geometry: torch.Tensor,   # [B, 1, X, Y, Z]
        gt_semantics: torch.Tensor,    # [B, X, Y, Z]
        gt_geometry: torch.Tensor      # [B, X, Y, Z]
    ):
        # Semantic segmentation loss
        loss_semantic = self.semantic_loss(pred_semantics, gt_semantics)
        
        # Geometry completion loss
        loss_geometry = self.geometry_loss(pred_geometry.squeeze(1), gt_geometry.float())
        
        # Lovasz loss for better IoU
        loss_lovasz = self.lovasz_loss(
            F.softmax(pred_semantics, dim=1),
            gt_semantics
        )
        
        total_loss = (
            self.semantic_weight * loss_semantic +
            self.geometry_weight * loss_geometry +
            self.lovasz_weight * loss_lovasz
        )
        
        return total_loss, {
            'semantic': loss_semantic.item(),
            'geometry': loss_geometry.item(),
            'lovasz': loss_lovasz.item()
        }
```

### 2. Coarse-to-Fine Training
```python
def train_coarse_to_fine(
    model: SemanticSceneCompletion,
    train_loader: DataLoader,
    num_epochs: int = 100
):
    """Progressive training from coarse to fine resolution."""
    
    resolutions = [
        (50, 50, 8),    # Coarse: 20cm voxels
        (100, 100, 16), # Medium: 10cm voxels
        (200, 200, 32)  # Fine: 5cm voxels
    ]
    
    for resolution in resolutions:
        # Adjust model for current resolution
        model.adjust_resolution(resolution)
        
        # Train at current resolution
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs // len(resolutions)):
            for batch in train_loader:
                # Downsample ground truth to current resolution
                gt_voxels = downsample_voxels(batch['voxels'], resolution)
                
                # Forward pass
                pred_voxels = model(batch['images'], batch['camera_params'])
                
                # Compute loss
                loss = compute_loss(pred_voxels, gt_voxels)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

## Point Cloud Extraction

### 1. Voxel to Point Cloud Conversion
```python
def voxel_to_pointcloud(
    voxel_grid: torch.Tensor,  # [X, Y, Z, C]
    voxel_size: float = 0.05,
    origin: torch.Tensor = None,
    confidence_threshold: float = 0.5,
    return_semantics: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extract point cloud from completed voxel grid.
    """
    X, Y, Z, C = voxel_grid.shape
    
    # Get occupied voxels
    occupancy = voxel_grid[..., 0]  # First channel is occupancy
    occupied_mask = occupancy > confidence_threshold
    
    # Get voxel centers
    occupied_indices = torch.nonzero(occupied_mask)  # [N, 3]
    
    # Convert to world coordinates
    points = occupied_indices.float() * voxel_size
    if origin is not None:
        points += origin
    
    result = {'pos': points}
    
    if return_semantics:
        # Get semantic labels
        semantic_probs = voxel_grid[..., 1:]  # Skip occupancy channel
        semantic_probs = semantic_probs[occupied_mask]
        labels = semantic_probs.argmax(dim=-1)
        confidence = semantic_probs.max(dim=-1)[0]
        
        result['labels'] = labels
        result['confidence'] = confidence
    
    return result
```

### 2. Surface Extraction
```python
def extract_surface_from_voxels(
    voxel_grid: torch.Tensor,
    method: str = "marching_cubes"
) -> Dict[str, torch.Tensor]:
    """
    Extract surface mesh or points from voxel grid.
    """
    if method == "marching_cubes":
        from skimage import measure
        
        # Convert to numpy
        occupancy = voxel_grid[..., 0].cpu().numpy()
        
        # Extract surface mesh
        verts, faces, normals, _ = measure.marching_cubes(
            occupancy, level=0.5
        )
        
        # Sample points on surface
        surface_points = sample_points_from_mesh(verts, faces, num_points=100000)
        
        # Get semantics for surface points
        surface_semantics = interpolate_voxel_semantics(
            surface_points, voxel_grid[..., 1:]
        )
        
        return {
            'pos': torch.from_numpy(surface_points),
            'labels': surface_semantics.argmax(dim=-1),
            'normals': torch.from_numpy(normals)
        }
```

## Integration with Pylon

```python
from utils.point_cloud_ops.rendering import render_segmentation

def pylon_scene_completion(
    pc_data: Dict[str, torch.Tensor],
    images: List[torch.Tensor],
    camera_params: List[Tuple[torch.Tensor, torch.Tensor]],
    completion_model: SemanticSceneCompletion,
    voxel_size: float = 0.05
) -> Dict[str, torch.Tensor]:
    """
    Complete and segment point cloud using scene completion.
    """
    # Convert input point cloud to sparse voxels
    sparse_voxels = pointcloud_to_voxel_grid(
        pc_data['pos'],
        voxel_size=voxel_size
    )
    
    # Run scene completion
    with torch.no_grad():
        completed_voxels = completion_model(
            images,
            camera_params,
            sparse_voxels
        )
    
    # Extract completed point cloud
    completed_pc = voxel_to_pointcloud(
        completed_voxels,
        voxel_size=voxel_size,
        return_semantics=True
    )
    
    # Merge with original points (higher confidence for observed)
    merged_pc = merge_pointclouds(
        pc_data,
        completed_pc,
        observed_weight=0.8
    )
    
    return merged_pc
```

## Advantages
1. **Complete scene understanding**: Predicts semantics for occluded regions
2. **Geometric reasoning**: Learns shape priors for completion
3. **Dense predictions**: Provides labels for entire volume, not just visible points
4. **Structured output**: Voxel grid enables downstream volumetric processing

## Limitations
1. **Memory intensive**: 3D voxel grids consume significant memory
2. **Resolution trade-off**: Higher resolution requires exponentially more memory
3. **Training data requirements**: Needs complete 3D ground truth for supervision
4. **Limited range**: Typically works within bounded volumes
5. **Discretization artifacts**: Voxel representation loses fine details