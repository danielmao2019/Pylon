# Neural Radiance Field (NeRF) Based Semantic Segmentation

## Overview
NeRF-based methods build continuous neural scene representations that can be queried at any 3D location to obtain both appearance and semantic information. These methods train neural networks to represent the scene as a continuous volumetric function, enabling semantic queries at arbitrary point cloud locations.

## Core Concepts

### 1. Semantic Neural Radiance Fields
```python
class SemanticNeRF(nn.Module):
    """
    Neural radiance field with semantic output head.
    
    F: (x, y, z, θ, φ) → (RGB, σ, semantics)
    where σ is density and semantics is class distribution
    """
    
    def __init__(
        self,
        pos_encoding_dims: int = 10,
        view_encoding_dims: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 20
    ):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(3, pos_encoding_dims)
        self.view_encoder = PositionalEncoding(3, view_encoding_dims)
        
        pos_dim = 3 + 6 * pos_encoding_dims  # 3 + 2L * 3
        view_dim = 3 + 6 * view_encoding_dims
        
        # Density and feature network
        self.density_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Density output
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature output for view-dependent effects
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        
        # RGB network (view-dependent)
        self.rgb_net = nn.Sequential(
            nn.Linear(hidden_dim + view_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
        # Semantic network (view-independent)
        self.semantic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
```

### 2. Positional Encoding
```python
class PositionalEncoding(nn.Module):
    """Fourier feature mapping for high-frequency detail."""
    
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0, encoding_dim - 1, encoding_dim)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: [N, input_dim] positions or directions
            
        Returns:
            encoded: [N, input_dim + 2 * encoding_dim * input_dim]
        """
        # Original features
        encoded = [x]
        
        # Add sine and cosine features for each frequency
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq))
            encoded.append(torch.cos(x * freq))
        
        return torch.cat(encoded, dim=-1)
```

### 3. Volume Rendering with Semantics
```python
def render_semantic_rays(
    nerf_model: SemanticNeRF,
    ray_origins: torch.Tensor,      # [N, 3]
    ray_directions: torch.Tensor,   # [N, 3]
    near: float = 0.1,
    far: float = 10.0,
    num_samples: int = 64,
    use_hierarchical: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Render RGB and semantics along rays using volume rendering.
    """
    N = ray_origins.shape[0]
    
    # Sample points along rays
    t_vals = torch.linspace(near, far, num_samples, device=ray_origins.device)
    
    # Stratified sampling for training
    if nerf_model.training:
        mids = 0.5 * (t_vals[1:] + t_vals[:-1])
        upper = torch.cat([mids, t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids])
        t_vals = lower + (upper - lower) * torch.rand_like(lower)
    
    # Compute 3D points
    points = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * t_vals.unsqueeze(0).unsqueeze(-1)
    points_flat = points.reshape(-1, 3)
    dirs_flat = ray_directions.unsqueeze(1).expand_as(points).reshape(-1, 3)
    
    # Query NeRF
    with torch.cuda.amp.autocast(enabled=False):  # Full precision for stability
        # Encode positions and directions
        encoded_points = nerf_model.pos_encoder(points_flat)
        encoded_dirs = nerf_model.view_encoder(dirs_flat)
        
        # Get density and features
        hidden = nerf_model.density_net(encoded_points)
        density = nerf_model.density_head(hidden).squeeze(-1)
        features = nerf_model.feature_head(hidden)
        
        # Get RGB (view-dependent)
        rgb_input = torch.cat([features, encoded_dirs], dim=-1)
        rgb = nerf_model.rgb_net(rgb_input)
        
        # Get semantics (view-independent)
        semantics = nerf_model.semantic_net(features)
    
    # Reshape outputs
    density = density.reshape(N, num_samples)
    rgb = rgb.reshape(N, num_samples, 3)
    semantics = semantics.reshape(N, num_samples, -1)
    
    # Volume rendering
    dists = t_vals[1:] - t_vals[:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device)])
    
    # Alpha compositing
    alpha = 1.0 - torch.exp(-density * dists)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((N, 1), device=alpha.device), 1.0 - alpha], dim=1),
        dim=1
    )[:, :-1]
    
    weights = alpha * transmittance  # [N, num_samples]
    
    # Composite RGB
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
    
    # Composite semantics (weighted average of class probabilities)
    semantic_map = torch.sum(weights.unsqueeze(-1) * semantics, dim=1)
    semantic_map = F.softmax(semantic_map, dim=-1)
    
    # Depth map
    depth_map = torch.sum(weights * t_vals, dim=1)
    
    # Hierarchical sampling for fine network
    if use_hierarchical:
        # Sample more points in high-density regions
        fine_points, fine_weights = hierarchical_sampling(
            points, weights, num_samples // 2
        )
        # Query fine network and composite
        # ... (similar to above)
    
    return {
        'rgb': rgb_map,
        'semantics': semantic_map,
        'depth': depth_map,
        'weights': weights
    }
```

## Semantic-NeRF Training (CVPR 2022)

### 1. Multi-Task Loss Function
```python
class SemanticNeRFLoss(nn.Module):
    def __init__(
        self,
        rgb_weight: float = 1.0,
        semantic_weight: float = 1.0,
        depth_weight: float = 0.1,
        regularization_weight: float = 0.01
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.semantic_weight = semantic_weight
        self.depth_weight = depth_weight
        self.regularization_weight = regularization_weight
        
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_semantics: torch.Tensor,
        pred_depth: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_semantics: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        density_values: Optional[torch.Tensor] = None
    ):
        # RGB reconstruction loss
        loss_rgb = F.mse_loss(pred_rgb, gt_rgb)
        
        # Semantic segmentation loss
        loss_semantic = self.semantic_loss(
            pred_semantics,
            gt_semantics
        )
        
        # Depth supervision (if available)
        loss_depth = 0
        if gt_depth is not None:
            valid_depth = gt_depth > 0
            loss_depth = F.l1_loss(
                pred_depth[valid_depth],
                gt_depth[valid_depth]
            )
        
        # Regularization (encourage empty space to be empty)
        loss_reg = 0
        if density_values is not None:
            loss_reg = torch.mean(torch.abs(density_values))
        
        total_loss = (
            self.rgb_weight * loss_rgb +
            self.semantic_weight * loss_semantic +
            self.depth_weight * loss_depth +
            self.regularization_weight * loss_reg
        )
        
        return total_loss, {
            'rgb': loss_rgb.item(),
            'semantic': loss_semantic.item(),
            'depth': loss_depth.item() if gt_depth is not None else 0,
            'regularization': loss_reg.item() if density_values is not None else 0
        }
```

### 2. Training Pipeline
```python
def train_semantic_nerf(
    model: SemanticNeRF,
    train_dataset: Dataset,
    num_iterations: int = 200000,
    batch_size: int = 1024,  # Number of rays per batch
    learning_rate: float = 5e-4
):
    """Train Semantic-NeRF with 2D supervision."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)
    
    loss_fn = SemanticNeRFLoss()
    
    for iteration in range(num_iterations):
        # Sample random image and rays
        image_idx = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[image_idx]
        
        # Get image, semantic labels, camera parameters
        image = data['image']  # [H, W, 3]
        semantics_2d = data['semantics']  # [H, W]
        K, RT = data['intrinsics'], data['extrinsics']
        
        # Sample random rays
        ray_origins, ray_dirs, target_rgb, target_sem = sample_rays(
            image, semantics_2d, K, RT, batch_size
        )
        
        # Render rays
        outputs = render_semantic_rays(
            model, ray_origins, ray_dirs,
            near=0.1, far=10.0, num_samples=64
        )
        
        # Compute loss
        loss, loss_dict = loss_fn(
            outputs['rgb'],
            outputs['semantics'],
            outputs['depth'],
            target_rgb,
            target_sem
        )
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}")
            for k, v in loss_dict.items():
                print(f"  {k}: {v:.4f}")
```

## Panoptic Lifting (ICCV 2023)

### 1. Instance-Aware NeRF
```python
class PanopticNeRF(nn.Module):
    """NeRF with both semantic and instance segmentation."""
    
    def __init__(
        self,
        num_semantic_classes: int = 20,
        max_instances: int = 100,
        instance_feature_dim: int = 64
    ):
        super().__init__()
        
        # Base NeRF architecture
        self.base_nerf = SemanticNeRF(num_classes=num_semantic_classes)
        
        # Instance feature branch
        self.instance_feature_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, instance_feature_dim)
        )
        
        # Instance embedding lookup
        self.instance_embeddings = nn.Embedding(
            max_instances,
            instance_feature_dim
        )
        
        # Contrastive loss temperature
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, points: torch.Tensor, directions: torch.Tensor):
        # Get base features
        base_features = self.base_nerf.density_net(
            self.base_nerf.pos_encoder(points)
        )
        
        # Semantic predictions
        semantics = self.base_nerf.semantic_net(base_features)
        
        # Instance features
        instance_features = self.instance_feature_net(base_features)
        instance_features = F.normalize(instance_features, dim=-1)
        
        # RGB and density
        density = self.base_nerf.density_head(base_features)
        rgb = self.base_nerf.rgb_net(
            torch.cat([base_features, self.base_nerf.view_encoder(directions)], dim=-1)
        )
        
        return {
            'rgb': rgb,
            'density': density,
            'semantics': semantics,
            'instance_features': instance_features
        }
```

### 2. Contrastive Instance Learning
```python
def contrastive_instance_loss(
    instance_features: torch.Tensor,  # [N, D]
    instance_labels: torch.Tensor,    # [N]
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Contrastive loss for instance discrimination.
    Features from same instance should be similar.
    """
    # Normalize features
    features_norm = F.normalize(instance_features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features_norm, features_norm.T) / temperature
    
    # Create positive pair mask
    labels_expand = instance_labels.unsqueeze(1)
    mask_positive = (labels_expand == labels_expand.T).float()
    mask_positive.fill_diagonal_(0)  # Exclude self-similarity
    
    # Compute InfoNCE loss
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Mean log-likelihood for positive pairs
    mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (mask_positive.sum(dim=1) + 1e-6)
    
    loss = -mean_log_prob_pos.mean()
    
    return loss
```

## LERF - Language Embedded Radiance Fields (CVPR 2023)

### 1. CLIP Feature Integration
```python
class LERF(nn.Module):
    """Language-embedded radiance field using CLIP features."""
    
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        feature_dim: int = 512,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Load CLIP model
        import clip
        self.clip_model, self.clip_preprocess = clip.load(clip_model)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Base NeRF
        self.base_nerf = SemanticNeRF()
        
        # Language feature decoder
        self.language_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Scale learnable parameter for relevancy
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def encode_text(self, text_queries: List[str]) -> torch.Tensor:
        """Encode text queries using CLIP."""
        import clip
        
        text_tokens = clip.tokenize(text_queries).to(next(self.parameters()).device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def forward(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        text_queries: Optional[List[str]] = None
    ):
        # Standard NeRF outputs
        base_features = self.base_nerf.density_net(
            self.base_nerf.pos_encoder(points)
        )
        
        density = self.base_nerf.density_head(base_features)
        rgb = self.base_nerf.rgb_net(
            torch.cat([base_features, self.base_nerf.view_encoder(directions)], dim=-1)
        )
        
        # Language features
        language_features = self.language_decoder(base_features)
        language_features = F.normalize(language_features, dim=-1)
        
        outputs = {
            'rgb': rgb,
            'density': density,
            'language_features': language_features
        }
        
        # Compute relevancy if text queries provided
        if text_queries is not None:
            text_features = self.encode_text(text_queries)
            relevancy = torch.matmul(language_features, text_features.T) * self.scale
            outputs['relevancy'] = relevancy
        
        return outputs
```

### 2. Multi-Scale CLIP Supervision
```python
def clip_feature_loss(
    rendered_features: torch.Tensor,  # [H, W, D]
    image_patch: torch.Tensor,        # [H, W, 3]
    clip_model,
    patch_size: int = 32
) -> torch.Tensor:
    """
    Match rendered language features with CLIP image features.
    """
    H, W = image_patch.shape[:2]
    
    # Extract multi-scale patches
    losses = []
    scales = [1, 2, 4]  # Different patch scales
    
    for scale in scales:
        stride = patch_size * scale
        
        for i in range(0, H - stride + 1, stride):
            for j in range(0, W - stride + 1, stride):
                # Extract patch
                patch = image_patch[i:i+stride, j:j+stride]
                patch_features_rendered = rendered_features[i:i+stride, j:j+stride].mean(dim=(0, 1))
                
                # Get CLIP features for patch
                with torch.no_grad():
                    patch_preprocessed = clip_model.preprocess(patch).unsqueeze(0)
                    patch_features_clip = clip_model.encode_image(patch_preprocessed)
                    patch_features_clip = F.normalize(patch_features_clip, dim=-1)
                
                # Normalize rendered features
                patch_features_rendered = F.normalize(patch_features_rendered.unsqueeze(0), dim=-1)
                
                # Cosine similarity loss
                loss = 1 - F.cosine_similarity(
                    patch_features_rendered,
                    patch_features_clip
                ).mean()
                
                losses.append(loss)
    
    return torch.stack(losses).mean()
```

## Point Cloud Semantic Query

### 1. Query Points in NeRF
```python
def query_nerf_at_points(
    nerf_model: SemanticNeRF,
    point_cloud: torch.Tensor,  # [N, 3]
    chunk_size: int = 100000,
    return_confidence: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Query semantic labels at point cloud locations.
    """
    N = len(point_cloud)
    num_classes = nerf_model.semantic_net[-1].out_features
    
    all_semantics = []
    all_densities = []
    
    # Process in chunks for memory efficiency
    for i in range(0, N, chunk_size):
        chunk_points = point_cloud[i:min(i + chunk_size, N)]
        
        with torch.no_grad():
            # Encode positions
            encoded_points = nerf_model.pos_encoder(chunk_points)
            
            # Get features
            features = nerf_model.density_net(encoded_points)
            
            # Get density (for confidence)
            density = nerf_model.density_head(features).squeeze(-1)
            
            # Get semantics
            semantics = nerf_model.semantic_net(features)
            semantics = F.softmax(semantics, dim=-1)
            
            all_semantics.append(semantics)
            all_densities.append(density)
    
    semantics = torch.cat(all_semantics, dim=0)
    densities = torch.cat(all_densities, dim=0)
    
    # Get predicted labels
    labels = semantics.argmax(dim=-1)
    
    result = {'labels': labels}
    
    if return_confidence:
        # Use density as confidence (high density = high confidence)
        confidence = torch.sigmoid(densities)
        result['confidence'] = confidence
        
        # Also include semantic confidence
        semantic_confidence = semantics.max(dim=-1)[0]
        result['semantic_confidence'] = semantic_confidence
    
    return result
```

### 2. Open-Vocabulary Query
```python
def query_lerf_with_text(
    lerf_model: LERF,
    point_cloud: torch.Tensor,
    text_queries: List[str],
    threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Query points using natural language descriptions.
    
    Example queries: ["car", "tree", "building", "road"]
    """
    N = len(point_cloud)
    
    # Encode text queries
    text_features = lerf_model.encode_text(text_queries)
    
    all_relevancy = []
    
    # Process points
    chunk_size = 100000
    for i in range(0, N, chunk_size):
        chunk_points = point_cloud[i:min(i + chunk_size, N)]
        
        with torch.no_grad():
            # Dummy directions (not needed for language features)
            dummy_dirs = torch.zeros_like(chunk_points)
            
            outputs = lerf_model(chunk_points, dummy_dirs)
            language_features = outputs['language_features']
            
            # Compute relevancy scores
            relevancy = torch.matmul(language_features, text_features.T)
            relevancy = torch.sigmoid(relevancy * lerf_model.scale)
            
            all_relevancy.append(relevancy)
    
    relevancy_scores = torch.cat(all_relevancy, dim=0)  # [N, num_queries]
    
    # Get best matching query for each point
    best_scores, best_indices = relevancy_scores.max(dim=1)
    
    # Filter by threshold
    valid_mask = best_scores > threshold
    
    return {
        'labels': best_indices,
        'confidence': best_scores,
        'valid_mask': valid_mask,
        'relevancy_matrix': relevancy_scores
    }
```

## Integration with Pylon

```python
from utils.point_cloud_ops.rendering import render_rgb, render_segmentation

def pylon_nerf_segmentation(
    pc_data: Dict[str, torch.Tensor],
    images: List[torch.Tensor],
    camera_params: List[Tuple[torch.Tensor, torch.Tensor]],
    nerf_model: Optional[SemanticNeRF] = None,
    training_iterations: int = 10000
) -> Dict[str, torch.Tensor]:
    """
    Train NeRF and query semantics at point locations.
    """
    
    # Train NeRF if not provided
    if nerf_model is None:
        nerf_model = SemanticNeRF(num_classes=20)
        
        # Create training dataset from images
        train_dataset = create_nerf_dataset(images, camera_params)
        
        # Train model
        train_semantic_nerf(
            nerf_model,
            train_dataset,
            num_iterations=training_iterations
        )
    
    # Query at point cloud locations
    points = pc_data['pos']
    semantic_output = query_nerf_at_points(
        nerf_model,
        points,
        return_confidence=True
    )
    
    # Update point cloud data
    pc_data_with_semantics = pc_data.copy()
    pc_data_with_semantics['labels'] = semantic_output['labels']
    pc_data_with_semantics['label_confidence'] = semantic_output['confidence']
    
    return pc_data_with_semantics
```

## Advantages
1. **Continuous representation**: Can query semantics at any 3D location
2. **View consistency**: Learns globally consistent scene representation
3. **Implicit completion**: Can infer semantics in unobserved regions
4. **Open vocabulary**: LERF/CLIP integration enables text-based queries
5. **Multi-view fusion**: Naturally aggregates information from all views

## Limitations
1. **Training time**: Requires lengthy optimization (hours per scene)
2. **Scene-specific**: Must train separate model for each scene
3. **Memory requirements**: Stores entire scene in network weights
4. **Limited generalization**: Doesn't transfer to new scenes
5. **Resolution/accuracy trade-off**: Higher accuracy requires more network capacity