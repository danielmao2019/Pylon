# Open-Vocabulary 3D Segmentation

## Overview
Open-vocabulary 3D segmentation methods leverage vision-language models (VLMs) like CLIP, SAM, and DINO to enable segmentation with arbitrary text queries rather than fixed class sets. These approaches distill 2D open-vocabulary features into 3D representations, enabling zero-shot segmentation of novel categories.

## Core Concepts

### 1. Vision-Language Feature Alignment
```python
class VisionLanguageAlignment:
    """
    Core principle: Align 3D features with CLIP's vision-language space
    to enable text-based querying.
    """
    
    def __init__(self, clip_model: str = "ViT-L/14@336px"):
        import clip
        self.clip_model, self.preprocess = clip.load(clip_model)
        self.clip_model.eval()
        
        # Feature dimensions
        self.clip_dim = 768  # ViT-L/14
        self.projection_dim = 512
        
        # Learnable projection from 3D to CLIP space
        self.feature_projector = nn.Linear(self.projection_dim, self.clip_dim)
    
    def encode_text(self, text_queries: List[str]) -> torch.Tensor:
        """Encode text descriptions into CLIP space."""
        import clip
        tokens = clip.tokenize(text_queries).cuda()
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def compute_similarity(
        self,
        point_features: torch.Tensor,  # [N, D]
        text_features: torch.Tensor,   # [C, D]
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute similarity between point features and text embeddings."""
        # Normalize features
        point_features = F.normalize(point_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(point_features, text_features.T) / temperature
        
        return similarity  # [N, C]
```

## OpenScene (CVPR 2023)

### 1. Multi-View Feature Distillation
```python
class OpenScene(nn.Module):
    """
    Distill 2D open-vocabulary features into 3D point features.
    """
    
    def __init__(
        self,
        point_encoder: str = "minkowskinet",
        feature_dim: int = 512,
        voxel_size: float = 0.05
    ):
        super().__init__()
        
        # 3D backbone (sparse convolution)
        if point_encoder == "minkowskinet":
            self.point_encoder = MinkowskiUNet(
                in_channels=3,
                out_channels=feature_dim,
                D=3  # 3D sparse convolution
            )
        
        # 2D feature extractors
        self.clip_model = CLIPModel()
        self.lseg_model = LSeg()  # Open-vocabulary 2D segmentation
        self.openseg_model = OpenSeg()  # Alternative 2D model
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Temperature for distillation
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
```

### 2. 2D-3D Feature Distillation Pipeline
```python
def distill_2d_features_to_3d(
    point_cloud: torch.Tensor,          # [N, 3]
    point_colors: Optional[torch.Tensor],  # [N, 3]
    images: List[torch.Tensor],         # Multi-view images
    camera_params: List[Tuple],         # Camera parameters
    feature_extractor_2d,                # 2D open-vocabulary model
    feature_dim: int = 512
) -> torch.Tensor:
    """
    Distill 2D open-vocabulary features to 3D points.
    """
    N = len(point_cloud)
    aggregated_features = torch.zeros(N, feature_dim).cuda()
    aggregation_weights = torch.zeros(N).cuda()
    
    for img, (K, RT) in zip(images, camera_params):
        # Extract 2D features
        with torch.no_grad():
            # Get pixel-wise features
            if isinstance(feature_extractor_2d, LSeg):
                features_2d = feature_extractor_2d.forward_features(img.unsqueeze(0))
                # features_2d: [1, H, W, D]
            elif isinstance(feature_extractor_2d, CLIPModel):
                features_2d = extract_clip_features(img, feature_extractor_2d)
            else:
                features_2d = feature_extractor_2d(img.unsqueeze(0))
            
            features_2d = features_2d.squeeze(0)  # [H, W, D]
        
        # Project points to image
        points_2d, depths, visibility = project_points_to_image(
            point_cloud, K, RT, img.shape[:2]
        )
        
        # Sample features for visible points
        visible_indices = torch.where(visibility)[0]
        visible_points_2d = points_2d[visible_indices]
        
        # Bilinear interpolation for sub-pixel accuracy
        sampled_features = bilinear_sample_features(
            features_2d, visible_points_2d
        )
        
        # Weight by visibility and viewing angle
        weights = compute_visibility_weights(
            point_cloud[visible_indices],
            camera_position=RT[:3, 3],
            point_normals=None  # Can add if available
        )
        
        # Accumulate weighted features
        aggregated_features[visible_indices] += sampled_features * weights.unsqueeze(-1)
        aggregation_weights[visible_indices] += weights
    
    # Normalize by total weights
    valid_mask = aggregation_weights > 0
    aggregated_features[valid_mask] /= aggregation_weights[valid_mask].unsqueeze(-1)
    
    return aggregated_features, valid_mask
```

### 3. Multi-Model Ensemble
```python
class MultiModelDistillation:
    """Ensemble multiple 2D open-vocabulary models for robustness."""
    
    def __init__(self):
        self.models = {
            'lseg': LSeg(),
            'openseg': OpenSeg(),
            'clip': CLIPModel(),
            'dino': DINOv2()
        }
        
        # Model-specific projection layers
        self.projectors = nn.ModuleDict({
            name: nn.Linear(model.feature_dim, 512)
            for name, model in self.models.items()
        })
    
    def extract_ensemble_features(
        self,
        image: torch.Tensor,
        return_separate: bool = False
    ) -> torch.Tensor:
        """Extract and combine features from multiple models."""
        
        all_features = {}
        
        for name, model in self.models.items():
            with torch.no_grad():
                if name == 'clip':
                    features = self.extract_clip_patch_features(image, model)
                elif name == 'dino':
                    features = model.get_intermediate_layers(image)[0]
                else:
                    features = model.forward_features(image)
            
            # Project to common dimension
            features = self.projectors[name](features)
            all_features[name] = features
        
        if return_separate:
            return all_features
        
        # Weighted combination (can learn weights)
        weights = {'lseg': 0.3, 'openseg': 0.3, 'clip': 0.25, 'dino': 0.15}
        combined = sum(all_features[k] * weights[k] for k in all_features)
        
        return F.normalize(combined, dim=-1)
```

## OpenMask3D (2023)

### 1. SAM-Based Instance Segmentation
```python
class OpenMask3D(nn.Module):
    """
    Combine SAM (Segment Anything) with CLIP for open-vocabulary
    instance segmentation in 3D.
    """
    
    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h.pth",
        clip_model: str = "ViT-L/14"
    ):
        super().__init__()
        
        # Load SAM for instance mask generation
        from segment_anything import sam_model_registry, SamPredictor
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam)
        
        # Load CLIP for semantic understanding
        import clip
        self.clip_model, self.clip_preprocess = clip.load(clip_model)
        
        # 3D instance aggregation
        self.instance_aggregator = InstanceAggregator3D()
```

### 2. Instance Mask Projection
```python
class InstanceAggregator3D:
    """Aggregate 2D instance masks into 3D instance segmentation."""
    
    def __init__(self, min_points: int = 50, iou_threshold: float = 0.5):
        self.min_points = min_points
        self.iou_threshold = iou_threshold
    
    def aggregate_multi_view_instances(
        self,
        point_cloud: torch.Tensor,           # [N, 3]
        instance_masks_2d: List[List[torch.Tensor]],  # Per-view instance masks
        camera_params: List[Tuple],
        semantic_features: torch.Tensor      # [N, D] point features
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate 2D instance masks into 3D instances.
        """
        N = len(point_cloud)
        
        # Collect all 2D-3D associations
        point_to_instances = defaultdict(list)  # point_idx -> [(view_idx, instance_idx)]
        instance_features = {}  # (view_idx, instance_idx) -> feature
        
        for view_idx, (masks, (K, RT)) in enumerate(zip(instance_masks_2d, camera_params)):
            # Project points to this view
            points_2d, _, visibility = project_points_to_image(
                point_cloud, K, RT, masks[0].shape
            )
            
            visible_indices = torch.where(visibility)[0]
            visible_points_2d = points_2d[visible_indices].long()
            
            # Check which instance each visible point belongs to
            for instance_idx, mask in enumerate(masks):
                # Points falling in this mask
                in_mask = mask[visible_points_2d[:, 1], visible_points_2d[:, 0]]
                point_indices = visible_indices[in_mask]
                
                for pid in point_indices:
                    point_to_instances[pid.item()].append((view_idx, instance_idx))
                
                # Store instance feature (can be CLIP feature of cropped region)
                instance_features[(view_idx, instance_idx)] = \
                    self.extract_instance_feature(mask, semantic_features[point_indices])
        
        # Merge instances across views using graph clustering
        instance_graph = self.build_instance_graph(
            point_to_instances,
            instance_features
        )
        
        merged_instances = self.merge_instances(
            instance_graph,
            point_to_instances,
            N
        )
        
        return merged_instances
    
    def build_instance_graph(
        self,
        point_to_instances: Dict,
        instance_features: Dict
    ) -> nx.Graph:
        """Build graph connecting similar instances across views."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes (instances)
        for instance_id in instance_features.keys():
            G.add_node(instance_id, feature=instance_features[instance_id])
        
        # Add edges between instances sharing points
        instance_overlap = defaultdict(int)
        
        for point_instances in point_to_instances.values():
            # Count overlaps between instance pairs
            for i, inst1 in enumerate(point_instances):
                for inst2 in point_instances[i+1:]:
                    instance_overlap[(inst1, inst2)] += 1
        
        # Add edges for significant overlaps
        for (inst1, inst2), overlap in instance_overlap.items():
            # Compute IoU
            points1 = {p for p, insts in point_to_instances.items() if inst1 in insts}
            points2 = {p for p, insts in point_to_instances.items() if inst2 in insts}
            
            intersection = len(points1 & points2)
            union = len(points1 | points2)
            iou = intersection / (union + 1e-6)
            
            if iou > self.iou_threshold:
                G.add_edge(inst1, inst2, weight=iou)
        
        return G
    
    def merge_instances(
        self,
        instance_graph: nx.Graph,
        point_to_instances: Dict,
        num_points: int
    ) -> Dict[str, torch.Tensor]:
        """Merge instances using connected components."""
        import networkx as nx
        
        # Find connected components (merged instances)
        components = list(nx.connected_components(instance_graph))
        
        # Create instance segmentation
        instance_labels = torch.full((num_points,), -1, dtype=torch.long)
        instance_features = []
        
        for comp_idx, component in enumerate(components):
            # Get all points belonging to this merged instance
            instance_points = set()
            component_features = []
            
            for instance_id in component:
                points = {p for p, insts in point_to_instances.items() 
                         if instance_id in insts}
                instance_points.update(points)
                component_features.append(instance_graph.nodes[instance_id]['feature'])
            
            # Skip small instances
            if len(instance_points) < self.min_points:
                continue
            
            # Assign instance label
            for point_idx in instance_points:
                instance_labels[point_idx] = comp_idx
            
            # Average features for merged instance
            avg_feature = torch.stack(component_features).mean(dim=0)
            instance_features.append(avg_feature)
        
        return {
            'instance_labels': instance_labels,
            'instance_features': torch.stack(instance_features) if instance_features else torch.empty(0, 512),
            'num_instances': len(instance_features)
        }
```

### 3. Open-Vocabulary Instance Classification
```python
def classify_instances_with_clip(
    instance_features: torch.Tensor,    # [K, D] instance features
    text_queries: List[str],            # ["chair", "table", "car", ...]
    clip_model,
    confidence_threshold: float = 0.3
) -> Dict[str, torch.Tensor]:
    """
    Classify 3D instances using CLIP text queries.
    """
    # Encode text queries
    import clip
    text_tokens = clip.tokenize(text_queries).cuda()
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    # Normalize instance features
    instance_features = F.normalize(instance_features, dim=-1)
    
    # Compute similarities
    similarities = torch.matmul(instance_features, text_features.T)
    
    # Get predictions
    max_similarities, predictions = similarities.max(dim=1)
    
    # Filter low confidence
    valid_mask = max_similarities > confidence_threshold
    predictions[~valid_mask] = -1  # Unknown class
    
    return {
        'semantic_labels': predictions,
        'confidence': max_similarities,
        'similarity_matrix': similarities,
        'valid_mask': valid_mask
    }
```

## ConceptFusion (ICCV 2023)

### 1. Pixel-Aligned Feature Fusion
```python
class ConceptFusion(nn.Module):
    """
    Fuse open-vocabulary features from multiple foundation models
    into a unified 3D representation.
    """
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        
        # Foundation models
        self.models = {
            'clip': CLIPModel(),
            'dino': DINOv2(),
            'sam': SAMEncoder(),
            'lseg': LSeg()
        }
        
        # Per-pixel feature extraction
        self.pixel_feature_dim = feature_dim
        
        # Feature fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * len(self.models), feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def extract_pixel_aligned_features(
        self,
        image: torch.Tensor,
        compute_gradcam: bool = False
    ) -> torch.Tensor:
        """
        Extract spatially-aligned features from all models.
        """
        H, W = image.shape[-2:]
        all_features = []
        
        for name, model in self.models.items():
            if name == 'clip':
                # Use attention maps for spatial features
                features = self.extract_clip_spatial_features(image, model)
            elif name == 'dino':
                # Use patch tokens
                features = model.get_intermediate_layers(image, reshape=True)[0]
            elif name == 'sam':
                # Use encoder features
                features = model.image_encoder(image)
            else:
                features = model.forward_features(image)
            
            # Resize to common resolution
            features = F.interpolate(
                features.permute(0, 3, 1, 2),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            # Project to common dimension
            features = self.projectors[name](features)
            all_features.append(features)
        
        # Concatenate and fuse
        concatenated = torch.cat(all_features, dim=-1)
        fused = self.fusion_net(concatenated)
        
        return fused  # [B, H, W, D]
```

### 2. GradCAM-Based Relevancy
```python
def compute_gradcam_relevancy(
    model,
    image: torch.Tensor,
    text_query: str,
    layer_name: str = "layer4"
) -> torch.Tensor:
    """
    Compute GradCAM heatmap for text query relevancy.
    """
    # Hook to capture activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    target_layer = dict(model.named_modules())[layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    image_features = model.encode_image(image)
    text_features = model.encode_text(clip.tokenize([text_query]))
    
    # Compute similarity
    similarity = F.cosine_similarity(image_features, text_features)
    
    # Backward pass
    model.zero_grad()
    similarity.backward()
    
    # Compute GradCAM
    pooled_gradients = gradients[0].mean(dim=[2, 3], keepdim=True)
    weighted_activations = (activations[0] * pooled_gradients).sum(dim=1, keepdim=True)
    gradcam = F.relu(weighted_activations)
    
    # Normalize
    gradcam = gradcam / (gradcam.max() + 1e-8)
    
    # Resize to image size
    gradcam = F.interpolate(
        gradcam,
        size=image.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    
    # Clean up hooks
    forward_handle.remove()
    backward_handle.remove()
    
    return gradcam.squeeze()
```

## Training Strategies

### 1. Contrastive Learning on 3D Features
```python
class ContrastiveLearning3D:
    """Train 3D features to align with CLIP space."""
    
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(
        self,
        point_features: torch.Tensor,   # [N, D]
        point_labels: torch.Tensor,     # [N] indices
        text_descriptions: List[str],   # Class descriptions
        clip_model
    ) -> torch.Tensor:
        """
        Contrastive loss between 3D features and text embeddings.
        """
        # Encode text
        text_features = clip_model.encode_text(
            clip.tokenize(text_descriptions).cuda()
        )
        text_features = F.normalize(text_features, dim=-1)
        
        # Normalize point features
        point_features = F.normalize(point_features, dim=-1)
        
        # Compute logits
        logits = torch.matmul(point_features, text_features.T) / self.temperature
        
        # Cross-entropy loss
        loss = self.criterion(logits, point_labels)
        
        return loss
```

### 2. Knowledge Distillation Pipeline
```python
def train_3d_with_2d_distillation(
    point_cloud_model: nn.Module,
    train_dataset: Dataset,
    teacher_models_2d: Dict[str, nn.Module],
    num_epochs: int = 50
):
    """
    Train 3D model using 2D teacher models.
    """
    optimizer = torch.optim.AdamW(
        point_cloud_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    for epoch in range(num_epochs):
        for batch in train_dataset:
            point_cloud = batch['points']
            images = batch['images']
            camera_params = batch['camera_params']
            
            # Get 3D predictions
            point_features_3d = point_cloud_model(point_cloud)
            
            # Get 2D teacher features
            teacher_features = []
            for img, (K, RT) in zip(images, camera_params):
                # Extract features from each teacher
                img_features = {}
                for name, teacher in teacher_models_2d.items():
                    with torch.no_grad():
                        img_features[name] = teacher(img)
                
                # Project and sample at point locations
                projected_features = project_and_sample_features(
                    point_cloud, img_features, K, RT
                )
                teacher_features.append(projected_features)
            
            # Aggregate teacher features
            aggregated_teacher = aggregate_multiview_features(teacher_features)
            
            # Distillation loss
            loss = F.mse_loss(point_features_3d, aggregated_teacher)
            
            # Add contrastive loss if labels available
            if 'text_labels' in batch:
                contrastive_loss = compute_contrastive_loss(
                    point_features_3d,
                    batch['text_labels']
                )
                loss += 0.5 * contrastive_loss
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Integration with Pylon

```python
from utils.point_cloud_ops.rendering import render_rgb

def pylon_open_vocabulary_segmentation(
    pc_data: Dict[str, torch.Tensor],
    images: List[torch.Tensor],
    camera_params: List[Tuple[torch.Tensor, torch.Tensor]],
    text_queries: List[str],
    method: str = "openscene"  # or "openmask3d", "conceptfusion"
) -> Dict[str, torch.Tensor]:
    """
    Perform open-vocabulary segmentation on point cloud.
    """
    
    if method == "openscene":
        # Initialize OpenScene
        model = OpenScene()
        
        # Distill 2D features to 3D
        point_features, valid_mask = distill_2d_features_to_3d(
            pc_data['pos'],
            pc_data.get('rgb'),
            images,
            camera_params,
            model.lseg_model
        )
        
        # Encode text queries
        text_features = model.clip_model.encode_text(text_queries)
        
        # Compute similarities
        similarities = compute_similarity(
            point_features[valid_mask],
            text_features
        )
        
        # Get predictions
        labels = torch.full((len(pc_data['pos']),), -1, dtype=torch.long)
        labels[valid_mask] = similarities.argmax(dim=1)
        
    elif method == "openmask3d":
        # Use SAM for instance segmentation
        model = OpenMask3D()
        
        # Generate instance masks for each view
        instance_masks_2d = []
        for img in images:
            masks = model.sam_predictor.generate_masks(img)
            instance_masks_2d.append(masks)
        
        # Aggregate to 3D instances
        instance_result = model.instance_aggregator.aggregate_multi_view_instances(
            pc_data['pos'],
            instance_masks_2d,
            camera_params,
            point_features
        )
        
        # Classify instances with text
        instance_classes = classify_instances_with_clip(
            instance_result['instance_features'],
            text_queries,
            model.clip_model
        )
        
        # Map instance labels to points
        labels = map_instance_to_semantic_labels(
            instance_result['instance_labels'],
            instance_classes['semantic_labels']
        )
    
    elif method == "conceptfusion":
        # Use ConceptFusion
        model = ConceptFusion()
        
        # Extract pixel-aligned features
        all_features = []
        for img in images:
            features = model.extract_pixel_aligned_features(img)
            all_features.append(features)
        
        # Project to 3D and fuse
        point_features = fuse_multiview_concept_features(
            pc_data['pos'],
            all_features,
            camera_params
        )
        
        # Query with text
        text_features = encode_text_with_ensemble(
            text_queries,
            model.models
        )
        
        similarities = torch.matmul(point_features, text_features.T)
        labels = similarities.argmax(dim=1)
    
    # Update point cloud data
    pc_data_segmented = pc_data.copy()
    pc_data_segmented['labels'] = labels
    pc_data_segmented['label_names'] = text_queries
    
    return pc_data_segmented
```

## Advantages
1. **No fixed classes**: Can segment any category described by text
2. **Zero-shot capability**: No 3D training needed for new categories
3. **Leverages 2D models**: Benefits from large-scale 2D pretraining
4. **Semantic understanding**: Natural language interface for queries
5. **Compositional**: Can handle complex descriptions ("red car", "wooden chair")

## Limitations
1. **2D model dependency**: Quality limited by 2D feature extractors
2. **View consistency**: Features may vary across viewpoints
3. **Computational cost**: Multiple model inference per view
4. **Ambiguity**: Text descriptions may be ambiguous
5. **Feature alignment**: Challenging to align features from different models