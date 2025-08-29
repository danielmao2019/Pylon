# 2D-3D Consistency Optimization

## Overview
Consistency optimization methods formulate 3D segmentation as an optimization problem that enforces agreement between 3D labels and their 2D projections across multiple views. These approaches iteratively refine 3D labels to maximize consistency with 2D predictions while maintaining spatial smoothness in 3D.

## Core Concepts

### 1. Problem Formulation
```python
class ConsistencyOptimization:
    """
    Optimize 3D labels to be consistent with 2D observations.
    
    min_L E(L) = E_data(L) + λ_smooth * E_smooth(L) + λ_prior * E_prior(L)
    
    where:
    - L: 3D point labels
    - E_data: 2D projection consistency
    - E_smooth: 3D spatial smoothness
    - E_prior: Semantic priors
    """
    
    def __init__(
        self,
        num_classes: int,
        lambda_smooth: float = 0.1,
        lambda_prior: float = 0.05,
        optimization_method: str = "graph_cut"  # or "mean_field", "gradient_descent"
    ):
        self.num_classes = num_classes
        self.lambda_smooth = lambda_smooth
        self.lambda_prior = lambda_prior
        self.optimization_method = optimization_method
```

### 2. Energy Functions
```python
class EnergyFunctions:
    """Define energy terms for optimization."""
    
    @staticmethod
    def data_term(
        point_labels: torch.Tensor,         # [N]
        point_cloud: torch.Tensor,          # [N, 3]
        predictions_2d: List[torch.Tensor], # 2D segmentation maps
        camera_params: List[Tuple],
        confidence_2d: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Data term: Consistency with 2D predictions.
        """
        total_energy = 0.0
        
        for pred_2d, (K, RT) in zip(predictions_2d, camera_params):
            # Project points to this view
            points_2d, depths, visibility = project_points_to_image(
                point_cloud, K, RT, pred_2d.shape[:2]
            )
            
            # Sample 2D predictions at projected locations
            visible_indices = torch.where(visibility)[0]
            visible_points_2d = points_2d[visible_indices].long()
            
            # Get 2D predictions for visible points
            pred_at_points = pred_2d[
                visible_points_2d[:, 1],
                visible_points_2d[:, 0]
            ]
            
            # Compute disagreement
            labels_visible = point_labels[visible_indices]
            disagreement = (labels_visible != pred_at_points).float()
            
            # Weight by confidence if available
            if confidence_2d is not None:
                conf_at_points = confidence_2d[pred_2d.shape[0]][
                    visible_points_2d[:, 1],
                    visible_points_2d[:, 0]
                ]
                disagreement *= conf_at_points
            
            total_energy += disagreement.sum()
        
        return total_energy
    
    @staticmethod
    def smoothness_term(
        point_labels: torch.Tensor,    # [N]
        point_cloud: torch.Tensor,     # [N, 3]
        point_colors: Optional[torch.Tensor] = None,  # [N, 3]
        k_neighbors: int = 10
    ) -> torch.Tensor:
        """
        Smoothness term: Spatial coherence in 3D.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Build KNN graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
        nbrs.fit(point_cloud.cpu().numpy())
        distances, indices = nbrs.kneighbors(point_cloud.cpu().numpy())
        
        total_energy = 0.0
        
        for i in range(len(point_cloud)):
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_dists = distances[i, 1:]
            
            # Compute label disagreement with neighbors
            label_diff = (point_labels[i] != point_labels[neighbors]).float()
            
            # Weight by distance (closer neighbors have more influence)
            weights = torch.exp(-neighbor_dists / neighbor_dists.mean())
            
            # Optional: Consider color similarity
            if point_colors is not None:
                color_diff = torch.norm(
                    point_colors[i] - point_colors[neighbors],
                    dim=1
                )
                color_weights = torch.exp(-color_diff / 0.1)
                weights *= color_weights
            
            total_energy += (label_diff * weights).sum()
        
        return total_energy
    
    @staticmethod
    def prior_term(
        point_labels: torch.Tensor,
        class_priors: torch.Tensor  # [num_classes] prior probabilities
    ) -> torch.Tensor:
        """
        Prior term: Enforce expected class distributions.
        """
        # Compute empirical distribution
        label_counts = torch.bincount(point_labels, minlength=len(class_priors))
        empirical_dist = label_counts.float() / len(point_labels)
        
        # KL divergence from prior
        kl_div = torch.sum(
            empirical_dist * torch.log(empirical_dist / (class_priors + 1e-10) + 1e-10)
        )
        
        return kl_div
```

## Graph-Cut Optimization

### 1. Graph Construction
```python
class GraphCutOptimization:
    """Optimize using graph cuts (multi-label)."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        
    def build_graph(
        self,
        point_cloud: torch.Tensor,
        unary_costs: torch.Tensor,      # [N, C] cost for each label
        k_neighbors: int = 10,
        sigma_spatial: float = 0.1,
        sigma_color: float = 0.1
    ):
        """
        Build graph for optimization.
        """
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors
        
        N = len(point_cloud)
        G = nx.Graph()
        
        # Add nodes with unary costs
        for i in range(N):
            G.add_node(i, costs=unary_costs[i])
        
        # Build spatial neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs.fit(point_cloud.cpu().numpy())
        distances, indices = nbrs.kneighbors()
        
        # Add edges with pairwise costs
        for i in range(N):
            for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                if not G.has_edge(i, j):
                    # Compute edge weight (Potts model)
                    weight = np.exp(-dist**2 / (2 * sigma_spatial**2))
                    G.add_edge(i, j, weight=weight)
        
        return G
    
    def solve_alpha_expansion(
        self,
        graph: nx.Graph,
        initial_labels: torch.Tensor,
        max_iterations: int = 10
    ) -> torch.Tensor:
        """
        Alpha-expansion algorithm for multi-label optimization.
        """
        import maxflow
        
        current_labels = initial_labels.clone()
        N = len(current_labels)
        
        for iteration in range(max_iterations):
            label_changed = False
            
            # Try each label as alpha
            for alpha in range(self.num_classes):
                # Build binary graph for alpha-expansion
                g = maxflow.Graph[float]()
                nodes = g.add_nodes(N)
                
                # Add unary terms
                for i in range(N):
                    # Cost of keeping current label
                    cost_current = graph.nodes[i]['costs'][current_labels[i]]
                    # Cost of switching to alpha
                    cost_alpha = graph.nodes[i]['costs'][alpha]
                    
                    if current_labels[i] == alpha:
                        # Already has label alpha
                        g.add_tedge(nodes[i], 0, float('inf'))
                    else:
                        g.add_tedge(nodes[i], cost_alpha, cost_current)
                
                # Add pairwise terms
                for i, j, data in graph.edges(data=True):
                    weight = data['weight']
                    
                    if current_labels[i] == current_labels[j]:
                        # Same label - no cost
                        continue
                    elif current_labels[i] == alpha or current_labels[j] == alpha:
                        # One has alpha - add directed edge
                        if current_labels[i] == alpha:
                            g.add_edge(nodes[j], nodes[i], weight, 0)
                        else:
                            g.add_edge(nodes[i], nodes[j], weight, 0)
                    else:
                        # Both have different non-alpha labels
                        # Add auxiliary node
                        aux = g.add_node()
                        g.add_edge(nodes[i], aux, weight, 0)
                        g.add_edge(aux, nodes[j], weight, 0)
                        g.add_tedge(aux, 0, weight)
                
                # Solve max-flow/min-cut
                flow = g.maxflow()
                
                # Update labels based on cut
                for i in range(N):
                    if g.get_segment(nodes[i]) == 0:  # Source side
                        if current_labels[i] != alpha:
                            current_labels[i] = alpha
                            label_changed = True
            
            if not label_changed:
                break
        
        return current_labels
```

### 2. Unary Cost Computation
```python
def compute_unary_costs(
    point_cloud: torch.Tensor,
    predictions_2d: List[torch.Tensor],  # [H, W, C] class probabilities
    camera_params: List[Tuple],
    num_classes: int
) -> torch.Tensor:
    """
    Compute unary costs from 2D predictions.
    
    Returns:
        costs: [N, C] cost for assigning each label to each point
    """
    N = len(point_cloud)
    costs = torch.ones(N, num_classes) * 0.5  # Initialize with uniform cost
    view_counts = torch.zeros(N)
    
    for pred_2d, (K, RT) in zip(predictions_2d, camera_params):
        # Project points
        points_2d, depths, visibility = project_points_to_image(
            point_cloud, K, RT, pred_2d.shape[:2]
        )
        
        visible_indices = torch.where(visibility)[0]
        visible_points_2d = points_2d[visible_indices]
        
        # Sample probabilities using bilinear interpolation
        for idx, pt_2d in zip(visible_indices, visible_points_2d):
            # Bilinear sampling
            probs = bilinear_sample(pred_2d, pt_2d.unsqueeze(0)).squeeze()
            
            # Convert probabilities to costs (negative log-likelihood)
            point_costs = -torch.log(probs + 1e-10)
            
            # Accumulate costs
            costs[idx] += point_costs
            view_counts[idx] += 1
    
    # Average costs over views
    valid_mask = view_counts > 0
    costs[valid_mask] /= view_counts[valid_mask].unsqueeze(1)
    
    return costs
```

## Mean Field Inference

### 1. Conditional Random Field
```python
class DenseCRF3D:
    """Dense CRF for 3D point cloud segmentation."""
    
    def __init__(
        self,
        num_classes: int,
        theta_alpha: float = 10.0,    # Spatial kernel
        theta_beta: float = 0.1,      # Color kernel
        theta_gamma: float = 3.0,     # Smoothness
        num_iterations: int = 10
    ):
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
    
    def mean_field_inference(
        self,
        unary: torch.Tensor,           # [N, C] unary potentials
        point_cloud: torch.Tensor,     # [N, 3] positions
        point_colors: torch.Tensor,    # [N, 3] colors
        point_normals: Optional[torch.Tensor] = None  # [N, 3] normals
    ) -> torch.Tensor:
        """
        Mean field inference for dense CRF.
        """
        N, C = unary.shape
        
        # Initialize Q (beliefs)
        Q = F.softmax(-unary, dim=1)  # [N, C]
        
        # Precompute pairwise kernels
        kernels = self._compute_kernels(point_cloud, point_colors, point_normals)
        
        for iteration in range(self.num_iterations):
            # Message passing
            messages = torch.zeros_like(Q)
            
            for kernel, weight in kernels:
                # Compute messages for this kernel
                # This is essentially a weighted sum of neighbors' beliefs
                kernel_messages = self._compute_messages(Q, kernel)
                messages += weight * kernel_messages
            
            # Update beliefs
            Q_unnorm = torch.exp(-unary - messages)
            Q = Q_unnorm / Q_unnorm.sum(dim=1, keepdim=True)
        
        return Q
    
    def _compute_kernels(
        self,
        positions: torch.Tensor,
        colors: torch.Tensor,
        normals: Optional[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Compute Gaussian kernels for pairwise potentials.
        """
        N = len(positions)
        kernels = []
        
        # Spatial kernel (appearance kernel)
        spatial_dists = torch.cdist(positions, positions)
        color_dists = torch.cdist(colors, colors)
        
        kernel_appearance = torch.exp(
            -spatial_dists**2 / (2 * self.theta_alpha**2) -
            color_dists**2 / (2 * self.theta_beta**2)
        )
        kernels.append((kernel_appearance, 1.0))
        
        # Smoothness kernel (spatial only)
        kernel_smooth = torch.exp(
            -spatial_dists**2 / (2 * self.theta_gamma**2)
        )
        kernels.append((kernel_smooth, 0.5))
        
        # Normal kernel (if available)
        if normals is not None:
            normal_similarity = torch.matmul(normals, normals.T)
            kernel_normal = (normal_similarity + 1) / 2  # Map [-1, 1] to [0, 1]
            kernels.append((kernel_normal, 0.3))
        
        return kernels
    
    def _compute_messages(
        self,
        Q: torch.Tensor,        # [N, C] current beliefs
        kernel: torch.Tensor    # [N, N] kernel matrix
    ) -> torch.Tensor:
        """
        Compute messages using kernel.
        """
        # Normalize kernel (each row sums to 1)
        kernel_norm = kernel / kernel.sum(dim=1, keepdim=True)
        
        # Compute messages (weighted sum of neighbors' beliefs)
        messages = torch.matmul(kernel_norm, Q)
        
        return messages
```

### 2. Efficient Implementation with Permutohedral Lattice
```python
class PermutohedralLattice:
    """
    Efficient high-dimensional filtering using permutohedral lattice.
    Enables fast CRF inference for large point clouds.
    """
    
    def __init__(self, d: int, sigma: float = 1.0):
        """
        d: Dimensionality of feature space
        sigma: Standard deviation for Gaussian filter
        """
        self.d = d
        self.sigma = sigma
        self.scale = np.sqrt(2 / 3) * self.d * sigma
        
    def filter(
        self,
        positions: np.ndarray,  # [N, d] positions in feature space
        values: np.ndarray       # [N, c] values to filter
    ) -> np.ndarray:
        """
        Apply Gaussian filter in high-dimensional space.
        """
        N, c = values.shape
        
        # Scale positions
        scaled_pos = positions / self.scale
        
        # Embed into permutohedral lattice
        lattice_coords, barycentric = self._embed(scaled_pos)
        
        # Splat values onto lattice vertices
        lattice_values = self._splat(lattice_coords, barycentric, values)
        
        # Blur in lattice space
        blurred = self._blur(lattice_values)
        
        # Slice back to get filtered values
        filtered = self._slice(lattice_coords, barycentric, blurred)
        
        return filtered
    
    def _embed(self, positions):
        """Embed positions into permutohedral lattice."""
        # Implementation details omitted for brevity
        # See "Fast High-Dimensional Filtering Using the Permutohedral Lattice"
        pass
```

## Gradient-Based Optimization

### 1. Differentiable Rendering
```python
class DifferentiableOptimization(nn.Module):
    """
    Optimize 3D labels using gradient descent with differentiable rendering.
    """
    
    def __init__(
        self,
        num_points: int,
        num_classes: int,
        learning_rate: float = 0.01
    ):
        super().__init__()
        
        # Learnable label logits
        self.label_logits = nn.Parameter(
            torch.randn(num_points, num_classes)
        )
        
        # Temperature for soft labels
        self.temperature = 0.1
    
    def forward(self) -> torch.Tensor:
        """Get soft labels."""
        return F.softmax(self.label_logits / self.temperature, dim=1)
    
    def project_and_render(
        self,
        point_cloud: torch.Tensor,
        soft_labels: torch.Tensor,
        camera_K: torch.Tensor,
        camera_RT: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Differentiable projection and rendering.
        """
        # Project points
        points_cam = transform_points(point_cloud, camera_RT)
        points_2d = project_points(points_cam, camera_K)
        
        # Normalize to [-1, 1] for grid
        points_2d_norm = normalize_coords(points_2d, image_size)
        
        # Create sparse image using scatter operations
        H, W = image_size
        rendered = torch.zeros(self.num_classes, H, W).cuda()
        
        # Convert normalized coords back to pixel indices
        pixel_x = ((points_2d_norm[:, 0] + 1) * W / 2).long()
        pixel_y = ((points_2d_norm[:, 1] + 1) * H / 2).long()
        
        # Scatter soft labels to image
        valid_mask = (
            (pixel_x >= 0) & (pixel_x < W) &
            (pixel_y >= 0) & (pixel_y < H)
        )
        
        for c in range(self.num_classes):
            rendered[c].index_put_(
                (pixel_y[valid_mask], pixel_x[valid_mask]),
                soft_labels[valid_mask, c],
                accumulate=True
            )
        
        # Normalize by counts
        counts = torch.zeros(H, W).cuda()
        counts.index_put_(
            (pixel_y[valid_mask], pixel_x[valid_mask]),
            torch.ones(valid_mask.sum()).cuda(),
            accumulate=True
        )
        
        rendered = rendered / (counts.unsqueeze(0) + 1e-10)
        
        return rendered.permute(1, 2, 0)  # [H, W, C]
```

### 2. Optimization Loop
```python
def optimize_labels_gradient(
    model: DifferentiableOptimization,
    point_cloud: torch.Tensor,
    predictions_2d: List[torch.Tensor],
    camera_params: List[Tuple],
    num_iterations: int = 100,
    lambda_smooth: float = 0.1,
    lambda_entropy: float = 0.01
):
    """
    Optimize labels using gradient descent.
    """
    optimizer = torch.optim.Adam([model.label_logits], lr=0.01)
    
    for iteration in range(num_iterations):
        # Get current soft labels
        soft_labels = model()
        
        # Data term: consistency with 2D
        loss_data = 0
        for pred_2d, (K, RT) in zip(predictions_2d, camera_params):
            # Render current labels
            rendered = model.project_and_render(
                point_cloud, soft_labels, K, RT, pred_2d.shape[:2]
            )
            
            # Compare with 2D predictions
            loss_data += F.cross_entropy(
                rendered.reshape(-1, model.num_classes),
                pred_2d.reshape(-1)
            )
        
        # Smoothness term
        loss_smooth = compute_smoothness_loss(
            soft_labels, point_cloud, k_neighbors=10
        )
        
        # Entropy regularization (encourage confident predictions)
        entropy = -torch.sum(soft_labels * torch.log(soft_labels + 1e-10))
        loss_entropy = entropy / len(soft_labels)
        
        # Total loss
        loss = loss_data + lambda_smooth * loss_smooth + lambda_entropy * loss_entropy
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}")
    
    # Return hard labels
    return soft_labels.argmax(dim=1)
```

## Hierarchical Optimization

### 1. Multi-Resolution Approach
```python
class HierarchicalOptimization:
    """
    Optimize at multiple resolutions for efficiency.
    """
    
    def __init__(self, resolutions: List[float] = [0.2, 0.1, 0.05]):
        self.resolutions = resolutions
    
    def optimize_hierarchical(
        self,
        point_cloud: torch.Tensor,
        initial_predictions: torch.Tensor,
        camera_params: List[Tuple],
        predictions_2d: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Coarse-to-fine optimization.
        """
        current_labels = initial_predictions
        
        for resolution in self.resolutions:
            # Downsample point cloud
            downsampled_pc, downsample_indices = voxel_downsample(
                point_cloud, resolution
            )
            
            # Get labels for downsampled points
            downsampled_labels = current_labels[downsample_indices]
            
            # Optimize at this resolution
            optimized_labels = self.optimize_single_resolution(
                downsampled_pc,
                downsampled_labels,
                camera_params,
                predictions_2d,
                resolution
            )
            
            # Propagate to full resolution
            current_labels = self.propagate_labels(
                point_cloud,
                downsampled_pc,
                optimized_labels,
                current_labels
            )
        
        return current_labels
    
    def propagate_labels(
        self,
        full_pc: torch.Tensor,
        sparse_pc: torch.Tensor,
        sparse_labels: torch.Tensor,
        previous_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate labels from sparse to dense points.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest sparse point for each full point
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(sparse_pc.cpu().numpy())
        distances, indices = nbrs.kneighbors(full_pc.cpu().numpy())
        
        # Propagate labels
        propagated = sparse_labels[indices.squeeze()]
        
        # Blend with previous labels based on distance
        confidence = torch.exp(-torch.from_numpy(distances.squeeze()) / 0.1)
        
        # Keep previous label if far from any sparse point
        final_labels = torch.where(
            confidence > 0.5,
            propagated,
            previous_labels
        )
        
        return final_labels
```

## Integration with Pylon

```python
from utils.point_cloud_ops.rendering import render_segmentation

def pylon_consistency_optimization(
    pc_data: Dict[str, torch.Tensor],
    images: List[torch.Tensor],
    camera_params: List[Tuple[torch.Tensor, torch.Tensor]],
    segmentation_model_2d,
    optimization_method: str = "graph_cut",
    num_iterations: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Optimize 3D segmentation for consistency with 2D predictions.
    """
    
    points = pc_data['pos']
    colors = pc_data.get('rgb', torch.zeros_like(points))
    
    # Get 2D predictions
    predictions_2d = []
    confidence_2d = []
    
    for img in images:
        with torch.no_grad():
            pred = segmentation_model_2d(img.unsqueeze(0))
            predictions_2d.append(pred.squeeze(0))
            
            # Get confidence (max probability)
            conf = pred.max(dim=1)[0].squeeze(0)
            confidence_2d.append(conf)
    
    # Compute unary costs
    unary_costs = compute_unary_costs(
        points,
        predictions_2d,
        camera_params,
        segmentation_model_2d.num_classes
    )
    
    # Initialize with simple voting
    initial_labels = unary_costs.argmin(dim=1)
    
    if optimization_method == "graph_cut":
        # Graph-cut optimization
        optimizer = GraphCutOptimization(segmentation_model_2d.num_classes)
        graph = optimizer.build_graph(points, unary_costs)
        optimized_labels = optimizer.solve_alpha_expansion(
            graph, initial_labels, num_iterations
        )
        
    elif optimization_method == "mean_field":
        # Dense CRF optimization
        crf = DenseCRF3D(segmentation_model_2d.num_classes)
        Q = crf.mean_field_inference(
            unary_costs, points, colors
        )
        optimized_labels = Q.argmax(dim=1)
        
    elif optimization_method == "gradient":
        # Gradient-based optimization
        model = DifferentiableOptimization(
            len(points),
            segmentation_model_2d.num_classes
        )
        optimized_labels = optimize_labels_gradient(
            model, points, predictions_2d, camera_params, num_iterations
        )
        
    elif optimization_method == "hierarchical":
        # Hierarchical optimization
        optimizer = HierarchicalOptimization()
        optimized_labels = optimizer.optimize_hierarchical(
            points, initial_labels, camera_params, predictions_2d
        )
    
    # Update point cloud data
    pc_data_optimized = pc_data.copy()
    pc_data_optimized['labels'] = optimized_labels
    
    # Verify consistency by rendering
    for (K, RT) in camera_params[:1]:  # Check first view
        rendered_seg = render_segmentation_from_pointcloud(
            pc_data_optimized,
            K, RT,
            (images[0].shape[-1], images[0].shape[-2]),
            key='labels'
        )
        # Can compute consistency metric here
    
    return pc_data_optimized
```

## Advantages
1. **Multi-view consistency**: Enforces agreement across all views
2. **Spatial coherence**: Maintains smoothness in 3D space
3. **No training required**: Pure optimization approach
4. **Flexible formulation**: Can incorporate various priors and constraints
5. **Interpretable**: Clear objective function with understandable terms

## Limitations
1. **Computational cost**: Iterative optimization can be slow
2. **Local minima**: May get stuck in suboptimal solutions
3. **Parameter sensitivity**: Performance depends on weight parameters
4. **Memory requirements**: Graph/CRF construction needs neighbor relationships
5. **Assumption violations**: Assumes 2D predictions are reasonably accurate