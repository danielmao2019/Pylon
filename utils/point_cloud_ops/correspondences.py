from typing import Optional
import numpy as np
import torch
from scipy.spatial import cKDTree
import open3d as o3d
from utils.point_cloud_ops.apply_transform import apply_transform


def get_correspondences(src_points: torch.Tensor, tgt_points: torch.Tensor, transform: Optional[torch.Tensor], radius: float) -> torch.Tensor:
    """Find correspondences between two point clouds within a matching radius.

    Args:
        src_points (torch.Tensor): Source point cloud [M, 3]
        tgt_points (torch.Tensor): Target point cloud [N, 3]
        transform (torch.Tensor): Transformation matrix from source to target [4, 4] or None
        radius (float): Maximum distance threshold for correspondence matching

    Returns:
        torch.Tensor: Correspondence indices [K, 2] where K is number of correspondences
    """
    import psutil
    import os
    import gc
    
    def get_memory_info():
        """Get current memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"PID: {process.pid}, RSS: {mem_info.rss / 1024**3:.2f}GB, VMS: {mem_info.vms / 1024**3:.2f}GB, GPU: {gpu_mem:.2f}GB, GPU_cached: {gpu_cached:.2f}GB"
        else:
            return f"PID: {process.pid}, RSS: {mem_info.rss / 1024**3:.2f}GB, VMS: {mem_info.vms / 1024**3:.2f}GB"
    
    def get_detailed_memory_info():
        """Get detailed memory breakdown."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Get memory maps if available
        try:
            mem_maps = process.memory_maps()
            total_heap = sum(m.rss for m in mem_maps if m.path == '[heap]') / 1024**3
            total_anon = sum(m.rss for m in mem_maps if m.path == '' or m.path.startswith('[')) / 1024**3
        except:
            total_heap = 0
            total_anon = 0
        
        # Python object count
        gc.collect()  # Force garbage collection
        obj_count = len(gc.get_objects())
        
        return f"RSS: {mem_info.rss / 1024**3:.2f}GB, VMS: {mem_info.vms / 1024**3:.2f}GB, Heap: {total_heap:.2f}GB, Anon: {total_anon:.2f}GB, PyObjects: {obj_count:,}"
    
    print(f"        🔍 get_correspondences: radius={radius}")
    print(f"          Initial memory: {get_memory_info()}")
    print(f"          src_points shape: {src_points.shape}, device: {src_points.device}")
    print(f"          tgt_points shape: {tgt_points.shape}, device: {tgt_points.device}")
    if transform is not None:
        print(f"          transform shape: {transform.shape}, device: {transform.device}")
    
    assert src_points.device == tgt_points.device, f"{src_points.device=}, {tgt_points.device=}"
    device = src_points.device

    print(f"          Converting to numpy...")
    # Convert to numpy for scipy operations
    tgt_points_np = tgt_points.cpu().numpy()
    src_points_np = src_points.cpu().numpy()
    if transform is not None:
        transform_np = transform.cpu().numpy()
    else:
        transform_np = None
    
    print(f"          src_points_np shape: {src_points_np.shape}, size: {src_points_np.nbytes / 1024**2:.2f}MB")
    print(f"          tgt_points_np shape: {tgt_points_np.shape}, size: {tgt_points_np.nbytes / 1024**2:.2f}MB")
    print(f"          Memory after numpy conversion: {get_memory_info()}")

    # Transform source points to reference frame
    if transform_np is not None:
        print(f"          Applying transform...")
        src_points_transformed = apply_transform(src_points_np, transform_np)
        print(f"          Transformed src_points shape: {src_points_transformed.shape}")
        print(f"          Memory after transform: {get_memory_info()}")
    else:
        src_points_transformed = src_points_np

    print(f"          Building KD-tree...")
    print(f"          Pre-KD-tree detailed: {get_detailed_memory_info()}")
    # Build KD-tree for efficient search
    src_tree = cKDTree(src_points_transformed)
    print(f"          KD-tree built with {len(src_points_transformed)} points")
    print(f"          Memory after KD-tree: {get_memory_info()}")
    print(f"          Post-KD-tree detailed: {get_detailed_memory_info()}")

    print(f"          Finding correspondences within radius {radius}...")
    # Find correspondences within radius
    indices_list = src_tree.query_ball_point(tgt_points_np, radius)
    
    # Count total correspondences before creating them
    total_correspondences = sum(len(indices) for indices in indices_list)
    print(f"          Found {total_correspondences} total correspondences")
    print(f"          Memory after query_ball_point: {get_memory_info()}")
    
    if total_correspondences > 100000:  # Large number threshold
        print(f"          ⚠️  WARNING: Very large number of correspondences: {total_correspondences:,}")

    print(f"          Creating correspondence pairs...")
    print(f"          🔍 MEMORY FRAGMENTATION ANALYSIS:")
    
    # Get baseline memory before list creation
    baseline_detailed = get_detailed_memory_info()
    print(f"            BEFORE list creation: {baseline_detailed}")
    baseline_rss = float(baseline_detailed.split("RSS: ")[1].split("GB")[0])
    
    # Calculate theoretical memory needed for Python list
    theoretical_list_size = total_correspondences * (24 + 2 * 28)  # tuple overhead + 2 ints
    print(f"            Theoretical list size: {theoretical_list_size / 1024**2:.2f}MB")
    
    # Create correspondence pairs
    corr_list = [
        (i, j)
        for j, indices in enumerate(indices_list)
        for i in indices
    ]
    
    # Get memory after list creation
    after_list_detailed = get_detailed_memory_info()
    print(f"            AFTER list creation: {after_list_detailed}")
    after_list_rss = float(after_list_detailed.split("RSS: ")[1].split("GB")[0])
    
    # Calculate actual memory increase and fragmentation
    actual_increase = (after_list_rss - baseline_rss) * 1024**3  # Convert to bytes
    fragmentation_factor = actual_increase / theoretical_list_size if theoretical_list_size > 0 else 0
    print(f"            Actual memory increase: {actual_increase / 1024**2:.2f}MB")
    print(f"            Fragmentation factor: {fragmentation_factor:.1f}x")
    print(f"            Memory overhead: {(actual_increase - theoretical_list_size) / 1024**2:.2f}MB")
    
    print(f"          Created {len(corr_list)} correspondence pairs")
    print(f"          Memory after creating pairs: {get_memory_info()}")
    
    # Handle empty case - ensure we get a 2D array with shape (0, 2)
    if len(corr_list) == 0:
        corr_indices = np.empty((0, 2), dtype=np.int64)
        print(f"          No correspondences found")
    else:
        print(f"          Converting to numpy array...")
        print(f"          🔍 NUMPY ALLOCATION ANALYSIS:")
        
        # Track memory before numpy allocation  
        before_numpy_detailed = get_detailed_memory_info()
        print(f"            BEFORE numpy: {before_numpy_detailed}")
        before_numpy_rss = float(before_numpy_detailed.split("RSS: ")[1].split("GB")[0])
        
        theoretical_numpy_size = len(corr_list) * 2 * 8  # 2 int64s per correspondence
        print(f"            Theoretical numpy size: {theoretical_numpy_size / 1024**2:.2f}MB")
        print(f"          About to allocate numpy array: {len(corr_list)} x 2 int64 = {theoretical_numpy_size / 1024**2:.2f}MB")
        
        try:
            corr_indices = np.array(corr_list, dtype=np.int64)
            print(f"          ✅ Successfully created numpy array")
            
            # Track memory after numpy allocation
            after_numpy_detailed = get_detailed_memory_info() 
            print(f"            AFTER numpy: {after_numpy_detailed}")
            after_numpy_rss = float(after_numpy_detailed.split("RSS: ")[1].split("GB")[0])
            
            # Calculate numpy allocation overhead
            numpy_actual_increase = (after_numpy_rss - before_numpy_rss) * 1024**3
            numpy_fragmentation = numpy_actual_increase / theoretical_numpy_size if theoretical_numpy_size > 0 else 0
            print(f"            Numpy actual increase: {numpy_actual_increase / 1024**2:.2f}MB") 
            print(f"            Numpy fragmentation factor: {numpy_fragmentation:.1f}x")
            
            print(f"          corr_indices shape: {corr_indices.shape}, size: {corr_indices.nbytes / 1024**2:.2f}MB")
            print(f"          Memory after numpy array: {get_memory_info()}")
        except Exception as e:
            print(f"          ❌ FAILED to create numpy array: {e}")
            print(f"          Exception type: {type(e)}")
            print(f"          Memory at failure: {get_memory_info()}")
            raise

    print(f"          Converting to torch tensor...")
    print(f"          About to allocate torch tensor: {corr_indices.nbytes / 1024**2:.2f}MB")
    print(f"          Memory before torch allocation: {get_memory_info()}")
    
    try:
        result = torch.tensor(corr_indices, dtype=torch.int64, device=device)
        print(f"          ✅ Successfully created torch tensor")
        print(f"          Result shape: {result.shape}, elements: {result.numel():,}")
        print(f"          Final memory: {get_memory_info()}")
        print(f"        ✅ get_correspondences completed")
    except Exception as e:
        print(f"          ❌ FAILED to create torch tensor: {e}")
        print(f"          Exception type: {type(e)}")
        print(f"          Memory at torch failure: {get_memory_info()}")
        raise
    
    return result


def get_correspondences_v2(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    trans: np.ndarray,
    search_voxel_size: float,
    K: Optional[int] = None
) -> torch.Tensor:
    """Find correspondences between two point clouds within a matching radius.

    Args:
        src_pcd (o3d.geometry.PointCloud): Source point cloud
        tgt_pcd (o3d.geometry.PointCloud): Target point cloud
        trans (np.ndarray): Transformation matrix from source to target
        search_voxel_size (float): Search voxel size
    """
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences
