import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

# Import the methods we want to test directly from the class
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
import os
from unittest.mock import patch, mock_open


@pytest.fixture
def mock_point_cloud():
    """Create a mock point cloud with 100 points."""
    # Create random point cloud data
    xyz = torch.rand(100, 3)
    feat = torch.rand(100, 3)  # RGB features
    
    # Create a point cloud tensor with positions and colors
    pc = torch.cat([xyz, feat], dim=1)
    return pc


@pytest.fixture
def mock_file_paths():
    """Create mock file paths for testing."""
    return {
        'pc_0_filepath': '/fake/path/pc_0.txt',
        'pc_1_filepath': '/fake/path/pc_1.txt',
        'pc_1_seg_filepath': '/fake/path/pc_1_seg.txt',
    }


# Mock os.path.isdir to make it return True for our fake paths
@pytest.fixture
def mock_isdir(monkeypatch):
    def mock_isdir_fn(path):
        if './fake_path' in path or '/fake/path' in path:
            return True
        return os.path.isdir(path)
    
    monkeypatch.setattr(os.path, 'isdir', mock_isdir_fn)


@patch('data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset.SLPCCDDataset._init_annotations')
@patch('utils.io.load_point_cloud')
@patch('os.path.exists')
def test_slpccd_load_datapoint(mock_exists, mock_load_pc, mock_init_annotations, mock_point_cloud, mock_file_paths, mock_isdir):
    """Test the _load_datapoint method of SLPCCDDataset."""
    # Setup mocks
    mock_exists.return_value = True
    mock_load_pc.return_value = mock_point_cloud
    
    # Create dataset with mocked initialization
    dataset = SLPCCDDataset(data_root="./fake_path", split='train', num_points=64)
    
    # Mock the annotations
    dataset.annotations = [mock_file_paths]
    
    # Test loading a datapoint
    inputs, labels, meta_info = dataset._load_datapoint(0)
    
    # Validate inputs
    assert set(inputs.keys()) == set(SLPCCDDataset.INPUT_NAMES)
    
    # Validate point cloud data
    for pc_key in SLPCCDDataset.INPUT_NAMES:
        pc = inputs[pc_key]
        assert pc is not None
        
        # Check hierarchical structure if enabled
        if dataset.use_hierarchy:
            assert isinstance(pc, dict)
            assert 'xyz' in pc
            assert 'neighbors_idx' in pc
            assert 'knearst_idx_in_another_pc' in pc
            assert 'raw_length' in pc
            assert 'feat' in pc
            
            # Check hierarchical data
            assert isinstance(pc['xyz'], list)
            assert len(pc['xyz']) == dataset.hierarchy_levels
            assert isinstance(pc['neighbors_idx'], list)
            assert len(pc['neighbors_idx']) == dataset.hierarchy_levels
            
            # Check first level point cloud
            assert isinstance(pc['xyz'][0], torch.Tensor)
            assert pc['xyz'][0].ndim == 2
            assert pc['xyz'][0].size(1) == 3
            assert pc['xyz'][0].size(0) == dataset.num_points
        else:
            # Non-hierarchical case would be tested here
            pass
    
    # Validate labels
    assert set(labels.keys()) == set(SLPCCDDataset.LABEL_NAMES)
    assert 'change_map' in labels
    change_map = labels['change_map']
    assert isinstance(change_map, torch.Tensor)
    assert change_map.dtype == torch.int64 or change_map.dtype == torch.long
    
    # Validate meta_info
    assert isinstance(meta_info, dict)
    assert 'idx' in meta_info
    assert 'pc_0_filepath' in meta_info
    assert 'pc_1_filepath' in meta_info


@patch('data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset.SLPCCDDataset._init_annotations')
def test_slpccd_initialization(mock_init_annotations, mock_isdir):
    """Test the initialization of SLPCCDDataset with different parameters."""
    # Test with default parameters
    dataset = SLPCCDDataset(data_root="./fake_path", split='train')
    assert dataset.num_points == 8192
    assert dataset.random_subsample == True
    assert dataset.use_hierarchy == True
    assert dataset.hierarchy_levels == 3
    assert dataset.knn_size == 16
    assert dataset.cross_knn_size == 16
    
    # Test with custom parameters
    dataset = SLPCCDDataset(
        data_root="./fake_path", 
        split='train',
        num_points=4096,
        random_subsample=False,
        use_hierarchy=False,
        hierarchy_levels=2,
        knn_size=8,
        cross_knn_size=8
    )
    assert dataset.num_points == 4096
    assert dataset.random_subsample == False
    assert dataset.use_hierarchy == False
    assert dataset.hierarchy_levels == 2
    assert dataset.knn_size == 8
    assert dataset.cross_knn_size == 8


def test_normalize_point_cloud():
    """Test the _normalize_point_cloud method of SLPCCDDataset."""
    # Create an instance of the class with a mocked initialization
    dataset = MagicMock(spec=SLPCCDDataset)
    
    # Manually bind the method to the mock instance
    normalize_pc = SLPCCDDataset._normalize_point_cloud.__get__(dataset)
    
    # Create an off-center point cloud
    pc = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=torch.float32)
    
    # Normalize the point cloud
    normalized_pc = normalize_pc(pc)
    
    # Verify the point cloud is centered at the origin
    assert torch.allclose(torch.mean(normalized_pc, dim=0), torch.zeros(3), atol=1e-6)
    
    # Verify the point cloud is scaled to unit sphere
    max_dist = torch.max(torch.norm(normalized_pc, dim=1))
    assert torch.isclose(max_dist, torch.tensor(1.0), atol=1e-6)


def test_random_subsample_point_cloud():
    """Test the _random_subsample_point_cloud method of SLPCCDDataset."""
    # Create an instance of the class with a mocked initialization
    dataset = MagicMock(spec=SLPCCDDataset)
    
    # Manually bind the method to the mock instance
    subsample_pc = SLPCCDDataset._random_subsample_point_cloud.__get__(dataset)
    
    # Test subsampling when there are too many points
    pc_large = torch.rand(100, 3)
    subsampled_large = subsample_pc(pc_large, 50)
    assert subsampled_large.shape == (50, 3)
    
    # Test padding when there are too few points
    pc_small = torch.rand(20, 3)
    subsampled_small = subsample_pc(pc_small, 50)
    assert subsampled_small.shape == (50, 3)
    
    # Test edge case with empty point cloud
    pc_empty = torch.zeros((0, 3))
    subsampled_empty = subsample_pc(pc_empty, 50)
    assert subsampled_empty.shape == (50, 3)


def test_compute_knn():
    """Test the _compute_knn method of SLPCCDDataset."""
    # Create an instance of the class with a mocked initialization
    dataset = MagicMock(spec=SLPCCDDataset)
    
    # Patch the KDTree to avoid actual computation
    class MockKDTree:
        def __init__(self, data):
            self.data = data
            
        def query(self, data, k):
            n_points = data.shape[0]
            # Make mock nearest neighbor indices (just sequential numbers for testing)
            # Return indices for k neighbors (excluding self)
            indices = np.zeros((n_points, k), dtype=np.int64)
            for i in range(n_points):
                # Fill with indices that aren't i
                indices[i] = [(j + i + 1) % n_points for j in range(k)]
            
            distances = np.ones_like(indices)
            return distances, indices
    
    # Replace the actual KDTree with our mock
    original_kdtree = SLPCCDDataset._compute_knn.__globals__['KDTree']
    SLPCCDDataset._compute_knn.__globals__['KDTree'] = MockKDTree
    
    try:
        # Manually bind the method to the mock instance
        compute_knn = SLPCCDDataset._compute_knn.__get__(dataset)
        
        # Create a simple point cloud
        pc = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32)
        
        # Compute KNN with k=2
        neighbors = compute_knn(pc, k=2)
        
        # Check the shape of the output
        assert neighbors.shape == (5, 2)
    finally:
        # Restore the original KDTree class
        SLPCCDDataset._compute_knn.__globals__['KDTree'] = original_kdtree


def test_compute_cross_knn():
    """Test the _compute_cross_knn method of SLPCCDDataset."""
    # Create an instance of the class with a mocked initialization
    dataset = MagicMock(spec=SLPCCDDataset)
    
    # Patch the KDTree to avoid actual computation
    class MockKDTree:
        def __init__(self, data):
            self.data = data
            
        def query(self, data, k):
            n_points = data.shape[0]
            # Make mock nearest neighbor indices (just sequential numbers for testing)
            indices = np.tile(np.arange(k), (n_points, 1))
            distances = np.ones_like(indices)
            return distances, indices
    
    # Replace the actual KDTree with our mock
    original_kdtree = SLPCCDDataset._compute_cross_knn.__globals__['KDTree']
    SLPCCDDataset._compute_cross_knn.__globals__['KDTree'] = MockKDTree
    
    try:
        # Manually bind the method to the mock instance
        compute_cross_knn = SLPCCDDataset._compute_cross_knn.__get__(dataset)
        
        # Create two simple point clouds
        pc1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        pc2 = torch.tensor([
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32)
        
        # Compute KNN with k=2
        neighbors = compute_cross_knn(pc1, pc2, k=2)
        
        # Check the shape of the output
        assert neighbors.shape == (3, 2)
    finally:
        # Restore the original KDTree class
        SLPCCDDataset._compute_cross_knn.__globals__['KDTree'] = original_kdtree


def test_class_constants():
    """Test the class constants defined in SLPCCDDataset."""
    # Check that the required class constants are defined correctly
    assert SLPCCDDataset.INPUT_NAMES == ['pc_0', 'pc_1']
    assert SLPCCDDataset.LABEL_NAMES == ['change_map']
    assert SLPCCDDataset.NUM_CLASSES == 2
    assert SLPCCDDataset.INV_OBJECT_LABEL == {0: "unchanged", 1: "changed"}
    assert SLPCCDDataset.CLASS_LABELS == {"unchanged": 0, "changed": 1}
    assert SLPCCDDataset.IGNORE_LABEL == -1
    assert SLPCCDDataset.SPLIT_OPTIONS == {'train', 'val', 'test'}
    assert isinstance(SLPCCDDataset.DATASET_SIZE, dict)
    assert set(SLPCCDDataset.DATASET_SIZE.keys()) == {'train', 'val', 'test'} 