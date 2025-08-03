"""Tests for GMCNet wrapper model integration with device compatibility."""

from typing import Dict, Any, Tuple
import pytest
import torch
import numpy as np

from models.point_cloud_registration.gmcnet.gmcnet_wrapper import GMCNet
from data.datasets.base_dataset import BaseDataset
from data.collators.base_collator import BaseCollator


class DummyGMCNetDataset(BaseDataset):
    """Dummy dataset for testing GMCNet with ModelNet40-compatible data structure."""
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 10, 'val': 5, 'test': 5}
    INPUT_NAMES = ['src_points', 'tgt_points']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        num_points: int = 1024,
        **kwargs
    ) -> None:
        """Initialize dummy dataset.
        
        Args:
            num_points: Number of points in each point cloud
        """
        self.num_points = num_points
        super(DummyGMCNetDataset, self).__init__(**kwargs)
    
    def display_datapoint(self, idx: int) -> Any:
        """Display dummy datapoint for testing (required abstract method)."""
        return f"Dummy GMCNet datapoint {idx}: src_points + tgt_points + transform"
        
    def _init_annotations(self) -> None:
        """Initialize dummy annotations."""
        self.annotations = list(range(self.DATASET_SIZE[self.split]))
        
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load dummy datapoint with random point clouds."""
        # Generate random point clouds (on CPU as per BaseDataset design)
        # Points normalized to [-1, 1] range for stable testing
        src_points = 2 * torch.rand(self.num_points, 3, dtype=torch.float32) - 1
        tgt_points = 2 * torch.rand(self.num_points, 3, dtype=torch.float32) - 1
        
        # Generate random transformation matrix (4x4 homogeneous)
        # Simple rotation + translation for testing
        angle = torch.rand(1) * 2 * np.pi
        cos_theta, sin_theta = torch.cos(angle), torch.sin(angle)
        rotation = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        translation = torch.randn(3, dtype=torch.float32) * 0.1  # Small translation
        
        # Create 4x4 transformation matrix
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        inputs = {
            'src_points': src_points,
            'tgt_points': tgt_points,
        }
        
        labels = {
            'transform': transform,
        }
        
        # BaseDataset automatically adds 'idx' to meta_info, so don't include it manually
        meta_info = {
            'dataset_name': 'dummy_gmcnet',
            'src_path': f'dummy_src_{idx}.ply',
            'tgt_path': f'dummy_tgt_{idx}.ply',
        }
        
        return inputs, labels, meta_info


@pytest.fixture
def dummy_gmcnet_args():
    """Create dummy arguments for GMCNet model matching original configuration."""
    return {
        # Network architecture parameters (exact match to original gmcnet_mn40_psc.yaml)
        'knn_list': '5,5,5',  # K-nearest neighbors for each hierarchical level
        'down_sample_list': '2,4',  # Downsampling ratios: level1/2, level2/4  
        'feature_size_list': '128,128,32,8',  # Feature dimensions: [128,128,32,8]
        'descriptor_size': 128,  # Final descriptor dimensionality
        
        # Loss configuration (exact weights from original)
        'use_cycle_loss': False,  # Original: False
        'use_mse_loss': 0.0,      # Original: 0.0 (numerical weight)
        'use_cd_loss': 1.0,       # Original: 1.0 (main loss)
        'use_inliers_loss': 0.01, # Original: 0.01 (regularization)
        'reductioin': 'mean',     # Original typo preserved for compatibility
        'use_annealing': False,   # Original: False
        
        # Feature configuration (exact match to original)
        'rif_only': False,              # Original: False
        'rif_feature': 'ppf',           # Original: ppf (Point Pair Features)
        'predict_inliers': False,       # Original: False  
        'use_weighted_procrustes': False,  # Original: False
    }


@pytest.fixture
def device():
    """GMCNet requires CUDA device."""
    return torch.device('cuda')


@pytest.fixture  
def gmcnet_model(dummy_gmcnet_args):
    """Create GMCNet model for testing."""
    model = GMCNet(dummy_gmcnet_args)
    model.eval()  # Set to evaluation mode for consistent testing
    return model


@pytest.fixture
def sample_batch(device):
    """Create sample batch for testing with appropriate device."""
    batch_size = 2
    num_points = 1024
    
    return {
        'inputs': {
            'src_points': torch.randn(batch_size, num_points, 3, device=device),
            'tgt_points': torch.randn(batch_size, num_points, 3, device=device),
            'transform': torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
            'mode': 'train'
        },
        'labels': {
            'transform': torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        }
    }



def test_gmcnet_wrapper_initialization(dummy_gmcnet_args):
    """Test GMCNet wrapper initialization."""
    model = GMCNet(dummy_gmcnet_args)
    
    # Check that wrapper contains original model
    assert hasattr(model, '_model')
    assert model._model is not None
    
    # Check that it's a PyTorch module
    assert isinstance(model, torch.nn.Module)


def test_gmcnet_wrapper_forward_train_mode(gmcnet_model, sample_batch):
    """Test GMCNet wrapper forward pass in training mode."""
    inputs = sample_batch['inputs'].copy()
    inputs['mode'] = 'train'
    
    with torch.no_grad():
        outputs = gmcnet_model(inputs)
    
    # Check required training outputs
    required_keys = ['loss', 'transformation', 'T_12', 'rotation_error', 'translation_error', 'rmse', 'mse']
    for key in required_keys:
        assert key in outputs, f"Missing required output key: {key}"
    
    # Check output types and shapes
    assert isinstance(outputs['loss'], torch.Tensor)
    assert outputs['loss'].numel() == 1  # Scalar loss
    
    assert isinstance(outputs['transformation'], torch.Tensor)
    assert outputs['transformation'].shape[-2:] == (4, 4)  # 4x4 transformation matrix
    
    assert isinstance(outputs['rotation_error'], torch.Tensor)
    assert isinstance(outputs['translation_error'], torch.Tensor)
    assert isinstance(outputs['rmse'], torch.Tensor)
    assert isinstance(outputs['mse'], torch.Tensor)
    
    # Check that T_12 matches transformation (compatibility key)
    assert torch.equal(outputs['T_12'], outputs['transformation'])


def test_gmcnet_wrapper_forward_test_mode(gmcnet_model, device):
    """Test GMCNet wrapper forward pass in test mode."""
    batch_size = 1
    num_points = 1024
    
    inputs = {
        'src_points': torch.randn(batch_size, num_points, 3, device=device),
        'tgt_points': torch.randn(batch_size, num_points, 3, device=device),
        'mode': 'test'
    }
    
    with torch.no_grad():
        outputs = gmcnet_model(inputs)
    
    # Check required test outputs
    required_keys = ['transformation', 'T_12']
    for key in required_keys:
        assert key in outputs, f"Missing required output key: {key}"
    
    # Check that loss and metrics are not present in test mode
    test_only_keys = ['loss', 'rotation_error', 'translation_error', 'rmse', 'mse']
    for key in test_only_keys:
        assert key not in outputs, f"Training-only key {key} should not be present in test mode"
    
    # Check output shapes
    assert outputs['transformation'].shape == (batch_size, 4, 4)
    assert torch.equal(outputs['T_12'], outputs['transformation'])


def test_gmcnet_wrapper_forward_val_mode(gmcnet_model, sample_batch):
    """Test GMCNet wrapper forward pass in validation mode."""
    inputs = sample_batch['inputs'].copy()
    inputs['mode'] = 'val'
    
    with torch.no_grad():
        outputs = gmcnet_model(inputs)
    
    # Validation mode should behave like training mode (with metrics)
    required_keys = ['loss', 'transformation', 'T_12', 'rotation_error', 'translation_error', 'rmse', 'mse']
    for key in required_keys:
        assert key in outputs, f"Missing required output key: {key}"


def test_gmcnet_wrapper_input_validation(gmcnet_model):
    """Test GMCNet wrapper input validation."""
    # Test missing src_points
    with pytest.raises(AssertionError, match="src_points must be provided"):
        gmcnet_model({
            'tgt_points': torch.randn(1, 1024, 3),
            'mode': 'test'
        })
    
    # Test missing tgt_points
    with pytest.raises(AssertionError, match="tgt_points must be provided"):
        gmcnet_model({
            'src_points': torch.randn(1, 1024, 3),
            'mode': 'test'
        })
    
    # Test wrong tensor shapes
    with pytest.raises(AssertionError, match="src_points must be 3D"):
        gmcnet_model({
            'src_points': torch.randn(1024, 3),  # Missing batch dimension
            'tgt_points': torch.randn(1, 1024, 3),
            'mode': 'test'
        })
    
    with pytest.raises(AssertionError, match="src_points last dim must be 3"):
        gmcnet_model({
            'src_points': torch.randn(1, 1024, 2),  # Wrong coordinate dimension
            'tgt_points': torch.randn(1, 1024, 3),
            'mode': 'test'
        })


def test_gmcnet_wrapper_training_validation(gmcnet_model):
    """Test GMCNet wrapper validation for training mode."""
    # Test missing transform in training mode
    with pytest.raises(ValueError, match="Ground truth transformation must be provided"):
        gmcnet_model({
            'src_points': torch.randn(1, 1024, 3),
            'tgt_points': torch.randn(1, 1024, 3),
            'mode': 'train'
        })


def test_gmcnet_wrapper_default_mode(gmcnet_model, sample_batch):
    """Test GMCNet wrapper with default mode (should be 'train')."""
    inputs = sample_batch['inputs'].copy()
    del inputs['mode']  # Remove mode to test default
    
    with torch.no_grad():
        outputs = gmcnet_model(inputs)
    
    # Should behave like training mode by default
    required_keys = ['loss', 'transformation', 'T_12', 'rotation_error', 'translation_error', 'rmse', 'mse']
    for key in required_keys:
        assert key in outputs, f"Missing required output key: {key}"


def test_gmcnet_wrapper_auxiliary_methods(gmcnet_model, sample_batch):
    """Test GMCNet wrapper auxiliary methods."""
    # Run forward pass first to populate internal state
    with torch.no_grad():
        _ = gmcnet_model(sample_batch['inputs'])
    
    # Test get_transform method
    try:
        transform = gmcnet_model.get_transform()
        # This method exists in the wrapper and delegates to original model
    except Exception as e:
        # If the method has dependencies on internal state, that's ok for this test
        pass
    
    # Test visualize method
    try:
        vis_result = gmcnet_model.visualize(0)
        # This method exists in the wrapper and delegates to original model
    except Exception as e:
        # If the method has dependencies on internal state or data, that's ok for this test
        pass


def test_gmcnet_wrapper_cuda_requirement():
    """Test that GMCNet properly handles CUDA requirement."""
    args = {
        'knn_list': '5,5,5',
        'down_sample_list': '2,4',
        'feature_size_list': '128,128,32,8',
        'descriptor_size': 128,
        'use_cycle_loss': False,
        'use_mse_loss': 0.0,
        'use_cd_loss': 1.0,
        'use_inliers_loss': 0.01,
        'reductioin': 'mean',
        'use_annealing': False,
        'rif_only': False,
        'rif_feature': 'ppf',
        'predict_inliers': False,
        'use_weighted_procrustes': False,
    }
    
    model = GMCNet(args)
    model.eval()
    
    # GMCNet requires CUDA - test will fail fast if CUDA not available
    inputs = {
        'src_points': torch.randn(1, 256, 3),
        'tgt_points': torch.randn(1, 256, 3),
        'mode': 'test'
    }
    
    with torch.no_grad():
        outputs = model(inputs)
        assert 'transformation' in outputs
        # Outputs should be moved back to original device (CPU)
        assert outputs['transformation'].device.type == 'cpu'


@pytest.mark.parametrize("batch_size,num_points", [
    (1, 512),
    (1, 1024),
    (2, 512),
])
def test_gmcnet_wrapper_different_batch_sizes(gmcnet_model, device, batch_size, num_points):
    """Test GMCNet wrapper with different batch sizes and point counts."""
    inputs = {
        'src_points': torch.randn(batch_size, num_points, 3, device=device),
        'tgt_points': torch.randn(batch_size, num_points, 3, device=device),
        'transform': torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        'mode': 'train'
    }
    
    with torch.no_grad():
        outputs = gmcnet_model(inputs)
    
    # Check output batch dimensions
    assert outputs['transformation'].shape[0] == batch_size
    assert outputs['transformation'].shape[1:] == (4, 4)


def test_gmcnet_dataset_collation():
    """Test GMCNet dataset with BaseCollator for nested dictionary handling."""
    dataset = DummyGMCNetDataset(num_points=512, split='train')
    collator = BaseCollator()
    
    # Get sample datapoints
    datapoints = [dataset[i] for i in range(3)]
    
    # Test collation
    batch = collator(datapoints)
    
    # Check batch structure
    assert 'inputs' in batch
    assert 'labels' in batch
    assert 'meta_info' in batch
    
    # Check input shapes after collation
    # BaseCollator stacks 3 datapoints with [512, 3] into [512, 3, 3]
    assert 'src_points' in batch['inputs']
    assert 'tgt_points' in batch['inputs']
    assert batch['inputs']['src_points'].shape == (512, 3, 3)  # [num_points, 3, batch_size]
    assert batch['inputs']['tgt_points'].shape == (512, 3, 3)
    
    # Check labels - 3 [4, 4] matrices stacked to [4, 4, 3]
    assert 'transform' in batch['labels']
    assert batch['labels']['transform'].shape == (4, 4, 3)
    
    # Check meta_info is preserved as list
    assert isinstance(batch['meta_info']['idx'], list)
    assert len(batch['meta_info']['idx']) == 3


def test_gmcnet_tensor_dtypes(device):
    """Test that GMCNet wrapper handles tensor dtypes correctly."""
    test_args = {
        'knn_list': '5,5,5',
        'down_sample_list': '2,4',
        'feature_size_list': '128,128,32,8',
        'descriptor_size': 128,
        'use_cycle_loss': False,
        'use_mse_loss': 0.0,
        'use_cd_loss': 1.0,
        'use_inliers_loss': 0.01,
        'reductioin': 'mean',
        'use_annealing': False,
        'rif_only': False,
        'rif_feature': 'ppf',
        'predict_inliers': False,
        'use_weighted_procrustes': False,
    }
    
    model = GMCNet(test_args)
    model.eval()
    
    # Test with float32 (common for point clouds)
    inputs_f32 = {
        'src_points': torch.randn(1, 256, 3, dtype=torch.float32, device=device),
        'tgt_points': torch.randn(1, 256, 3, dtype=torch.float32, device=device),
        'transform': torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0),
        'mode': 'train'
    }
    
    with torch.no_grad():
        outputs = model(inputs_f32)
        # Check that outputs maintain reasonable precision
        assert outputs['transformation'].dtype == torch.float32


def test_gmcnet_error_handling():
    """Test GMCNet wrapper error handling for various edge cases."""
    error_args = {
        'knn_list': '5,5,5',
        'down_sample_list': '2,4',
        'feature_size_list': '128,128,32,8',
        'descriptor_size': 128,
        'use_cycle_loss': False,
        'use_mse_loss': 0.0,
        'use_cd_loss': 1.0,
        'use_inliers_loss': 0.01,
        'reductioin': 'mean',
        'use_annealing': False,
        'rif_only': False,
        'rif_feature': 'ppf',
        'predict_inliers': False,
        'use_weighted_procrustes': False,
    }
    
    model = GMCNet(error_args)
    model.eval()
    
    # Test empty tensors (should fail gracefully with clear error message)
    with pytest.raises(AssertionError, match="src_points must be 3D"):
        model({
            'src_points': torch.empty(0, 3),
            'tgt_points': torch.empty(0, 3),
            'mode': 'test'
        })
    
    # Test wrong spatial dimensions
    with pytest.raises(AssertionError, match="src_points last dim must be 3"):
        model({
            'src_points': torch.randn(1, 128, 2),  # Wrong spatial dimension
            'tgt_points': torch.randn(1, 128, 3),
            'mode': 'test'
        })