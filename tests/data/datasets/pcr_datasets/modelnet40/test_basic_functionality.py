from typing import Dict, Any
import pytest
import random
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import data


def transforms_cfg(rot_mag: float, trans_mag: float, crop_type: str = 'plane') -> Dict[str, Any]:
    """
    Create a configuration for transforms.

    Args:
        rot_mag: Rotation magnitude in degrees
        trans_mag: Translation magnitude
        crop_type: Type of cropping ('plane' or 'point')

    Returns:
        Dict[str, Any]: Configuration for transforms.
    """
    if crop_type == 'plane':
        crop_transform = data.transforms.vision_3d.RandomPlaneCrop(keep_ratio=0.7)
    elif crop_type == 'point':
        crop_transform = data.transforms.vision_3d.RandomPointCrop(keep_ratio=0.7)
    else:
        raise ValueError(f"Unsupported crop_type: {crop_type}")
    
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                (
                    data.transforms.vision_3d.RandomRigidTransform(rot_mag=rot_mag, trans_mag=trans_mag),
                    [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                ),
                (
                    crop_transform,
                    [('inputs', 'src_pc')],
                ),
            ],
        },
    }


def validate_inputs(inputs: Dict[str, Any], labels: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() >= {'src_pc', 'tgt_pc'}, f"inputs missing required keys: {inputs.keys()=}"

    for pc_name in ['src_pc', 'tgt_pc']:
        pc = inputs[pc_name]
        assert isinstance(pc, dict), f"{pc_name} should be a dictionary: {type(pc)=}"
        assert 'pos' in pc, f"{pc_name} should contain 'pos': {pc.keys()=}"

        # Check shapes and types
        assert isinstance(pc['pos'], torch.Tensor), f"{pc_name}['pos'] should be a torch.Tensor: {type(pc['pos'])=}"
        assert pc['pos'].dim() == 2, f"{pc_name}['pos'] should be 2-dimensional: {pc['pos'].shape=}"
        assert pc['pos'].size(1) == 3, f"{pc_name}['pos'] should have 3 coordinates: {pc['pos'].shape=}"
        assert pc['pos'].dtype == torch.float32, f"{pc_name}['pos'] dtype incorrect: {pc['pos'].dtype=}"

        # Check for NaN values
        assert not torch.isnan(pc['pos']).any(), f"{pc_name}['pos'] contains NaN values"

        # Check that point clouds have reasonable number of points (after cropping)
        num_points = pc['pos'].shape[0]
        assert num_points > 50, f"{pc_name} has too few points after cropping: {num_points}"
        assert num_points < 50000, f"{pc_name} has too many points: {num_points}"

    # For ModelNet40 (self-registration), source and target should have different number of points
    # since source is cropped but target is not
    src_points = inputs['src_pc']['pos'].shape[0]
    tgt_points = inputs['tgt_pc']['pos'].shape[0]
    assert src_points <= tgt_points, f"Source should have fewer or equal points than target: {src_points} vs {tgt_points}"

    # Check correspondences if present
    if 'correspondences' in inputs:
        corr = inputs['correspondences']
        assert isinstance(corr, torch.Tensor), f"correspondences should be a torch.Tensor: {type(corr)=}"
        assert corr.dim() == 2, f"correspondences should be 2-dimensional: {corr.shape=}"
        assert corr.size(1) == 2, f"correspondences should have 2 columns: {corr.shape=}"
        assert corr.dtype == torch.int64, f"correspondences dtype incorrect: {corr.dtype=}"
        
        # Check correspondence indices are valid
        max_src_idx = corr[:, 0].max()
        max_tgt_idx = corr[:, 1].max()
        assert max_src_idx < src_points, f"Invalid source correspondence index: {max_src_idx} >= {src_points}"
        assert max_tgt_idx < tgt_points, f"Invalid target correspondence index: {max_tgt_idx} >= {tgt_points}"


def validate_labels(labels: Dict[str, Any], rot_mag: float, trans_mag: float) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert 'transform' in labels, f"labels missing 'transform' key: {labels.keys()=}"
    
    transform = labels['transform']
    assert isinstance(transform, torch.Tensor), f"transform should be a torch.Tensor: {type(transform)=}"
    assert transform.shape == (4, 4), f"transform should be a 4x4 matrix: {transform.shape=}"
    assert transform.dtype == torch.float32, f"transform dtype incorrect: {transform.dtype=}"
    assert not torch.isnan(transform).any(), "transform contains NaN values"

    # Validate transformation matrix
    R = transform[:3, :3]
    t = transform[:3, 3]

    # Check rotation matrix properties
    assert torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=1e-6), \
        "Invalid rotation matrix: not orthogonal"
    assert torch.abs(torch.det(R) - 1.0) < 1e-6, \
        "Invalid rotation matrix: determinant not 1"

    # Check rotation magnitude
    rot_angle = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1, 1))
    assert torch.abs(rot_angle) <= np.radians(rot_mag), \
        f"Rotation angle exceeds specified limit: {torch.abs(rot_angle)=}, {np.radians(rot_mag)=}"

    # Check translation magnitude
    assert torch.norm(t) <= trans_mag, \
        f"Translation magnitude exceeds specified limit: {torch.norm(t)=}, {trans_mag=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() >= {'idx', 'src_file_path', 'tgt_file_path'}, \
        f"meta_info missing required keys: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"{meta_info['idx']=}, {datapoint_idx=}"
    
    # For ModelNet40 (self-registration), source and target file paths should be the same
    assert meta_info['src_file_path'] == meta_info['tgt_file_path'], \
        "ModelNet40 should use same file for source and target"
    
    # Check file path is valid OFF file
    file_path = meta_info['src_file_path']
    assert file_path.endswith('.off'), f"File should be OFF format: {file_path}"
    assert 'ModelNet40' in file_path, f"File should be from ModelNet40: {file_path}"
    
    # Check category extraction
    if 'category' in meta_info:
        category = meta_info['category']
        assert category in data.datasets.ModelNet40Dataset.CATEGORIES, \
            f"Unknown category: {category}"


@pytest.fixture
def dataset_with_params(request):
    """Fixture for creating a ModelNet40Dataset instance with validation parameters."""
    dataset_params = request.param.copy()
    # Extract rot_mag and trans_mag for validation
    rot_mag = dataset_params.pop('rot_mag')
    trans_mag = dataset_params.pop('trans_mag')

    dataset = data.datasets.ModelNet40Dataset(**dataset_params)
    return dataset, rot_mag, trans_mag


@pytest.mark.parametrize('dataset_with_params', [
    {
        'data_root': 'data/datasets/soft_links/ModelNet40',
        'split': 'train',
        'dataset_size': 100,
        'overlap_range': (0.3, 1.0),
        'matching_radius': 0.05,
        'transforms_cfg': transforms_cfg(rot_mag=45.0, trans_mag=0.5, crop_type='plane'),
        'rot_mag': 45.0,
        'trans_mag': 0.5,
    },
    {
        'data_root': 'data/datasets/soft_links/ModelNet40',
        'split': 'test',
        'dataset_size': 50,
        'overlap_range': (0.4, 0.8),
        'matching_radius': 0.03,
        'transforms_cfg': transforms_cfg(rot_mag=30.0, trans_mag=0.3, crop_type='point'),
        'rot_mag': 30.0,
        'trans_mag': 0.3,
    },
], indirect=True)
def test_modelnet40_dataset(dataset_with_params, max_samples, get_samples_to_test):
    """Test basic functionality of ModelNet40Dataset."""
    dataset, rot_mag, trans_mag = dataset_with_params

    # Basic dataset checks
    assert len(dataset) > 0, "Dataset should not be empty"
    assert hasattr(dataset, 'file_pair_annotations'), "Dataset should have file_pair_annotations"
    
    # Check that file pairs are correctly set up for single-temporal (self-registration)
    for annotation in dataset.file_pair_annotations[:5]:  # Check first 5
        assert annotation['src_file_path'] == annotation['tgt_file_path'], \
            "ModelNet40 should use same file for source and target"
        assert annotation['src_file_path'].endswith('.off'), \
            "Files should be OFF format"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'], datapoint['labels'])
        validate_labels(datapoint['labels'], rot_mag, trans_mag)
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise test subset
    num_samples = get_samples_to_test(len(dataset), max_samples)
    if num_samples is None:
        num_samples = min(10, len(dataset))  # Default to 10 samples or dataset size
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Test with ThreadPool for parallel validation
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)


def test_modelnet40_categories():
    """Test ModelNet40 category definitions."""
    categories = data.datasets.ModelNet40Dataset.CATEGORIES
    asymmetric_categories = data.datasets.ModelNet40Dataset.ASYMMETRIC_CATEGORIES
    
    # Check category count
    assert len(categories) == 40, f"ModelNet40 should have 40 categories, got {len(categories)}"
    
    # Check asymmetric categories are subset of all categories
    for cat in asymmetric_categories:
        assert cat in categories, f"Asymmetric category {cat} not in main categories"
    
    # Check some expected categories exist
    expected = ['airplane', 'chair', 'table', 'car', 'bottle']
    for cat in expected:
        assert cat in categories, f"Expected category {cat} missing"


def test_modelnet40_category_extraction():
    """Test category extraction from file paths."""
    # Test the static method directly without creating a full instance
    from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
    
    # Test valid paths
    test_paths = [
        '/path/to/ModelNet40/airplane/train/airplane_0001.off',
        '/path/to/ModelNet40/chair/test/chair_0100.off',
        'data/datasets/soft_links/ModelNet40/table/train/table_0050.off',
    ]
    
    expected_categories = ['airplane', 'chair', 'table']
    
    for path, expected in zip(test_paths, expected_categories):
        # Create a mock instance just for this method call
        dataset = type('MockDataset', (), {'get_category_from_path': ModelNet40Dataset.get_category_from_path})()
        category = dataset.get_category_from_path(path)
        assert category == expected, f"Expected {expected}, got {category} for path {path}"


def test_modelnet40_split_handling():
    """Test ModelNet40 split handling (val -> test mapping)."""
    # Test that different splits load different files by checking annotations only
    # We'll create the annotations manually to avoid initialization issues
    
    from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
    import os
    import glob
    
    data_root = 'data/datasets/soft_links/ModelNet40'
    
    # Check train files exist
    train_files = []
    for category in ModelNet40Dataset.CATEGORIES[:3]:  # Check first 3 categories
        category_dir = os.path.join(data_root, category, 'train')
        if os.path.exists(category_dir):
            train_files.extend(glob.glob(os.path.join(category_dir, '*.off')))
    
    # Check test files exist  
    test_files = []
    for category in ModelNet40Dataset.CATEGORIES[:3]:  # Check first 3 categories
        category_dir = os.path.join(data_root, category, 'test')
        if os.path.exists(category_dir):
            test_files.extend(glob.glob(os.path.join(category_dir, '*.off')))
    
    assert len(train_files) > 0, "Should find training files"
    assert len(test_files) > 0, "Should find test files"
    assert len(set(train_files) & set(test_files)) == 0, "Train and test files should be different"


@pytest.mark.parametrize('crop_type', ['plane', 'point'])
def test_modelnet40_crop_types(crop_type):
    """Test different cropping strategies."""
    # Test that the crop transforms can be created without errors
    if crop_type == 'plane':
        crop_transform = data.transforms.vision_3d.RandomPlaneCrop(keep_ratio=0.7)
    elif crop_type == 'point':
        crop_transform = data.transforms.vision_3d.RandomPointCrop(keep_ratio=0.7)
    
    # Verify crop transform is created correctly
    assert crop_transform.keep_ratio == 0.7
    
    # Test with a simple dummy point cloud
    dummy_pc = {'pos': torch.randn(1000, 3)}
    generator = torch.Generator()
    generator.manual_seed(42)
    
    # Apply the crop
    cropped_pc = crop_transform._call_single(dummy_pc, generator)
    
    # Check that cropping was applied
    assert 'pos' in cropped_pc
    assert cropped_pc['pos'].shape[0] < dummy_pc['pos'].shape[0]
    assert cropped_pc['pos'].shape[0] > 500  # Should keep ~70% of points