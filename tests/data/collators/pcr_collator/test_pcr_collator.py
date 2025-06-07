import torch
from data.collators.buffer.buffer_collate_fn import buffer_collate_fn
from data.collators.geotransformer.geotransformer_collate_fn import geotransformer_collate_fn
from data.collators.overlappredator.overlappredator_collate_fn import overlappredator_collate_fn
from .buffer_collator_gt import buffer_collate_fn_gt
from .geotransformer_collator_gt import geotransformer_collate_fn_gt
from .overlappredator_collator_gt import overlappredator_collate_fn_gt


def create_dummy_buffer_data():
    """Create dummy input data for buffer collator."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy point clouds
    src_pc_fds = {'pos': torch.randn(100, 3, device=device)}
    tgt_pc_fds = {'pos': torch.randn(100, 3, device=device)}
    
    # Create dummy keypoints with normals as features
    src_pc_sds = {
        'pos': torch.randn(50, 3, device=device),
        'normals': torch.randn(50, 3, device=device)
    }
    tgt_pc_sds = {
        'pos': torch.randn(50, 3, device=device),
        'normals': torch.randn(50, 3, device=device)
    }
    
    # Create dummy transform
    transform = torch.eye(4, device=device)
    
    return [{
        'inputs': {
            'src_pc_fds': src_pc_fds,
            'tgt_pc_fds': tgt_pc_fds,
            'src_pc_sds': src_pc_sds,
            'tgt_pc_sds': tgt_pc_sds,
        },
        'labels': {
            'transform': transform
        },
        'meta_info': {
            'src_file': 'dummy_src.ply',
            'tgt_file': 'dummy_tgt.ply'
        }
    }]


def create_dummy_geotransformer_data():
    """Create dummy input data for geotransformer collator."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy point clouds with features
    src_pc = {
        'pos': torch.randn(100, 3, device=device),
        'feat': torch.randn(100, 32, device=device)  # 32-dim features
    }
    tgt_pc = {
        'pos': torch.randn(100, 3, device=device),
        'feat': torch.randn(100, 32, device=device)  # 32-dim features
    }
    
    # Create dummy transform
    transform = torch.eye(4, device=device)
    
    return [{
        'inputs': {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': torch.randint(0, 100, (50, 2), device=device)
        },
        'labels': {
            'transform': transform
        },
        'meta_info': {
            'src_file': 'dummy_src.ply',
            'tgt_file': 'dummy_tgt.ply'
        }
    }]


def create_dummy_overlappredator_data():
    """Create dummy input data for overlappredator collator."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy point clouds with features
    src_pc = {
        'pos': torch.randn(100, 3, device=device),
        'feat': torch.randn(100, 32, device=device)  # 32-dim features
    }
    tgt_pc = {
        'pos': torch.randn(100, 3, device=device),
        'feat': torch.randn(100, 32, device=device)  # 32-dim features
    }
    
    # Create dummy transform
    transform = torch.eye(4, device=device)
    
    return [{
        'inputs': {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': torch.randint(0, 100, (50, 2), device=device)
        },
        'labels': {
            'transform': transform
        },
        'meta_info': {}
    }]


class DummyConfig:
    """Dummy config class for testing."""
    def __init__(self):
        self.data = type('Data', (), {'voxel_size_0': 0.1})()
        self.point = type('Point', (), {'conv_radius': 2.5})()
        self.first_subsampling_dl = 0.1
        self.conv_radius = 2.5
        self.deform_radius = 3.0
        self.architecture = ['conv', 'pool', 'conv', 'pool', 'conv']


def test_buffer_collator():
    """Test buffer collator implementation against ground truth."""
    # Create dummy data and config
    list_data = create_dummy_buffer_data()
    config = DummyConfig()
    neighborhood_limits = [16, 16, 16]
    
    # Get results from both implementations
    result_gt = buffer_collate_fn_gt(list_data, config, neighborhood_limits)
    result_new = buffer_collate_fn(list_data, config, neighborhood_limits)
    
    # Compare results
    assert result_gt.keys() == result_new.keys()
    assert result_gt['inputs'].keys() == result_new['inputs'].keys()
    
    # Compare each key in inputs
    for key in result_gt['inputs']:
        if isinstance(result_gt['inputs'][key], list):
            assert len(result_gt['inputs'][key]) == len(result_new['inputs'][key]) == 3, f"{key=}"
            for idx, (gt_item, new_item) in enumerate(zip(result_gt['inputs'][key], result_new['inputs'][key])):
                assert gt_item.shape == new_item.shape, f"{key=}, {idx=}"
                assert torch.allclose(gt_item, new_item), f"{key=}, {idx=}"
        else:
            assert torch.allclose(result_gt['inputs'][key], result_new['inputs'][key])
    
    # Compare labels and meta_info
    assert torch.allclose(result_gt['labels']['transform'], result_new['labels']['transform'])
    assert result_gt['meta_info'] == result_new['meta_info']


def test_geotransformer_collator():
    """Test geotransformer collator implementation against ground truth."""
    # Create dummy data
    list_data = create_dummy_geotransformer_data()
    num_stages = 3
    voxel_size = 0.1
    search_radius = 0.2
    neighbor_limits = [16, 16, 16]
    
    # Get results from both implementations
    result_gt = geotransformer_collate_fn_gt(list_data, num_stages, voxel_size, search_radius, neighbor_limits)
    result_new = geotransformer_collate_fn(list_data, num_stages, voxel_size, search_radius, neighbor_limits)
    
    # Compare results
    assert result_gt.keys() == result_new.keys()
    assert result_gt['inputs'].keys() == result_new['inputs'].keys()
    
    # Compare each key in inputs
    for key in result_gt['inputs']:
        if isinstance(result_gt['inputs'][key], list):
            assert len(result_gt['inputs'][key]) == len(result_new['inputs'][key])
            for idx, (gt_item, new_item) in enumerate(zip(result_gt['inputs'][key], result_new['inputs'][key])):
                assert gt_item.shape == new_item.shape, f"{key=}, {idx=}"
                assert torch.allclose(gt_item, new_item), f"{key=}, {idx=}"
        else:
            assert torch.allclose(result_gt['inputs'][key], result_new['inputs'][key])
    
    # Compare labels and meta_info
    assert torch.allclose(result_gt['labels']['transform'], result_new['labels']['transform'])
    assert result_gt['meta_info'] == result_new['meta_info']


def test_overlappredator_collator():
    """Test overlappredator collator implementation against ground truth."""
    # Create dummy data and config
    list_data = create_dummy_overlappredator_data()
    config = DummyConfig()
    neighborhood_limits = [16, 16, 16, 16, 16]
    
    # Get results from both implementations
    result_gt = overlappredator_collate_fn_gt(list_data, config, neighborhood_limits)
    result_new = overlappredator_collate_fn(list_data, config, neighborhood_limits)
    
    # Compare results
    assert result_gt.keys() == result_new.keys()
    assert result_gt['inputs'].keys() == result_new['inputs'].keys()
    
    # Compare each key in inputs
    for key in result_gt['inputs']:
        if isinstance(result_gt['inputs'][key], list):
            assert len(result_gt['inputs'][key]) == len(result_new['inputs'][key])
            for gt_item, new_item in zip(result_gt['inputs'][key], result_new['inputs'][key]):
                assert torch.allclose(gt_item, new_item)
        else:
            assert torch.allclose(result_gt['inputs'][key], result_new['inputs'][key])
    
    # Compare labels and meta_info
    assert torch.allclose(result_gt['labels']['rot'], result_new['labels']['rot'])
    assert torch.allclose(result_gt['labels']['trans'], result_new['labels']['trans'])
    assert torch.allclose(result_gt['labels']['src_pc'], result_new['labels']['src_pc'])
    assert torch.allclose(result_gt['labels']['tgt_pc'], result_new['labels']['tgt_pc'])
    assert torch.allclose(result_gt['labels']['correspondences'], result_new['labels']['correspondences'])
    assert result_gt['meta_info'] == result_new['meta_info']
