"""Tests for enhanced BaseCollator with nested dictionary support and buffer_stack integration."""

from typing import Dict, Any, List, Tuple, Callable, Optional
import pytest
import torch
import numpy as np

from data.collators.base_collator import BaseCollator
from data.datasets.base_dataset import BaseDataset


class EnhancedDummyDataset(BaseDataset):
    """Enhanced dummy dataset for testing BaseCollator with complex nested structures."""
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 8, 'val': 4, 'test': 4}
    INPUT_NAMES = ['complex_data']
    LABEL_NAMES = ['target']
    SHA1SUM = None
    
    def __init__(self, structure_type: str = 'nested', **kwargs) -> None:
        """Initialize dataset with different structure types.
        
        Args:
            structure_type: Type of structure to generate ('nested', 'flat', 'mixed')
        """
        self.structure_type = structure_type
        super(EnhancedDummyDataset, self).__init__(**kwargs)
        
    def _init_annotations(self) -> None:
        """Initialize dummy annotations."""
        self.annotations = list(range(self.DATASET_SIZE[self.split]))
        
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load datapoint with specified structure type."""
        if self.structure_type == 'nested':
            inputs = self._create_nested_structure(idx)
        elif self.structure_type == 'flat':
            inputs = self._create_flat_structure(idx)
        elif self.structure_type == 'mixed':
            inputs = self._create_mixed_structure(idx)
        else:
            raise ValueError(f"Unknown structure_type: {self.structure_type}")
        
        labels = {
            'target': torch.tensor(idx, dtype=torch.long),
            'regression_target': torch.tensor(idx * 0.1, dtype=torch.float32)
        }
        
        meta_info = {
            'structure_type': self.structure_type,
            'complexity': 'high' if self.structure_type == 'nested' else 'low'
        }
        
        return inputs, labels, meta_info
    
    def _create_nested_structure(self, idx: int) -> Dict[str, Any]:
        """Create nested structure without deeply nested dicts mixed with lists."""
        return {
            'tensors': {
                'tensor_data': torch.randn(50, 10, dtype=torch.float32),
                'features': torch.randn(50, 5, dtype=torch.float32),
                'mask': torch.rand(50) > 0.5,
                'simple_tensor': torch.randn(20, dtype=torch.float32),
                'top_level_tensor': torch.randn(30, 3, dtype=torch.float32)
            },
            'scalars': {
                'scalar_data': idx * 2,
                'count': idx + 10,
                'multiplied': idx * 3
            },
            'strings': {
                'name': f'item_{idx}',
                'type': 'nested'
            },
            'lists': {
                'list_data': [idx, idx + 1, idx + 2],
                'multiplied_list': [idx * 3, idx * 4]
            }
        }
    
    def _create_flat_structure(self, idx: int) -> Dict[str, Any]:
        """Create flat structure."""
        return {
            'tensor1': torch.randn(100, dtype=torch.float32),
            'tensor2': torch.randn(100, 2, dtype=torch.float32),
            'scalar': idx,
            'boolean': idx % 2 == 0,
            'string': f'sample_{idx}'
        }
    
    def _create_mixed_structure(self, idx: int) -> Dict[str, Any]:
        """Create mixed flat and nested structure."""
        return {
            'flat_tensor': torch.randn(25, dtype=torch.float32),
            'flat_scalar': idx * 5,
            'nested': {
                'inner_tensor': torch.randn(25, 4, dtype=torch.float32),
                'inner_scalar': idx + 100,
                'inner_nested': {
                    'deep_tensor': torch.randn(10, dtype=torch.float32),
                    'deep_value': idx * 0.5
                }
            }
        }
    
    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> Optional['html.Div']:
        """Test dataset uses default display behavior."""
        return None


@pytest.fixture
def base_collator():
    """Create standard BaseCollator."""
    return BaseCollator()


@pytest.fixture
def custom_collator():
    """Create BaseCollator with custom collators."""
    def sum_collator(values: List[torch.Tensor]) -> torch.Tensor:
        """Sum tensors instead of stacking."""
        return torch.stack(values).sum(dim=0)
    
    def mean_collator(values: List[torch.Tensor]) -> torch.Tensor:
        """Average tensors instead of stacking."""
        return torch.stack(values).mean(dim=0)
    
    custom_collators = {
        'inputs': {
            'flat_tensor': sum_collator,
        }
    }
    
    return BaseCollator(collators=custom_collators)


def test_base_collator_nested_structure_handling():
    """Test BaseCollator with deeply nested structures."""
    dataset = EnhancedDummyDataset(
        structure_type='nested', 
        split='train', 
        use_cpu_cache=False,
        use_disk_cache=False
    )
    collator = BaseCollator()
    
    # Get sample datapoints
    datapoints = [dataset[i] for i in range(3)]
    
    # Test collation
    batch = collator(datapoints)
    
    # Check top-level structure
    assert 'inputs' in batch
    assert 'labels' in batch
    assert 'meta_info' in batch
    
    # Check nested structure preservation
    inputs = batch['inputs']
    assert 'tensors' in inputs
    assert 'scalars' in inputs
    assert 'strings' in inputs
    assert 'lists' in inputs
    
    # Check tensor group
    tensors = inputs['tensors']
    assert 'tensor_data' in tensors
    assert 'features' in tensors
    assert 'mask' in tensors
    assert 'simple_tensor' in tensors
    assert 'top_level_tensor' in tensors
    
    # Check tensor shapes (buffer_stack adds batch dimension as last dim)
    assert tensors['tensor_data'].shape == (50, 10, 3)  # batch_size=3
    assert tensors['features'].shape == (50, 5, 3)  # batch_size=3
    assert tensors['simple_tensor'].shape == (20, 3)  # batch_size=3
    assert tensors['top_level_tensor'].shape == (30, 3, 3)  # batch_size=3
    
    # Check that scalars become lists
    scalars = inputs['scalars']
    assert isinstance(scalars['scalar_data'], list)
    assert scalars['scalar_data'] == [0, 2, 4]  # idx * 2 for idx in [0, 1, 2]
    assert scalars['count'] == [10, 11, 12]  # idx + 10
    
    # Check nested lists (buffer_stack transposes nested lists)
    lists = inputs['lists']
    assert isinstance(lists['list_data'], list)
    assert len(lists['list_data']) == 3  # Length of original list
    assert lists['list_data'] == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    # multiplied_list gets transposed: [[0*3, 1*3, 2*3], [0*4, 1*4, 2*4]] = [[0,3,6], [0,4,8]]
    assert lists['multiplied_list'] == [[0, 3, 6], [0, 4, 8]]
    
    # Check strings
    strings = inputs['strings']
    assert strings['name'] == ['item_0', 'item_1', 'item_2']
    assert strings['type'] == ['nested', 'nested', 'nested']


def test_base_collator_flat_structure_handling():
    """Test BaseCollator with flat structures."""
    dataset = EnhancedDummyDataset(structure_type='flat', split='train', use_cpu_cache=False, use_disk_cache=False)
    collator = BaseCollator()
    
    datapoints = [dataset[i] for i in range(4)]
    batch = collator(datapoints)
    
    inputs = batch['inputs']
    
    # Check tensor handling (buffer_stack adds batch dimension as last dim)
    assert inputs['tensor1'].shape == (100, 4)  # batch_size=4
    assert inputs['tensor2'].shape == (100, 2, 4)  # batch_size=4
    
    # Check scalar handling
    assert inputs['scalar'] == [0, 1, 2, 3]
    assert inputs['boolean'] == [True, False, True, False]  # idx % 2 == 0
    assert inputs['string'] == ['sample_0', 'sample_1', 'sample_2', 'sample_3']


def test_base_collator_mixed_structure_handling():
    """Test BaseCollator with mixed flat and nested structures."""
    dataset = EnhancedDummyDataset(structure_type='mixed', split='train', use_cpu_cache=False, use_disk_cache=False)
    collator = BaseCollator()
    
    datapoints = [dataset[i] for i in range(2)]
    batch = collator(datapoints)
    
    inputs = batch['inputs']
    
    # Check flat components (buffer_stack adds batch dimension as last dim)
    assert inputs['flat_tensor'].shape == (25, 2)  # batch_size=2
    assert inputs['flat_scalar'] == [0, 5]  # idx * 5
    
    # Check nested components
    assert 'nested' in inputs
    nested = inputs['nested']
    assert nested['inner_tensor'].shape == (25, 4, 2)  # batch_size=2
    assert nested['inner_scalar'] == [100, 101]  # idx + 100
    
    # Check deeply nested
    assert 'inner_nested' in nested
    inner_nested = nested['inner_nested']
    assert inner_nested['deep_tensor'].shape == (10, 2)  # batch_size=2
    assert inner_nested['deep_value'] == [0.0, 0.5]  # idx * 0.5


def test_base_collator_custom_collators():
    """Test BaseCollator with custom collator functions."""
    dataset = EnhancedDummyDataset(structure_type='mixed', split='train', use_cpu_cache=False, use_disk_cache=False)
    
    def custom_concat_collator(values: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate tensors along first dimension."""
        return torch.cat(values, dim=0)
    
    def custom_max_collator(values: List[torch.Tensor]) -> torch.Tensor:
        """Take element-wise maximum."""
        return torch.stack(values).max(dim=0)[0]
    
    custom_collators = {
        'inputs': {
            'flat_tensor': custom_concat_collator,
        },
        'labels': {
            'regression_target': custom_max_collator
        }
    }
    
    collator = BaseCollator(collators=custom_collators)
    datapoints = [dataset[i] for i in range(3)]
    batch = collator(datapoints)
    
    # Check custom collation for flat_tensor (should be concatenated)
    assert batch['inputs']['flat_tensor'].shape == (75,)  # 3 * 25 = 75
    
    # Check custom collation for regression_target (should be max)
    # regression_target values are [0.0, 0.1, 0.2] for idx [0, 1, 2]
    expected_max = torch.tensor([0.0, 0.1, 0.2]).max()
    assert torch.allclose(batch['labels']['regression_target'], expected_max)
    
    # Check that other tensors use default collation (buffer_stack adds batch dim as last)
    assert batch['inputs']['nested']['inner_tensor'].shape == (25, 4, 3)  # batch_size=3


def test_base_collator_error_handling_with_context():
    """Test BaseCollator error handling with informative context."""
    collator = BaseCollator()
    
    # Create problematic data with mismatched tensor shapes
    problematic_data = [
        {
            'inputs': {
                'good_data': torch.randn(10, 5),
                'bad_data': torch.randn(10, 3)  # Different last dimension
            },
            'labels': {'target': torch.tensor(0)},
            'meta_info': {'idx': 0}
        },
        {
            'inputs': {
                'good_data': torch.randn(10, 5),
                'bad_data': torch.randn(10, 4)  # Different last dimension from first sample
            },
            'labels': {'target': torch.tensor(1)},
            'meta_info': {'idx': 1}
        }
    ]
    
    # Should raise RuntimeError with context information
    with pytest.raises(RuntimeError, match=r"Cannot collate values for key1='inputs', key2='bad_data'"):
        collator(problematic_data)
    
    # Test with nested problematic data
    nested_problematic_data = [
        {
            'inputs': {
                'nested': {
                    'inner': {
                        'tensor': torch.randn(5, 2)
                    }
                }
            },
            'labels': {'target': torch.tensor(0)},
            'meta_info': {'idx': 0}
        },
        {
            'inputs': {
                'nested': {
                    'inner': {
                        'tensor': torch.randn(5, 3)  # Different dimension
                    }
                }
            },
            'labels': {'target': torch.tensor(1)},
            'meta_info': {'idx': 1}
        }
    ]
    
    with pytest.raises(RuntimeError, match=r"Cannot collate values for key1='inputs', key2='nested'"):
        collator(nested_problematic_data)


def test_base_collator_empty_and_single_batch():
    """Test BaseCollator with empty and single-item batches."""
    collator = BaseCollator()
    
    # Test empty batch
    empty_result = collator([])
    assert empty_result == {}
    
    # Test single item batch
    dataset = EnhancedDummyDataset(structure_type='flat', split='train', use_cpu_cache=False, use_disk_cache=False)
    single_item = [dataset[0]]
    
    batch = collator(single_item)
    
    # Check structure is preserved
    assert 'inputs' in batch
    assert 'labels' in batch
    assert 'meta_info' in batch
    
    # Check that single items create correct structure
    # buffer_stack adds last dimension for batch, so (100,) becomes (100, 1) for single item
    assert batch['inputs']['tensor1'].shape == (100, 1)
    assert batch['inputs']['scalar'] == [0]  # List of length 1
    assert batch['meta_info']['idx'] == [0]  # idx is automatically added by BaseDataset


def test_base_collator_meta_info_preservation():
    """Test that BaseCollator correctly preserves meta_info without collation."""
    dataset = EnhancedDummyDataset(structure_type='nested', split='train', use_cpu_cache=False, use_disk_cache=False)
    collator = BaseCollator()
    
    datapoints = [dataset[i] for i in range(3)]
    batch = collator(datapoints)
    
    # Check meta_info is preserved as lists (not collated)
    meta_info = batch['meta_info']
    assert isinstance(meta_info['idx'], list)
    assert isinstance(meta_info['structure_type'], list)
    assert isinstance(meta_info['complexity'], list)
    
    # Check values
    assert meta_info['idx'] == [0, 1, 2]
    assert meta_info['structure_type'] == ['nested', 'nested', 'nested']
    assert meta_info['complexity'] == ['high', 'high', 'high']


def test_base_collator_dtype_preservation():
    """Test that BaseCollator preserves tensor dtypes through buffer_stack."""
    class DtypeDataset(BaseDataset):
        SPLIT_OPTIONS = ['train']
        DATASET_SIZE = {'train': 3}
        INPUT_NAMES = ['mixed_dtypes']
        LABEL_NAMES = ['target']
        SHA1SUM = None
        
        def _init_annotations(self):
            self.annotations = list(range(3))
            
        def _load_datapoint(self, idx):
            inputs = {
                'float32': torch.randn(10, dtype=torch.float32),
                'float64': torch.randn(10, dtype=torch.float64),
                'int32': torch.randint(0, 100, (10,), dtype=torch.int32),
                'int64': torch.randint(0, 100, (10,), dtype=torch.int64),
                'bool': torch.rand(10) > 0.5,
                'nested': {
                    'inner_float16': torch.randn(5, dtype=torch.float16),
                    'inner_uint8': torch.randint(0, 255, (5,), dtype=torch.uint8)
                }
            }
            labels = {'target': torch.tensor(idx, dtype=torch.long)}
            meta_info = {}
            return inputs, labels, meta_info
        
        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None
        ) -> Optional['html.Div']:
            """Test dataset uses default display behavior."""
            return None
    
    dataset = DtypeDataset(split='train', use_cpu_cache=False, use_disk_cache=False)
    collator = BaseCollator()
    
    datapoints = [dataset[i] for i in range(3)]
    batch = collator(datapoints)
    
    inputs = batch['inputs']
    
    # Check dtype preservation (shapes will have batch dimension as last dim)
    assert inputs['float32'].dtype == torch.float32
    assert inputs['float32'].shape == (10, 3)  # batch_size=3
    assert inputs['float64'].dtype == torch.float64
    assert inputs['int32'].dtype == torch.int32
    assert inputs['int64'].dtype == torch.int64
    assert inputs['bool'].dtype == torch.bool
    assert inputs['nested']['inner_float16'].dtype == torch.float16
    assert inputs['nested']['inner_uint8'].dtype == torch.uint8


def test_base_collator_large_nested_structures():
    """Test BaseCollator with large nested structures for performance."""
    class LargeNestedDataset(BaseDataset):
        SPLIT_OPTIONS = ['train']
        DATASET_SIZE = {'train': 2}
        INPUT_NAMES = ['large_nested']
        LABEL_NAMES = ['target']
        SHA1SUM = None
        
        def _init_annotations(self):
            self.annotations = [0, 1]
            
        def _load_datapoint(self, idx):
            # Create large nested structure
            inputs = {}
            for i in range(5):  # 5 top-level keys
                inputs[f'section_{i}'] = {}
                for j in range(10):  # 10 second-level keys each
                    inputs[f'section_{i}'][f'subsection_{j}'] = {
                        'data': torch.randn(100, 20, dtype=torch.float32),
                        'metadata': {
                            'id': i * 10 + j,
                            'valid': (i + j) % 2 == 0
                        }
                    }
            
            labels = {'target': torch.tensor(idx)}
            meta_info = {}
            return inputs, labels, meta_info
        
        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None
        ) -> Optional['html.Div']:
            """Test dataset uses default display behavior."""
            return None
    
    dataset = LargeNestedDataset(split='train', use_cpu_cache=False, use_disk_cache=False)
    collator = BaseCollator()
    
    datapoints = [dataset[0], dataset[1]]
    batch = collator(datapoints)
    
    # Check that large structure is handled correctly
    assert 'inputs' in batch
    
    # Spot check a few nested values
    for i in range(5):
        assert f'section_{i}' in batch['inputs']
        for j in range(10):
            section = batch['inputs'][f'section_{i}'][f'subsection_{j}']
            assert section['data'].shape == (100, 20, 2)  # batch_size=2
            assert section['metadata']['id'] == [i * 10 + j, i * 10 + j]
            expected_valid = [(i + j) % 2 == 0, (i + j) % 2 == 0]
            assert section['metadata']['valid'] == expected_valid


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_base_collator_varying_batch_sizes(batch_size):
    """Test BaseCollator with different batch sizes."""
    dataset = EnhancedDummyDataset(structure_type='nested', split='train', use_cpu_cache=False, use_disk_cache=False)
    collator = BaseCollator()
    
    # Take samples up to available dataset size
    num_samples = min(batch_size, dataset.DATASET_SIZE['train'])
    datapoints = [dataset[i] for i in range(num_samples)]
    
    batch = collator(datapoints)
    
    # Check that batch size is reflected in meta_info
    assert len(batch['meta_info']['idx']) == num_samples
    
    # Check tensor shapes have batch dimension as last dim
    inputs = batch['inputs']
    assert inputs['tensors']['tensor_data'].shape == (50, 10, num_samples)
    assert inputs['tensors']['top_level_tensor'].shape == (30, 3, num_samples)
    
    # Check list lengths match batch size
    assert len(inputs['scalars']['scalar_data']) == num_samples
    assert len(inputs['lists']['multiplied_list']) == 2  # Original list length preserved
    # But each element should be a list of length num_samples
    assert len(inputs['lists']['multiplied_list'][0]) == num_samples  # First element of original list
