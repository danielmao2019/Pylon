"""Tests for SyntheticTransformPCRDataset cache version discrimination."""

import tempfile
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset


# Create a concrete implementation for testing
class ConcreteSyntheticTransformPCRDataset(SyntheticTransformPCRDataset):
    """Concrete implementation for testing."""
    
    def _init_annotations(self) -> None:
        """Simple implementation for testing."""
        self.file_pair_annotations = [
            {'src_filepath': 'dummy1.ply', 'tgt_filepath': 'dummy1.ply'},
            {'src_filepath': 'dummy2.ply', 'tgt_filepath': 'dummy2.ply'},
        ]
    
    def _sample_crop(self, seed: int, file_idx: int) -> dict:
        """Sample crop parameters for testing."""
        return {
            'crop_seed': seed
        }
    
    def _build_crop(self, crop_params: dict) -> object:
        """Build crop transform for testing."""
        # Simple mock crop transform that does nothing
        class MockCrop:
            def __init__(self, seed):
                self.seed = seed
            
            def _call_single(self, pc_data, generator=None):
                return pc_data
        
        return MockCrop(crop_params['crop_seed'])
    
    def _apply_crop(self, crop_transform: object, pc_data: dict, crop_params: dict) -> dict:
        """Apply crop transform for testing."""
        # Just return the original data for testing
        return pc_data


def test_synthetic_transform_pcr_dataset_version_discrimination():
    """Test that SyntheticTransformPCRDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Same parameters should have same hash
        dataset1a = ConcreteSyntheticTransformPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        dataset1b = ConcreteSyntheticTransformPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different rotation_mag should have different hash
        dataset2 = ConcreteSyntheticTransformPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=30.0,  # Different
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different translation_mag should have different hash
        dataset3 = ConcreteSyntheticTransformPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.3,  # Different
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different matching_radius should have different hash
        dataset4 = ConcreteSyntheticTransformPCRDataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.1,  # Different
            split='train'
        )
        assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()
        
        # Different dataset_size should have different hash
        dataset5 = ConcreteSyntheticTransformPCRDataset(
            data_root=temp_dir,
            dataset_size=200,  # Different
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() != dataset5.get_cache_version_hash()
        


def test_all_parameters_affect_version_hash():
    """Test that all content-affecting parameters impact the version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_args = {
            'data_root': temp_dir,
            'dataset_size': 100,
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
            'split': 'train',
        }
        
        # Test each parameter individually
        parameter_variants = [
            ('dataset_size', 200),
            ('rotation_mag', 30.0),
            ('translation_mag', 0.3),
            ('matching_radius', 0.1),
            ('overlap_range', (0.2, 0.8)),  # Different from default
            ('min_points', 256),  # Different from default 512
        ]
        
        dataset1 = ConcreteSyntheticTransformPCRDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = ConcreteSyntheticTransformPCRDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets = []
        
        # Generate many different dataset configurations
        for dataset_size in [50, 100, 200]:
            for rotation_mag in [30.0, 45.0, 60.0]:
                for translation_mag in [0.3, 0.5, 0.7]:
                    datasets.append(ConcreteSyntheticTransformPCRDataset(
                        data_root=temp_dir,
                        dataset_size=dataset_size,
                        rotation_mag=rotation_mag,
                        translation_mag=translation_mag,
                        matching_radius=0.05,
                        split='train'
                    ))
        
        # Collect all hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        
        # Ensure all hashes are unique (no collisions)
        assert len(hashes) == len(set(hashes)), \
            f"Hash collision detected! Duplicate hashes found in: {hashes}"
        
        # Ensure all hashes are properly formatted
        for hash_val in hashes:
            assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
            assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"


