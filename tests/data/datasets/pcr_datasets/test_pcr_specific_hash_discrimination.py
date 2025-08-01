"""PCR-specific hash discrimination tests.

This module contains tests specific to PCR dataset parameters that don't
apply to other datasets, particularly rotation_mag, translation_mag, and
other PCR-specific parameters.
"""

import pytest
from utils.builders.builder import build_from_config


@pytest.mark.parametrize('pcr_dataset_class', [
    'BiTemporalPCRDataset',
    'SingleTemporalPCRDataset',
])
def test_pcr_rotation_magnitude_discrimination(pcr_dataset_class, threedmatch_data_root):
    """Test PCR-specific rotation magnitude parameter discrimination."""
    # Import the appropriate class
    if pcr_dataset_class == 'BiTemporalPCRDataset':
        from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
        dataset_class = BiTemporalPCRDataset
    else:
        from data.datasets.pcr_datasets.single_temporal_pcr_dataset import SingleTemporalPCRDataset
        dataset_class = SingleTemporalPCRDataset
    
    base_config = {
        'data_root': f'{threedmatch_data_root}/test/7-scenes-redkitchen',
        'dataset_size': 10,  # Small for testing
        'translation_mag': 0.5,
        'matching_radius': 0.1,
        'min_points': 500
    }
    
    # Test rotation magnitude variation
    config1 = {
        'class': dataset_class,
        'args': {**base_config, 'rotation_mag': 45.0}
    }
    config2 = {
        'class': dataset_class,
        'args': {**base_config, 'rotation_mag': 90.0}
    }
    
    dataset1 = build_from_config(config1)
    dataset2 = build_from_config(config2)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 != hash2, f"{pcr_dataset_class}: Different rotation_mag should produce different hashes: {hash1} == {hash2}"


@pytest.mark.parametrize('pcr_dataset_class', [
    'BiTemporalPCRDataset',
    'SingleTemporalPCRDataset',
])
def test_pcr_translation_magnitude_discrimination(pcr_dataset_class, threedmatch_data_root):
    """Test PCR-specific translation magnitude parameter discrimination."""
    # Import the appropriate class
    if pcr_dataset_class == 'BiTemporalPCRDataset':
        from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
        dataset_class = BiTemporalPCRDataset
    else:
        from data.datasets.pcr_datasets.single_temporal_pcr_dataset import SingleTemporalPCRDataset
        dataset_class = SingleTemporalPCRDataset
    
    base_config = {
        'data_root': f'{threedmatch_data_root}/test/7-scenes-redkitchen',
        'dataset_size': 10,  # Small for testing
        'rotation_mag': 45.0,
        'matching_radius': 0.1,
        'min_points': 500
    }
    
    # Test translation magnitude variation
    config1 = {
        'class': dataset_class,
        'args': {**base_config, 'translation_mag': 0.5}
    }
    config2 = {
        'class': dataset_class,
        'args': {**base_config, 'translation_mag': 1.0}
    }
    
    dataset1 = build_from_config(config1)
    dataset2 = build_from_config(config2)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 != hash2, f"{pcr_dataset_class}: Different translation_mag should produce different hashes: {hash1} == {hash2}"


def test_pcr_comprehensive_parameter_combinations(threedmatch_data_root):
    """Test comprehensive PCR parameter combinations produce unique hashes."""
    from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
    
    base_config = {
        'data_root': f'{threedmatch_data_root}/test/7-scenes-redkitchen',
        'dataset_size': 5,  # Very small for testing
    }
    
    test_combinations = [
        {'rotation_mag': 45.0, 'translation_mag': 0.5, 'matching_radius': 0.1},
        {'rotation_mag': 90.0, 'translation_mag': 0.5, 'matching_radius': 0.1},
        {'rotation_mag': 45.0, 'translation_mag': 1.0, 'matching_radius': 0.1},
        {'rotation_mag': 45.0, 'translation_mag': 0.5, 'matching_radius': 0.05},
    ]
    
    hashes = []
    for combination in test_combinations:
        config = {
            'class': BiTemporalPCRDataset,
            'args': {**base_config, **combination}
        }
        dataset = build_from_config(config)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for {combination}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we have unique hashes
    assert len(hashes) == len(test_combinations), f"Expected {len(test_combinations)} unique hashes, got {len(hashes)}"
