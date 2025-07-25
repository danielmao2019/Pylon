"""Tests for PASCALContextDataset cache version discrimination."""

import pytest
import tempfile
import os
import json
import numpy as np
import scipy.io
from PIL import Image
from data.datasets.multi_task_datasets.pascal_context_dataset import PASCALContextDataset


def create_dummy_pascal_context_structure(data_root: str) -> None:
    """Create a dummy PASCAL Context directory structure for testing."""
    # Create directory structure
    os.makedirs(os.path.join(data_root, 'ImageSets', 'Context'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'ImageSets', 'Parts'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'semseg', 'VOC12'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'semseg', 'pascal-context'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'human_parts'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'normals_distill'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'sal_distill'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'pascal-context', 'trainval'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'db_info'), exist_ok=True)
    
    # Define sample IDs
    train_ids = ['2008_000001', '2008_000002', '2008_000003']
    val_ids = ['2008_000004', '2008_000005']
    
    # Create split files
    with open(os.path.join(data_root, 'ImageSets', 'Context', 'train.txt'), 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    
    with open(os.path.join(data_root, 'ImageSets', 'Context', 'val.txt'), 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")
    
    # Create db_info files
    with open(os.path.join(data_root, 'db_info', 'pascal_part.json'), 'w') as f:
        json.dump({"1": {"head": 1, "torso": 2}, "15": {"head": 1, "torso": 2}}, f)
    
    with open(os.path.join(data_root, 'db_info', 'nyu_classes.json'), 'w') as f:
        json.dump(["person", "dog", "cat", "chair", "table"], f)
    
    with open(os.path.join(data_root, 'db_info', 'context_classes.json'), 'w') as f:
        json.dump({"person": 15, "dog": 12, "cat": 8, "chair": 9, "table": 11, "tvmonitor": 20}, f)
    
    # Create dummy data for all IDs
    all_ids = train_ids + val_ids
    for img_id in all_ids:
        # Create RGB image
        img = np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(data_root, 'JPEGImages', f"{img_id}.jpg"))
        
        # Create semantic segmentation (prefer VOC12)
        semantic = np.random.randint(0, 21, (375, 500), dtype=np.uint8)
        Image.fromarray(semantic).save(os.path.join(data_root, 'semseg', 'VOC12', f"{img_id}.png"))
        
        # Create human parts annotation
        # Structure: anno[0][0][1][0] is a list of objects
        # Each object has: [0] index, [1] category (15 for human), [2] mask, [3] parts
        obj_mask = np.random.randint(0, 2, (375, 500), dtype=np.uint8)
        parts_data = {
            'anno': [[{
                'field1': None,
                'field2': [(
                    np.array([[0]]),  # object index
                    np.array([[15]]),  # category (15 = human)
                    obj_mask,  # object mask
                    [[  # parts list
                        (np.array([['head']]), np.random.randint(0, 2, (375, 500), dtype=np.uint8)),
                        (np.array([['torso']]), np.random.randint(0, 2, (375, 500), dtype=np.uint8))
                    ]]
                )]
            }]]
        }
        scipy.io.savemat(os.path.join(data_root, 'human_parts', f"{img_id}.mat"), parts_data)
        
        # Create normal map
        normal = np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8)
        Image.fromarray(normal).save(os.path.join(data_root, 'normals_distill', f"{img_id}.png"))
        
        # Create saliency map
        saliency = np.random.randint(0, 255, (375, 500), dtype=np.uint8)
        Image.fromarray(saliency).save(os.path.join(data_root, 'sal_distill', f"{img_id}.png"))
        
        # Create label map for normal validation
        label_map = np.random.choice([8, 9, 11, 12, 15, 20], size=(375, 500))
        scipy.io.savemat(
            os.path.join(data_root, 'pascal-context', 'trainval', f"{img_id}.mat"),
            {'LabelMap': label_map}
        )


def test_pascal_context_dataset_version_discrimination():
    """Test that PASCALContextDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = PASCALContextDataset(
            data_root=temp_dir,
            split='train',
            num_human_parts=6,
            area_thres=0
        )
        dataset1b = PASCALContextDataset(
            data_root=temp_dir,
            split='train',
            num_human_parts=6,
            area_thres=0
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = PASCALContextDataset(
            data_root=temp_dir,
            split='val',  # Different
            num_human_parts=6,
            area_thres=0
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different num_human_parts should have different hash
        dataset3 = PASCALContextDataset(
            data_root=temp_dir,
            split='train',
            num_human_parts=14,  # Different
            area_thres=0
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different area_thres should have different hash
        dataset4 = PASCALContextDataset(
            data_root=temp_dir,
            split='train',
            num_human_parts=6,
            area_thres=100  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()
        
        # Different data_root should have different hash
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_pascal_context_structure(temp_dir2)
            dataset5 = PASCALContextDataset(
                data_root=temp_dir2,  # Different
                split='train',
                num_human_parts=6,
                area_thres=0
            )
            assert dataset1a.get_cache_version_hash() != dataset5.get_cache_version_hash()


def test_split_variants():
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        split_variants = ['train', 'val']
        
        datasets = []
        for split in split_variants:
            dataset = PASCALContextDataset(
                data_root=temp_dir,
                split=split,
                num_human_parts=6,
                area_thres=0
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All split variants should produce different hashes, got: {hashes}"


def test_num_human_parts_variants():
    """Test that different num_human_parts produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        parts_variants = [1, 4, 6, 14]
        
        datasets = []
        for num_parts in parts_variants:
            dataset = PASCALContextDataset(
                data_root=temp_dir,
                split='train',
                num_human_parts=num_parts,
                area_thres=0
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All num_human_parts variants should produce different hashes, got: {hashes}"


def test_area_threshold_variants():
    """Test that different area thresholds produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        threshold_variants = [0, 50, 100, 200]
        
        datasets = []
        for threshold in threshold_variants:
            dataset = PASCALContextDataset(
                data_root=temp_dir,
                split='train',
                num_human_parts=6,
                area_thres=threshold
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All area_thres variants should produce different hashes, got: {hashes}"


def test_different_annotation_data():
    """Test that different annotation data produces different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        dataset1 = PASCALContextDataset(
            data_root=temp_dir,
            split='train',
            num_human_parts=6,
            area_thres=0
        )
        
        # Add new image to train split
        new_id = '2008_000099'
        with open(os.path.join(temp_dir, 'ImageSets', 'Context', 'train.txt'), 'a') as f:
            f.write(f"{new_id}\n")
        
        # Create corresponding files
        img = np.zeros((375, 500, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, 'JPEGImages', f"{new_id}.jpg"))
        
        semantic = np.zeros((375, 500), dtype=np.uint8)
        Image.fromarray(semantic).save(os.path.join(temp_dir, 'semseg', 'VOC12', f"{new_id}.png"))
        
        # Minimal human parts data
        parts_data = {'anno': [[{'field1': None, 'field2': []}]]}
        scipy.io.savemat(os.path.join(temp_dir, 'human_parts', f"{new_id}.mat"), parts_data)
        
        Image.fromarray(img).save(os.path.join(temp_dir, 'normals_distill', f"{new_id}.png"))
        Image.fromarray(semantic).save(os.path.join(temp_dir, 'sal_distill', f"{new_id}.png"))
        
        dataset2 = PASCALContextDataset(
            data_root=temp_dir,
            split='train',
            num_human_parts=6,
            area_thres=0
        )
        
        # Should have different hashes due to different annotation list
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash():
    """Test that parameters inherited from BaseDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        base_args = {
            'data_root': temp_dir,
            'split': 'train',
            'num_human_parts': 6,
            'area_thres': 0,
        }
        
        # Test inherited parameters from BaseDataset
        parameter_variants = [
            ('initial_seed', 42),  # Different from default None
            ('cache_size', 1000),  # Different from default
        ]
        
        dataset1 = PASCALContextDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = PASCALContextDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_pascal_context_structure(temp_dir)
        
        datasets = []
        
        # Generate different dataset configurations
        for split in ['train', 'val']:
            for num_parts in [1, 6, 14]:
                for area_thres in [0, 100]:
                    for initial_seed in [None, 42]:
                        datasets.append(PASCALContextDataset(
                            data_root=temp_dir,
                            split=split,
                            num_human_parts=num_parts,
                            area_thres=area_thres,
                            initial_seed=initial_seed
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


