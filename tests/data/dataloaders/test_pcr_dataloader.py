"""Test PCRDataloader implementation."""

import os
import tempfile
import shutil
from typing import Dict, Any, List
import torch
import pytest
from data.dataloaders.pcr_dataloader import PCRDataloader


class DummyPCRDataloaderForTesting(PCRDataloader):
    """Concrete PCRDataloader for testing."""
    
    def _get_cache_version_dict(self, dataset, collator) -> Dict[str, Any]:
        """Get cache version dict for test PCR dataloader."""
        return {
            'dataloader_class': self.__class__.__name__,
            'dataset_version': dataset.get_cache_version_hash(),
            'collator_version': collator.get_cache_version_hash(),
            'test_version': 'v1'
        }


def collect_dataloader_outputs(dataloader) -> List[Dict[str, Any]]:
    """Helper function to collect all outputs from a dataloader."""
    outputs = []
    for batch in dataloader:
        # Convert tensors to CPU and detach for comparison
        batch_cpu = {}
        for key, value in batch.items():
            if key == 'inputs':
                batch_cpu[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        batch_cpu[key][sub_key] = {
                            k: v.cpu().detach().clone() if isinstance(v, torch.Tensor) else v 
                            for k, v in sub_value.items()
                        }
                    else:
                        batch_cpu[key][sub_key] = sub_value.cpu().detach().clone() if isinstance(sub_value, torch.Tensor) else sub_value
            elif key == 'labels':
                batch_cpu[key] = {
                    k: v.cpu().detach().clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in value.items()
                }
            elif key == 'meta_info':
                batch_cpu[key] = value  # meta_info is list of dicts, keep as-is
            else:
                batch_cpu[key] = value.cpu().detach().clone() if isinstance(value, torch.Tensor) else value
        
        outputs.append(batch_cpu)
    
    return outputs


def test_pcr_dataloader_cache_consistency(dummy_pcr_dataset, dummy_pcr_collator):
    """Test that PCRDataloader with cache enabled produces identical results to cache disabled."""
    
    try:
        
        # Set deterministic seed for DataLoader shuffling
        torch.manual_seed(456)
        
        # Create dataloader WITHOUT cache
        dataloader_no_cache = DummyPCRDataloaderForTesting(
            dataset=dummy_pcr_dataset,
            collator=dummy_pcr_collator,
            batch_size=1,
            shuffle=True,
            use_cpu_cache=False,
            use_disk_cache=False,
            max_cache_memory_percent=10.0,
            enable_cpu_validation=False,
            enable_disk_validation=False
        )
        
        # Collect outputs from dataloader without cache
        torch.manual_seed(456)  # Reset seed for consistent shuffling
        outputs_no_cache = collect_dataloader_outputs(dataloader_no_cache)
        
        # Create dataloader WITH cache
        torch.manual_seed(456)  # Reset seed for consistent shuffling
        dataloader_with_cache = DummyPCRDataloaderForTesting(
            dataset=dummy_pcr_dataset,
            collator=dummy_pcr_collator,
            batch_size=1,
            shuffle=True,
            use_cpu_cache=True,
            use_disk_cache=True,
            max_cache_memory_percent=10.0,
            enable_cpu_validation=True,
            enable_disk_validation=True
        )
        
        # Collect outputs from dataloader with cache (first pass - populates cache)
        torch.manual_seed(456)  # Reset seed for consistent shuffling
        outputs_with_cache_1st = collect_dataloader_outputs(dataloader_with_cache)
        
        # Collect outputs from dataloader with cache (second pass - uses cache)
        torch.manual_seed(456)  # Reset seed for consistent shuffling
        outputs_with_cache_2nd = collect_dataloader_outputs(dataloader_with_cache)
        
        # Verify all outputs are identical
        assert len(outputs_no_cache) == len(outputs_with_cache_1st) == len(outputs_with_cache_2nd), \
            f"Different number of batches: {len(outputs_no_cache)}, {len(outputs_with_cache_1st)}, {len(outputs_with_cache_2nd)}"
        
        for i, (no_cache, cache_1st, cache_2nd) in enumerate(zip(outputs_no_cache, outputs_with_cache_1st, outputs_with_cache_2nd)):
            # Check that all three outputs are identical
            _assert_batches_equal(no_cache, cache_1st, f"Batch {i}: no_cache vs cache_1st")
            _assert_batches_equal(cache_1st, cache_2nd, f"Batch {i}: cache_1st vs cache_2nd")
            _assert_batches_equal(no_cache, cache_2nd, f"Batch {i}: no_cache vs cache_2nd")
        
        print(f"✅ Test passed: All {len(outputs_no_cache)} batches are identical across cache/no-cache configurations")
        
        # Verify cache was actually used by checking collator call counts
        # (This would require modifying the collator to track calls, but the above test is sufficient)
        
    finally:
        # Cleanup the fixture's temp directory (this will also cleanup any cache subdirectories)
        if hasattr(dummy_pcr_dataset, 'data_root') and os.path.exists(dummy_pcr_dataset.data_root):
            shutil.rmtree(dummy_pcr_dataset.data_root)
        
        # Cleanup any cache directories that might have been created alongside data_root
        cache_dir = f"{dummy_pcr_dataset.data_root}_cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def _assert_batches_equal(batch1: Dict[str, Any], batch2: Dict[str, Any], context: str):
    """Assert that two batches are identical."""
    assert batch1.keys() == batch2.keys(), f"{context}: Different keys: {batch1.keys()} vs {batch2.keys()}"
    
    for key in batch1.keys():
        if key == 'inputs':
            _assert_inputs_equal(batch1[key], batch2[key], f"{context}, inputs")
        elif key == 'labels':
            _assert_labels_equal(batch1[key], batch2[key], f"{context}, labels")
        elif key == 'meta_info':
            _assert_meta_info_equal(batch1[key], batch2[key], f"{context}, meta_info")
        else:
            raise ValueError(f"Unexpected batch key: {key}")


def _assert_inputs_equal(inputs1: Dict[str, Any], inputs2: Dict[str, Any], context: str):
    """Assert that input dictionaries are identical."""
    assert inputs1.keys() == inputs2.keys(), f"{context}: Different input keys: {inputs1.keys()} vs {inputs2.keys()}"
    
    for key in inputs1.keys():
        pc1, pc2 = inputs1[key], inputs2[key]
        assert pc1.keys() == pc2.keys(), f"{context}, {key}: Different point cloud keys: {pc1.keys()} vs {pc2.keys()}"
        
        for pc_key in pc1.keys():
            if isinstance(pc1[pc_key], torch.Tensor):
                torch.testing.assert_close(
                    pc1[pc_key], pc2[pc_key],
                    msg=f"{context}, {key}.{pc_key}: Tensors not equal"
                )
            else:
                assert pc1[pc_key] == pc2[pc_key], f"{context}, {key}.{pc_key}: Values not equal: {pc1[pc_key]} vs {pc2[pc_key]}"


def _assert_labels_equal(labels1: Dict[str, torch.Tensor], labels2: Dict[str, torch.Tensor], context: str):
    """Assert that label dictionaries are identical."""
    assert labels1.keys() == labels2.keys(), f"{context}: Different label keys: {labels1.keys()} vs {labels2.keys()}"
    
    for key in labels1.keys():
        if isinstance(labels1[key], torch.Tensor):
            torch.testing.assert_close(
                labels1[key], labels2[key],
                msg=f"{context}, {key}: Label tensors not equal"
            )
        else:
            assert labels1[key] == labels2[key], f"{context}, {key}: Label values not equal: {labels1[key]} vs {labels2[key]}"


def _assert_meta_info_equal(meta1: List[Dict[str, Any]], meta2: List[Dict[str, Any]], context: str):
    """Assert that meta_info lists are identical."""
    assert len(meta1) == len(meta2), f"{context}: Different meta_info lengths: {len(meta1)} vs {len(meta2)}"
    
    for i, (m1, m2) in enumerate(zip(meta1, meta2)):
        assert m1.keys() == m2.keys(), f"{context}[{i}]: Different meta_info keys: {m1.keys()} vs {m2.keys()}"
        
        for key in m1.keys():
            assert m1[key] == m2[key], f"{context}[{i}], {key}: Meta info values not equal: {m1[key]} vs {m2[key]}"


if __name__ == "__main__":
    # For manual testing - import here to avoid issues
    import sys
    sys.path.append(os.path.dirname(__file__))
    from conftest import DummyPCRDataset, DummyPCRCollator
    
    temp_root = tempfile.mkdtemp(prefix="test_manual_")
    try:
        dataset = DummyPCRDataset(data_root=temp_root, split='train')
        collator = DummyPCRCollator()
        test_pcr_dataloader_cache_consistency(dataset, collator)
        print("Manual test passed!")
    finally:
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)