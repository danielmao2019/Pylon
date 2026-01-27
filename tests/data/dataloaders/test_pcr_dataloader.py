"""Test PCRDataloader implementation."""

import os
import shutil
from typing import Dict, Any, List
import torch
from data.dataloaders.pcr_dataloader import PCRDataloader
from utils.ops.dict_as_tensor import buffer_allclose


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
            num_workers=4,
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
            num_workers=4,
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

        for i, (no_cache, cache_1st, cache_2nd) in enumerate(zip(outputs_no_cache, outputs_with_cache_1st, outputs_with_cache_2nd, strict=True)):
            # Check that all three outputs are identical
            assert buffer_allclose(no_cache, cache_1st), f"Batch {i}: no_cache vs cache_1st not equal"
            assert buffer_allclose(cache_1st, cache_2nd), f"Batch {i}: cache_1st vs cache_2nd not equal"
            assert buffer_allclose(no_cache, cache_2nd), f"Batch {i}: no_cache vs cache_2nd not equal"

        print(f"✅ Test passed: All {len(outputs_no_cache)} batches are identical across cache/no-cache configurations")

    finally:
        # Cleanup cache directory if it was created
        cache_dir = f"{dummy_pcr_dataset.data_root}_cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
