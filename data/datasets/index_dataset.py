from typing import Tuple, Dict, List, Any, Optional
import torch
from data.datasets.base_dataset import BaseDataset


class IndexDataset(BaseDataset):
    """Dataset that simply returns the index as its output.
    
    This dataset is used as a wrapper for cached dataloaders where the actual
    dataset content is handled by the collator. The dataset itself only needs
    to provide indices for the collator to use as keys.
    """
    
    SPLIT_OPTIONS: List[str] = ['all']
    DATASET_SIZE: Optional[int] = None
    INPUT_NAMES: List[str] = ['index']
    LABEL_NAMES: List[str] = []
    SHA1SUM: Optional[str] = None

    def __init__(
        self,
        size: int,
        **kwargs
    ) -> None:
        """Initialize IndexDataset.
        
        Args:
            size: Number of indices this dataset should contain
            **kwargs: Additional arguments passed to BaseDataset
        """
        assert isinstance(size, int), f"size must be int, got {type(size)}"
        assert size > 0, f"size must be positive, got {size}"
        
        self._size = size
        # Set dataset size for BaseDataset validation
        self.DATASET_SIZE = size
        
        # Initialize with minimal required parameters for BaseDataset
        super().__init__(
            data_root=None,  # No data_root needed for index dataset
            split=None,      # No split needed
            **kwargs
        )

    def _init_annotations(self) -> None:
        """Initialize annotations as list of indices."""
        self.annotations = list(range(self._size))

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Not used - overridden by __getitem__."""
        raise NotImplementedError("This method should not be called as __getitem__ is overridden")

    def __getitem__(self, idx: int) -> int:
        """Return the index directly.
        
        Args:
            idx: Index to return
            
        Returns:
            The index itself
        """
        assert 0 <= idx < self._size, f"idx {idx} out of range [0, {self._size})"
        return idx

    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> Optional['html.Div']:
        """Display function for index dataset - not implemented."""
        return None