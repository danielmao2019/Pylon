"""Mock dataset for testing."""
import numpy as np
from typing import Dict, Any, Tuple

class MockDataset:
    """A mock dataset for testing purposes."""
    
    def __init__(self, size: int = 10):
        """Initialize mock dataset.
        
        Args:
            size: Number of items in the dataset
        """
        self.size = size
        self.data = {
            i: {
                'image1': np.random.rand(64, 64, 3),
                'image2': np.random.rand(64, 64, 3),
                'mask': np.random.randint(0, 2, (64, 64), dtype=np.uint8)
            } for i in range(size)
        }
        self.class_labels = {0: 'no_change', 1: 'change'}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.size}")
        return self.data[idx]

    def get_item_size(self, idx: int) -> Tuple[int, int]:
        """Get the size of images at given index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (height, width)
        """
        item = self.data[idx]
        return item['image1'].shape[:2] 