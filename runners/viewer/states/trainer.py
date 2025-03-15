import numpy as np


class TrainingState:
    def __init__(self):
        self.current_iteration = 0
        self.max_iteration = 100  # TODO: Get from actual training data
        self.class_colors = self._get_default_colors()
        
        # Temporary: Create dummy data for testing
        self._create_dummy_data()
    
    def _get_default_colors(self):
        """Return default color mapping for visualization."""
        return {
            0: [0, 0, 0],      # Background (black)
            1: [255, 0, 0],    # Change class 1 (red)
            2: [0, 255, 0],    # Change class 2 (green)
            3: [0, 0, 255],    # Change class 3 (blue)
        }
    
    def _create_dummy_data(self):
        """Create dummy data for testing."""
        # This is temporary and will be replaced with actual data loading
        self.dummy_data = {
            'input1': np.random.rand(1, 3, 64, 64),
            'input2': np.random.rand(1, 3, 64, 64),
            'pred': np.random.randint(0, 4, (1, 64, 64)),
            'gt': np.random.randint(0, 4, (1, 64, 64))
        }
    
    def get_current_data(self):
        """Get data for current iteration."""
        # TODO: Implement actual data loading
        return self.dummy_data
    
    def next_iteration(self):
        """Move to next iteration if available."""
        if self.current_iteration < self.max_iteration - 1:
            self.current_iteration += 1
            self._create_dummy_data()  # Temporary: Create new dummy data
        return self.current_iteration
    
    def prev_iteration(self):
        """Move to previous iteration if available."""
        if self.current_iteration > 0:
            self.current_iteration -= 1
            self._create_dummy_data()  # Temporary: Create new dummy data
        return self.current_iteration
    
    def class_to_rgb(self, class_indices):
        """Convert class indices to RGB values."""
        rgb = np.zeros((*class_indices.shape, 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            mask = class_indices == class_idx
            rgb[mask] = color
        return rgb
