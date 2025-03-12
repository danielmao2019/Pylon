"""
Configuration for the 3DCDNet model.

This model is based on the paper:
3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning
"""

from typing import Dict, Any

from models.change_detection import ThreeCDNet

model_config: Dict[str, Any] = {
    "model_class": ThreeCDNet,
    "init_args": {
        "num_classes": 2,  # Binary change detection (change/no-change)
        "input_dim": 3,    # XYZ coordinates
        "feature_dims": [64, 128, 256],
        "dropout": 0.1,
        "k_neighbors": 16,
        "sub_sampling_ratio": [4, 4, 4, 4]  # Downsampling ratios for each level
    }
}
