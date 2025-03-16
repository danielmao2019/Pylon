from typing import Dict, Any

# Default configuration for HCGMNet
hcgmnet_config: Dict[str, Any] = {
    'class': 'models.change_detection.hcgmnet.HCGMNet',
    'args': {
        'num_classes': 2,
    }
}
