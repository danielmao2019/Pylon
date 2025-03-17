from typing import Dict, Any
import models


# Default configuration for HCGMNet
hcgmnet_config: Dict[str, Any] = {
    'class': models.change_detection.HCGMNet,
    'args': {
        'num_classes': 2,
    }
}
