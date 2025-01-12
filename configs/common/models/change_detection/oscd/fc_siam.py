import models


model_config = {
    'class': models.change_detection.FullyConvolutionalSiameseNetwork,
    'args': {
        'arch': None,
        'in_channels': 3,
        'num_classes': 2,
    },
}
