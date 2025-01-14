import models


model_config = {
    'class': models.change_detection.FullyConvolutionalSiameseNetwork,
    'args': {
        'arch': None,
        'in_channels': None,
        'num_classes': 2,
    },
}
