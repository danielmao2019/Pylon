import models


model_config = {
    'class': models.change_detection.SiameseKPConv,
    'args': {
        'in_channels': 3,  # Default input channels (RGB)
        'out_channels': 7,  # Urb3DCD dataset has 7 classes
        'point_influence': 0.05,  # Kernel point influence distance
        'down_channels': [32, 64, 128, 256],  # Channels in downsampling blocks
        'up_channels': [256, 128, 64, 32],  # Channels in upsampling blocks
        'bn_momentum': 0.02,  # Batch normalization momentum
        'dropout': 0.1,  # Dropout rate
        'conv_type': 'simple',  # Type of convolution blocks to use: 'simple', 'resnet', or 'kpdual'
    },
}
