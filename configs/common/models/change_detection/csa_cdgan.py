import models


model_config = {
    'class': models.change_detection.CSA_CDGAN,
    'args': {
        'isize': 256,
        'nc': 3,
        'nz': 100,
        'ndf': 64,
        'n_extra_layers': 3,
    },
}
