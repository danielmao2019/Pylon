import models


model_config = {
    'class': models.change_detection.csa_cdgan.CSA_CDGAN,
    'args': {
        'generator_cfg': {
            'class': models.change_detection.csa_cdgan.CSA_CDGAN_Generator,
            'args': {
                'isize': 256,
                'nc': 3 * 2,
                'nz': 100,
                'ndf': 64,
                'n_extra_layers': 3,
            },
        },
        'discriminator_cfg': {
            'class': models.change_detection.csa_cdgan.CSA_CDGAN_Discriminator,
            'args': {
                'isize': 256,
                'nc': 2,
                'nz': 1,
                'ndf': 64,
                'n_extra_layers': 3,
            },
        },
    },
}
