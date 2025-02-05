import models


model_config = {
    'class': models.gans.GAN,
    'args': {
        'generator_cfg': {
            'class': models.gans.gan.Generator,
            'args': {
                'latent_dim': 128,
                'img_shape': (1, 28, 28),
            },
        },
        'discriminator_cfg': {
            'class': models.gans.gan.Discriminator,
            'args': {
                'img_shape': (1, 28, 28),
            },
        },
    },
}
