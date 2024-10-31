import models


model_config_all_tasks = {
    'class': models.CelebA_ResNet18,
    'args': {
        'return_shared_rep': True,
    },
}
