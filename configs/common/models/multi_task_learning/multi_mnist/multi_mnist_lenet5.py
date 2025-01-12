import models


model_config_all_tasks = {
    'class': models.multi_task_learning.MultiMNIST_LeNet5,
    'args': {
        'return_shared_rep': True,
    },
}
