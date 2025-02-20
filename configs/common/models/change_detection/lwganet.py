import models
import models.change_detection.lwganet


model_config = {
    'class': models.change_detection.BaseNet_LWGANet_L2,
    'args': {
        'preptrained_path': '/pub7/yuchen/Pylon/models/change_detection/lwganet/lwganet_l2_e296.pth',
    },
}