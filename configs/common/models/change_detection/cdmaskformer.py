import models


model_config = {
    'class': models.change_detection.CDMaskFormer,
    'args': {
        'backbone_name': 'resnet50',
        'backbone_args': {
            'pretrained': True
        },
        'cdmask_args': {
            'num_classes': 1,
            'num_queries': 10,
            'dec_layers': 6
        }
    }
}
