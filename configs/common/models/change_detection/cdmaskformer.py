import models


model_cfg = {
    'class': models.change_detection.cdmaskformer.CDMaskFormer,
    'args': {
        'backbone_cfg': {
            'class': models.change_detection.cdmaskformer.CDMaskFormerBackbone,
            'args': {},
        },
        'head_cfg': {
            'class': models.change_detection.cdmaskformer.CDMaskFormerHead,
            'args': {
                'channels': [64, 128, 192, 256],
                'num_classes': 1,
                'num_queries': 5,
                'dec_layers': 14,
            },
        },
    },
}
