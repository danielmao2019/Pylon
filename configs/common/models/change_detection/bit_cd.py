import models


model_config = {
    'class': models.change_detection.BASE_Transformer,
    'args': {
        'input_nc':3,
        'output_nc':2,
        'token_len':4, 
        'resnet_stages_num':4,
        'with_pos':'learned',
        'enc_depth':1,
        'dec_depth':8,
    },
} 