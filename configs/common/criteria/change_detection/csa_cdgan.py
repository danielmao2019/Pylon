import criteria


criterion_cfg = {
    'class': criteria.vision_2d.change_detection.CSA_CDGAN_Criterion,
    'args': {
        'g_weight': 200,
        'd_weight': 1,
    }
}
