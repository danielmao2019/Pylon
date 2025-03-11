from models.change_detection.ppsl.ppsl_model import PPSLModel
import torch


def test_ppsl_model() -> None:
    model = PPSLModel().to('cuda')
    img_1 = torch.zeros(size=(4, 3, 224, 224), device='cuda')
    img_2 = torch.zeros(size=(4, 3, 224, 224), device='cuda')
    outputs = model(img_1, img_2)
    semantic_out = outputs['semantic_map']
    assert semantic_out.shape == (4, 2, 56, 56)
    change_out = outputs['change_map']
    assert change_out.shape == (4, 2, 224, 224)
