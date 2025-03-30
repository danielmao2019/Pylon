import pytest
import torch
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
from easydict import EasyDict


@pytest.fixture
def dummy_data():
    """Create dummy data matching the expected structure."""
    # Create dummy point clouds
    num_points = 256  # Points per cloud

    # Create source and target point clouds
    src_points = torch.randn(num_points, 3)
    tgt_points = torch.randn(num_points, 3)
    src_feats = torch.randn(num_points, 1)  # Match backbone input_dim
    tgt_feats = torch.randn(num_points, 1)  # Match backbone input_dim

    # Create data dictionary with the expected structure
    data_dict = {
        'inputs': {
            'src_pc': {
                'pos': [src_points],  # List of point clouds at different resolutions
                'feat': src_feats
            },
            'tgt_pc': {
                'pos': [tgt_points],  # List of point clouds at different resolutions
                'feat': tgt_feats
            }
        }
    }

    # Move all tensors to CUDA
    def move_to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif isinstance(obj, dict):
            return {k: move_to_cuda(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cuda(item) for item in obj]
        return obj

    data_dict = move_to_cuda(data_dict)
    return data_dict


def test_overlappredator_forward(dummy_data):
    """Test the forward pass of the OverlapPredator model."""
    # Initialize model using the builder
    model = build_from_config(model_cfg)
    model = model.cuda()  # Move model to CUDA
    model.eval()  # Set to evaluation mode

    # Run forward pass
    with torch.no_grad():
        feats_f, scores_overlap, scores_saliency = model(dummy_data['inputs'])

    # Validate output structure and shapes
    # 1. Check feature outputs
    assert feats_f.shape[1] == model_cfg['args']['final_feats_dim']
    assert feats_f.shape[0] == dummy_data['inputs']['src_pc']['pos'][0].shape[0] + \
                             dummy_data['inputs']['tgt_pc']['pos'][0].shape[0]

    # 2. Check score outputs
    assert scores_overlap.shape[0] == dummy_data['inputs']['src_pc']['pos'][0].shape[0] + \
                                    dummy_data['inputs']['tgt_pc']['pos'][0].shape[0]
    assert scores_saliency.shape[0] == dummy_data['inputs']['src_pc']['pos'][0].shape[0] + \
                                     dummy_data['inputs']['tgt_pc']['pos'][0].shape[0]

    # 3. Check value ranges
    assert torch.all(scores_overlap >= 0) and torch.all(scores_overlap <= 1)
    assert torch.all(scores_saliency >= 0) and torch.all(scores_saliency <= 1)

    # 4. Check feature normalization
    assert torch.allclose(torch.norm(feats_f, p=2, dim=1), torch.ones_like(torch.norm(feats_f, p=2, dim=1)))