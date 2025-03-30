import pytest
import torch
from utils.builders.builder import build_from_config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
from easydict import EasyDict


@pytest.fixture
def dummy_data():
    """Create dummy data matching the expected structure."""
    # Create dummy point clouds - using more points to match model requirements
    num_points = 256  # Points per cloud (increased from 100)
    num_stages = 4  # Match backbone num_stages
    num_neighbors = 16  # Number of neighbors per point

    # Create dummy multi-resolution data
    def create_dummy_multi_res_data(points, feats):
        pos_list = []
        lengths_list = []
        neighbors_list = []
        subsampling_list = []
        upsampling_list = []

        for i in range(num_stages):
            # Points and lengths for each stage
            num_points_at_stage = num_points // (2**i)
            pos_list.append(points[:num_points_at_stage])
            lengths_list.append(torch.tensor([num_points_at_stage]))

            # Create neighbor indices - ensure 2D shape [num_points_at_stage, num_neighbors]
            neighbors = torch.randint(0, num_points_at_stage, (num_points_at_stage, num_neighbors))
            neighbors_list.append(neighbors)

            # Create subsampling and upsampling indices
            if i < num_stages - 1:
                next_stage_points = num_points_at_stage // 2
                # For each point in the next stage, create neighbor indices from current stage
                subsampling = torch.randint(0, num_points_at_stage, (next_stage_points, num_neighbors))
                # For each point in current stage, store its nearest neighbor in next stage
                upsampling = torch.randint(0, next_stage_points, (num_points_at_stage, 1))
                subsampling_list.append(subsampling)
                upsampling_list.append(upsampling)

        return {
            'pos': pos_list,
            'feat': feats,
            'lengths': lengths_list,
            'neighbors': neighbors_list,
            'subsampling': subsampling_list,
            'upsampling': upsampling_list
        }

    # Create source and target point clouds
    src_points = torch.randn(num_points, 3)
    tgt_points = torch.randn(num_points, 3)
    src_feats = torch.randn(num_points, 1)  # Match backbone input_dim
    tgt_feats = torch.randn(num_points, 1)  # Match backbone input_dim

    # Create data dictionary with the expected structure
    data_dict = {
        'inputs': {
            'pc_1': create_dummy_multi_res_data(src_points, src_feats),
            'pc_2': create_dummy_multi_res_data(tgt_points, tgt_feats)
        },
        'labels': {
            'transform': torch.eye(4).unsqueeze(0),  # Identity transform
            'src_points': src_points,
            'tgt_points': tgt_points,
            'src_feats': src_feats,
            'tgt_feats': tgt_feats
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