import pytest
import torch
from metrics.vision_2d.depth_estimation_metric import DepthEstimationMetric


@pytest.mark.parametrize("y_pred, y_true", [
    (
        torch.tensor([[
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]], dtype=torch.float32).permute(0, 3, 1, 2),  # [1, 1, 2, 2]
        torch.tensor([[
            [1.0, 2.0],
            [3.0, 4.0],
        ]], dtype=torch.float32),  # [1, 2, 2]
    ),
])
def test_depth_estimation_metric_call(y_pred, y_true):
    """Tests depth estimation metric computation for a single datapoint."""
    metric = DepthEstimationMetric()
    score = metric(y_pred, y_true)
    
    # Check structure
    assert set(score.keys()) == {'l1'}
    assert isinstance(score['l1'], torch.Tensor)
    assert score['l1'].ndim == 0  # Scalar tensor


@pytest.mark.parametrize("y_preds, y_trues", [
    (
        [
            torch.tensor([[
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]], dtype=torch.float32).permute(0, 3, 1, 2),  # [1, 1, 2, 2]
            torch.tensor([[
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]], dtype=torch.float32).permute(0, 3, 1, 2),  # [1, 1, 2, 2]
        ],
        [
            torch.tensor([[
                [1.0, 2.0],
                [3.0, 4.0],
            ]], dtype=torch.float32),  # [1, 2, 2]
            torch.tensor([[
                [1.0, 2.0],
                [3.0, 4.0],
            ]], dtype=torch.float32),  # [1, 2, 2]
        ],
    ),
])
def test_depth_estimation_metric_summarize(y_preds, y_trues):
    """Tests depth estimation metric summarization across multiple datapoints."""
    metric = DepthEstimationMetric()
    
    # Compute scores for each datapoint
    for y_pred, y_true in zip(y_preds, y_trues):
        metric(y_pred, y_true)
    
    # Summarize results
    result = metric.summarize()
    
    # Check structure
    assert set(result.keys()) == {'aggregated', 'per_datapoint'}
    
    # Check aggregated structure
    assert set(result['aggregated'].keys()) == {'l1'}
    
    # Check per_datapoint structure
    assert set(result['per_datapoint'].keys()) == {'l1'}
