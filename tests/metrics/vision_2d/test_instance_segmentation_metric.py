import pytest
import torch
from metrics.vision_2d.instance_segmentation_metric import InstanceSegmentationMetric


def create_datapoint(y_pred, y_true, idx=0):
    """Helper function to create datapoint for tests."""
    return {
        'inputs': {},  # Empty for these tests
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': idx}
    }


@pytest.mark.parametrize("y_pred, y_true", [
    (
        torch.tensor([[
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]], dtype=torch.float32),
        torch.tensor([[
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]], dtype=torch.float32),
    ),
])
def test_instance_segmentation_metric_call(y_pred, y_true):
    """Tests instance segmentation metric computation for a single datapoint."""
    metric = InstanceSegmentationMetric(ignore_index=-1)
    datapoint = create_datapoint(y_pred, y_true)
    score = metric(datapoint)

    # Check structure
    assert set(score.keys()) == {'l1'}
    assert isinstance(score['l1'], torch.Tensor)
    assert score['l1'].ndim == 0  # Scalar tensor


@pytest.mark.parametrize("y_preds, y_trues", [
    (
        [
            torch.tensor([[
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ]], dtype=torch.float32),
            torch.tensor([[
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ]], dtype=torch.float32),
        ],
        [
            torch.tensor([[
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ]], dtype=torch.float32),
            torch.tensor([[
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ]], dtype=torch.float32),
        ],
    ),
])
def test_instance_segmentation_metric_summarize(y_preds, y_trues):
    """Tests instance segmentation metric summarization across multiple datapoints."""
    metric = InstanceSegmentationMetric(ignore_index=-1)

    # Compute scores for each datapoint
    for idx, (y_pred, y_true) in enumerate(zip(y_preds, y_trues, strict=True)):
        datapoint = create_datapoint(y_pred, y_true, idx)
        metric(datapoint)

    # Summarize results
    result = metric.summarize()

    # Check structure
    assert set(result.keys()) == {'aggregated', 'per_datapoint'}

    # Check aggregated structure
    assert set(result['aggregated'].keys()) == {'l1'}

    # Check per_datapoint structure
    assert set(result['per_datapoint'].keys()) == {'l1'}
