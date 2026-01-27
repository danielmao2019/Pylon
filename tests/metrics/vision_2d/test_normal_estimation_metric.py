import pytest
import torch
from metrics.vision_2d.normal_estimation_metric import NormalEstimationMetric


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
        torch.tensor([
            [
                [[0.9, 0.3], [0.5, 0.2]],  # x component
                [[0.1, 0.7], [0.6, 0.3]],  # y component
                [[0.2, 0.4], [0.8, 0.9]],  # z component
            ]
        ], dtype=torch.float32),  # [1, 3, 2, 2]
        torch.tensor([
            [
                [[1.0, 0.0], [0.0, 1.0]],  # x component
                [[0.0, 1.0], [0.0, 1.0]],  # y component
                [[0.0, 0.0], [1.0, 0.0]],  # z component
            ]
        ], dtype=torch.float32),  # [1, 3, 2, 2]
    ),
])
def test_normal_estimation_metric_call(y_pred, y_true):
    """Tests normal estimation metric structure for a single datapoint."""
    metric = NormalEstimationMetric()
    datapoint = create_datapoint(y_pred, y_true)
    score = metric(datapoint)

    # Check structure
    assert set(score.keys()) == {'angle'}
    assert isinstance(score['angle'], torch.Tensor)
    assert score['angle'].ndim == 0  # Scalar tensor


@pytest.mark.parametrize("y_pred, y_true, expected", [
    (
        torch.tensor([[[[1]], [[0]], [[0]]]], dtype=torch.float32),
        torch.tensor([[[[1]], [[1]], [[0]]]], dtype=torch.float32),
        torch.tensor(45, dtype=torch.float32),
    )
])
def test_normal_estimation_metric_value(y_pred, y_true, expected) -> None:
    """Tests normal estimation metric computation for a single datapoint."""
    metric = NormalEstimationMetric()
    datapoint = create_datapoint(y_pred, y_true)
    score = metric(datapoint)
    assert type(score) == dict and set(score.keys()) == set(['angle']), f"{score=}"
    assert torch.equal(score['angle'], expected)


@pytest.mark.parametrize("y_preds, y_trues", [
    (
        [
            torch.tensor([
                [
                    [[0.9, 0.3], [0.5, 0.2]],  # x component
                    [[0.1, 0.7], [0.6, 0.3]],  # y component
                    [[0.2, 0.4], [0.8, 0.9]],  # z component
                ]
            ], dtype=torch.float32),  # [1, 3, 2, 2]
            torch.tensor([
                [
                    [[0.8, 0.4], [0.7, 0.1]],  # x component
                    [[0.2, 0.6], [0.3, 0.8]],  # y component
                    [[0.3, 0.5], [0.6, 0.4]],  # z component
                ]
            ], dtype=torch.float32),  # [1, 3, 2, 2]
        ],
        [
            torch.tensor([
                [
                    [[1.0, 0.0], [0.0, 1.0]],  # x component
                    [[0.0, 1.0], [0.0, 1.0]],  # y component
                    [[0.0, 0.0], [1.0, 0.0]],  # z component
                ]
            ], dtype=torch.float32),  # [1, 3, 2, 2]
            torch.tensor([
                [
                    [[1.0, 0.0], [0.0, 1.0]],  # x component
                    [[0.0, 1.0], [0.0, 1.0]],  # y component
                    [[0.0, 0.0], [1.0, 0.0]],  # z component
                ]
            ], dtype=torch.float32),  # [1, 3, 2, 2]
        ],
    ),
])
def test_normal_estimation_metric_summarize(y_preds, y_trues):
    """Tests normal estimation metric summarization structure across multiple datapoints."""
    metric = NormalEstimationMetric()

    # Compute scores for each datapoint
    for idx, (y_pred, y_true) in enumerate(zip(y_preds, y_trues, strict=True)):
        datapoint = create_datapoint(y_pred, y_true, idx)
        metric(datapoint)

    # Summarize results
    result = metric.summarize()

    # Check structure
    assert set(result.keys()) == {'aggregated', 'per_datapoint'}

    # Check aggregated structure
    assert set(result['aggregated'].keys()) == {'angle'}
    assert isinstance(result['aggregated']['angle'], torch.Tensor)
    assert result['aggregated']['angle'].ndim == 0  # Scalar tensor

    # Check per_datapoint structure
    assert set(result['per_datapoint'].keys()) == {'angle'}
    assert isinstance(result['per_datapoint']['angle'], torch.Tensor)
    assert result['per_datapoint']['angle'].ndim == 1  # 1D tensor with one value per datapoint
    assert result['per_datapoint']['angle'].shape[0] == len(y_preds)  # Number of values matches number of datapoints
