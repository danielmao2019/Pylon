import pytest
from metrics.vision_2d.normal_estimation_metric import NormalEstimationMetric
import torch


@pytest.mark.parametrize("output, label, expected", [
    (
        torch.tensor([[[[1]], [[0]], [[0]]]], dtype=torch.float32),
        torch.tensor([[[[1]], [[1]], [[0]]]], dtype=torch.float32),
        torch.tensor(45, dtype=torch.float32),
    )
])
def test_normal_estimation_metric(output, label, expected) -> None:
    metric = NormalEstimationMetric()
    score = metric(y_pred=output, y_true=label)
    assert type(score) == dict and set(score.keys()) == set(['angle']), f"{score=}"
    assert torch.equal(score['angle'], expected)
