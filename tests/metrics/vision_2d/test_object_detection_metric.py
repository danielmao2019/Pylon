from typing import Dict, List
import pytest
import torch
from metrics.vision_2d.object_detection_metric import ObjectDetectionMetric


@pytest.mark.parametrize("y_pred, y_true, areas, limits", [
    (
        {
            'labels': torch.tensor([[0, 1, 2]], dtype=torch.int64),
            'bboxes': torch.tensor([[
                [0.1, 0.1, 0.3, 0.3],  # small box
                [0.4, 0.4, 0.8, 0.8],  # medium box
                [0.6, 0.6, 0.9, 0.9],  # large box
            ]], dtype=torch.float32),
            'objectness': torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32),
        },
        {
            'bboxes': [torch.tensor([
                [0.1, 0.1, 0.3, 0.3],  # small box
                [0.4, 0.4, 0.8, 0.8],  # medium box
                [0.6, 0.6, 0.9, 0.9],  # large box
            ], dtype=torch.float32)],
            'areas': [torch.tensor([0.04, 0.16, 0.09], dtype=torch.float32)],
        },
        ['small', 'medium', 'large'],
        [3],
    ),
])
def test_object_detection_metric_call(y_pred, y_true, areas, limits):
    """Tests object detection metric computation for a single datapoint."""
    metric = ObjectDetectionMetric(areas=areas, limits=limits)
    scores: List[Dict[str, torch.Tensor]] = metric(y_pred, y_true)

    # Check structure
    for bbox_score in scores:
        for area in areas:
            for limit in limits:
                key = f"gt_overlaps_{area}@{limit}"
                assert key in bbox_score
                assert set(bbox_score[key].keys()) == {'per_bbox', 'AR', 'recalls', 'thresholds'}
                assert isinstance(bbox_score[key]['per_bbox'], torch.Tensor)
                assert isinstance(bbox_score[key]['AR'], torch.Tensor)
                assert isinstance(bbox_score[key]['recalls'], torch.Tensor)
                assert isinstance(bbox_score[key]['thresholds'], torch.Tensor)


@pytest.mark.parametrize("y_preds, y_trues, areas, limits", [
    (
        [
            {
                'labels': torch.tensor([[0, 1]], dtype=torch.int64),
                'bboxes': torch.tensor([[
                    [0.1, 0.1, 0.3, 0.3],  # small box
                    [0.4, 0.4, 0.8, 0.8],  # medium box
                ]], dtype=torch.float32),
                'objectness': torch.tensor([[0.9, 0.8]], dtype=torch.float32),
            },
            {
                'labels': torch.tensor([[0, 1]], dtype=torch.int64),
                'bboxes': torch.tensor([[
                    [0.1, 0.1, 0.3, 0.3],  # small box
                    [0.4, 0.4, 0.8, 0.8],  # medium box
                ]], dtype=torch.float32),
                'objectness': torch.tensor([[0.8, 0.9]], dtype=torch.float32),
            },
        ],
        [
            {
                'bboxes': [torch.tensor([
                    [0.1, 0.1, 0.3, 0.3],  # small box
                    [0.4, 0.4, 0.8, 0.8],  # medium box
                ], dtype=torch.float32)],
                'areas': [torch.tensor([0.04, 0.16], dtype=torch.float32)],
            },
            {
                'bboxes': [torch.tensor([
                    [0.1, 0.1, 0.3, 0.3],  # small box
                    [0.4, 0.4, 0.8, 0.8],  # medium box
                ], dtype=torch.float32)],
                'areas': [torch.tensor([0.04, 0.16], dtype=torch.float32)],
            },
        ],
        ['small'],
        [3],
    ),
])
def test_object_detection_metric_summarize(y_preds, y_trues, areas, limits):
    """Tests object detection metric summarization across multiple datapoints."""
    metric = ObjectDetectionMetric(areas=areas, limits=limits)
    
    # Compute scores for each datapoint
    for y_pred, y_true in zip(y_preds, y_trues):
        metric(y_pred, y_true)
    
    # Summarize results
    result = metric.summarize()
    
    # Check structure
    assert set(result.keys()) == {'aggregated', 'per_datapoint'}
    
    # Check aggregated structure
    for area in areas:
        for limit in limits:
            key = f"gt_overlaps_{area}@{limit}"
            assert key in result['aggregated']
            assert set(result['aggregated'][key].keys()) == {'AR', 'recalls', 'thresholds'}
    
    # Check per_datapoint structure
    for area in areas:
        for limit in limits:
            key = f"gt_overlaps_{area}@{limit}"
            assert key in result['per_datapoint']
            assert set(result['per_datapoint'][key].keys()) == {'AR', 'recalls', 'thresholds'}
