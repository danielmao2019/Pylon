import pytest
from metrics.common import ConfusionMatrix
import torch


@pytest.mark.parametrize("y_pred, y_true, num_classes, expected", [
    (torch.tensor([
        [-1.1595,  1.4228, -2.3782,  0.0046, -0.3676,  1.9378, -1.6972,  0.3562, 0.4178,  0.8286],
        [ 0.2378, -0.1123, -0.7496,  0.5125, -0.3321,  1.7965, -0.0187, -1.9048, 0.4636,  1.7302],
        [ 0.7681,  1.6511, -2.1126, -0.3833, -1.2043,  0.4223, -0.3369,  0.8411, 0.8540,  0.3408],
        [ 0.4446,  2.1341, -0.0582,  0.2403, -0.5561,  0.1431, -0.2221,  1.0661, 0.1410,  1.8385],
    ], dtype=torch.float32), torch.tensor([7, 1, 4, 3], dtype=torch.int64), 10, {
        'class_tp': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'class_tn': torch.tensor([4, 1, 4, 3, 3, 2, 4, 3, 4, 4]),
        'class_fp': torch.tensor([0, 2, 0, 0, 0, 2, 0, 0, 0, 0]),
        'class_fn': torch.tensor([0, 1, 0, 1, 1, 0, 0, 1, 0, 0]),
    }),
    (torch.tensor([
        [-1.1595,  1.4228, -2.3782,  0.0046, -0.3676,  1.9378, -1.6972,  0.3562, 0.4178,  0.8286],
        [ 0.2378, -0.1123, -0.7496,  0.5125, -0.3321,  1.7965, -0.0187, -1.9048, 0.4636,  1.7302],
        [ 0.7681,  1.6511, -2.1126, -0.3833, -1.2043,  0.4223, -0.3369,  0.8411, 0.8540,  0.3408],
        [ 0.4446,  2.1341, -0.0582,  0.2403, -0.5561,  0.1431, -0.2221,  1.0661, 0.1410,  1.8385],
    ], dtype=torch.float32), torch.tensor([7, 5, 4, 3], dtype=torch.int64), 10, {
        'class_tp': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'class_tn': torch.tensor([4, 2, 4, 3, 3, 2, 4, 3, 4, 4]),
        'class_fp': torch.tensor([0, 2, 0, 0, 0, 1, 0, 0, 0, 0]),
        'class_fn': torch.tensor([0, 0, 0, 1, 1, 0, 0, 1, 0, 0]),
    }),
])
def test_confusion_matrix_call(y_pred, y_true, num_classes, expected) -> None:
    metric = ConfusionMatrix(num_classes=num_classes)
    score = metric(y_pred=y_pred, y_true=y_true)
    assert set(score.keys()) >= set(expected.keys())
    for key in expected:
        assert torch.equal(score[key], expected[key])


@pytest.mark.parametrize("y_pred_list, y_true_list, num_classes, expected", [
    ([
        torch.tensor([[-1.1595,  1.4228, -2.3782,  0.0046, -0.3676,  1.9378, -1.6972,  0.3562, 0.4178,  0.8286]], dtype=torch.float32),
        torch.tensor([[ 0.2378, -0.1123, -0.7496,  0.5125, -0.3321,  1.7965, -0.0187, -1.9048, 0.4636,  1.7302]], dtype=torch.float32),
        torch.tensor([[ 0.7681,  1.6511, -2.1126, -0.3833, -1.2043,  0.4223, -0.3369,  0.8411, 0.8540,  0.3408]], dtype=torch.float32),
        torch.tensor([[ 0.4446,  2.1341, -0.0582,  0.2403, -0.5561,  0.1431, -0.2221,  1.0661, 0.1410,  1.8385]], dtype=torch.float32),
        torch.tensor([[-1.1595,  1.4228, -2.3782,  0.0046, -0.3676,  1.9378, -1.6972,  0.3562, 0.4178,  0.8286]], dtype=torch.float32),
        torch.tensor([[ 0.2378, -0.1123, -0.7496,  0.5125, -0.3321,  1.7965, -0.0187, -1.9048, 0.4636,  1.7302]], dtype=torch.float32),
        torch.tensor([[ 0.7681,  1.6511, -2.1126, -0.3833, -1.2043,  0.4223, -0.3369,  0.8411, 0.8540,  0.3408]], dtype=torch.float32),
        torch.tensor([[ 0.4446,  2.1341, -0.0582,  0.2403, -0.5561,  0.1431, -0.2221,  1.0661, 0.1410,  1.8385]], dtype=torch.float32),
    ], [
        torch.tensor([7], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
        torch.tensor([7], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
    ], 10, {
        'class_tp': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'class_tn': torch.tensor([8, 3, 8, 6, 6, 4, 8, 6, 8, 8]),
        'class_fp': torch.tensor([0, 4, 0, 0, 0, 3, 0, 0, 0, 0]),
        'class_fn': torch.tensor([0, 1, 0, 2, 2, 0, 0, 2, 0, 0]),
        'class_accuracy': torch.tensor([8/8, 3/8, 8/8, 6/8, 6/8, 5/8, 8/8, 6/8, 8/8, 8/8], dtype=torch.float32),
        'class_precision': torch.tensor([float('nan'), 0/4, float('nan'), float('nan'), float('nan'), 1/4, float('nan'), float('nan'), float('nan'), float('nan')], dtype=torch.float32),
        'class_recall': torch.tensor([float('nan'), 0/1, float('nan'), 0/2, 0/2, 1/1, float('nan'), 0/2, float('nan'), float('nan')], dtype=torch.float32),
        'class_f1': torch.tensor([float('nan'), 0/5, float('nan'), 0/2, 0/2, 2/5, float('nan'), 0/2, float('nan'), float('nan')], dtype=torch.float32),
        'accuracy': torch.tensor(1/8, dtype=torch.float32),
    }),
])
def test_confusion_matrix_summary(y_pred_list, y_true_list, num_classes, expected) -> None:
    metric = ConfusionMatrix(num_classes=num_classes)
    for y_pred, y_true in zip(y_pred_list, y_true_list):
        metric(y_pred=y_pred, y_true=y_true)
    summary = metric.summarize(output_path=None)
    assert set(summary.keys()) == {'aggregated', 'per_datapoint'}

    aggregated = summary['aggregated']
    assert set(aggregated.keys()) >= set(expected.keys())
    for key in expected:
        assert torch.equal(aggregated[key].isnan(), expected[key].isnan())
        assert torch.equal(aggregated[key][~aggregated[key].isnan()], expected[key][~expected[key].isnan()])

    per_datapoint = summary['per_datapoint']
    assert set(per_datapoint.keys()) >= set(expected.keys())
