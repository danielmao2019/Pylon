import torch
from metrics.vision_3d import CorrespondencePrecisionRecall
import numpy as np


def compute_precision_recall_numpy(predicted_matches, ground_truth_matches):
    """Original numpy implementation of precision and recall."""
    # Convert to sets for easier comparison
    predicted_set = set(predicted_matches)
    ground_truth_set = set(ground_truth_matches)

    # True positives are matches that are in both predicted and ground truth sets
    true_positives = len(predicted_set.intersection(ground_truth_set))

    # Precision = TP / (TP + FP)
    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0

    # F1 score is the harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def create_correspondence_tensor(source_points, target_points):
    """Utility function to create correspondence tensors."""
    assert len(source_points) == len(target_points)
    correspondences = []
    for src, tgt in zip(source_points, target_points):
        correspondences.append([src, tgt])
    return torch.tensor(correspondences, dtype=torch.float32)


def test_perfect_match():
    """Test with perfect correspondence matches."""
    # Create some correspondence pairs (perfectly matched)
    source_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ]
    target_points = [
        [0.01, 0.01, 0.01],  # Slightly offset but within threshold
        [1.01, 0.01, 0.01],
        [0.01, 1.01, 0.01],
        [1.01, 1.01, 0.01]
    ]

    # Create predicted and ground truth correspondence tensors
    # For this test, predicted == ground truth (perfect match)
    y_pred = create_correspondence_tensor(source_points, target_points)
    y_true = create_correspondence_tensor(source_points, target_points)

    # Create CorrespondencePrecisionRecall instance
    precision_recall = CorrespondencePrecisionRecall(threshold=0.02)

    # Compute precision, recall, and F1 score using the metric class
    metric_result = precision_recall(y_pred, y_true)

    # For perfect match, precision, recall, and F1 score should all be 1.0
    assert abs(metric_result["precision"].item() - 1.0) < 1e-5, f"Expected precision 1.0, got {metric_result['precision'].item()}"
    assert abs(metric_result["recall"].item() - 1.0) < 1e-5, f"Expected recall 1.0, got {metric_result['recall'].item()}"
    assert abs(metric_result["f1_score"].item() - 1.0) < 1e-5, f"Expected F1 score 1.0, got {metric_result['f1_score'].item()}"


def test_partial_match():
    """Test with partially matched correspondences."""
    # Create predicted correspondence pairs
    pred_source_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ]
    pred_target_points = [
        [0.01, 0.01, 0.01],  # Matches ground truth
        [1.01, 0.01, 0.01],  # Matches ground truth
        [0.01, 1.01, 0.01],  # Matches ground truth
        [1.5, 1.5, 0.01]     # Does not match any ground truth
    ]

    # Create ground truth correspondence pairs
    gt_source_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0]      # Does not match any prediction
    ]
    gt_target_points = [
        [0.01, 0.01, 0.01],
        [1.01, 0.01, 0.01],
        [0.01, 1.01, 0.01],
        [0.51, 0.51, 0.01]
    ]

    # Create correspondence tensors
    y_pred = create_correspondence_tensor(pred_source_points, pred_target_points)
    y_true = create_correspondence_tensor(gt_source_points, gt_target_points)

    # Create CorrespondencePrecisionRecall instance
    precision_recall = CorrespondencePrecisionRecall(threshold=0.02)

    # Compute precision, recall, and F1 score using the metric class
    metric_result = precision_recall(y_pred, y_true)

    # Expected values for precision, recall based on the test data:
    # - 3 out of 4 predictions match ground truth → precision = 0.75
    # - 3 out of 4 ground truth matches are found → recall = 0.75
    # - F1 score = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    expected_precision = 0.75
    expected_recall = 0.75
    expected_f1 = 0.75

    assert abs(metric_result["precision"].item() - expected_precision) < 1e-5, \
        f"Expected precision {expected_precision}, got {metric_result['precision'].item()}"
    assert abs(metric_result["recall"].item() - expected_recall) < 1e-5, \
        f"Expected recall {expected_recall}, got {metric_result['recall'].item()}"
    assert abs(metric_result["f1_score"].item() - expected_f1) < 1e-5, \
        f"Expected F1 score {expected_f1}, got {metric_result['f1_score'].item()}"


def test_no_match():
    """Test with no matching correspondences."""
    # Create completely different correspondence sets
    pred_source_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ]
    pred_target_points = [
        [0.01, 0.01, 0.01],
        [1.01, 0.01, 0.01]
    ]

    gt_source_points = [
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0]
    ]
    gt_target_points = [
        [5.01, 5.01, 5.01],
        [6.01, 6.01, 6.01]
    ]

    # Create correspondence tensors
    y_pred = create_correspondence_tensor(pred_source_points, pred_target_points)
    y_true = create_correspondence_tensor(gt_source_points, gt_target_points)

    # Create CorrespondencePrecisionRecall instance
    precision_recall = CorrespondencePrecisionRecall(threshold=0.02)

    # Compute precision, recall, and F1 score using the metric class
    metric_result = precision_recall(y_pred, y_true)

    # For no match, precision, recall, and F1 score should all be 0.0
    assert abs(metric_result["precision"].item() - 0.0) < 1e-5, f"Expected precision 0.0, got {metric_result['precision'].item()}"
    assert abs(metric_result["recall"].item() - 0.0) < 1e-5, f"Expected recall 0.0, got {metric_result['recall'].item()}"
    assert abs(metric_result["f1_score"].item() - 0.0) < 1e-5, f"Expected F1 score 0.0, got {metric_result['f1_score'].item()}"


def test_threshold_effect():
    """Test the effect of different thresholds on precision and recall."""
    # Create sample point clouds
    source_np = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    target_np = np.array([
        [0.01, 0.01, 0.01],  # Within threshold 0.05
        [1.03, 0.01, 0.01],  # Outside threshold 0.05
        [0.01, 1.01, 0.01],  # Within threshold 0.05
        [1.01, 1.01, 0.01],  # Within threshold 0.05
        [0.5, 0.5, 0.5]      # Far from any source point
    ])

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Test different thresholds
    thresholds = [0.02, 0.05, 0.1]
    expected_results = [
        {"precision": 0.75, "recall": 0.75},  # 3 out of 4 points within 0.02
        {"precision": 1.0, "recall": 1.0},    # All 4 points within 0.05
        {"precision": 1.0, "recall": 1.0}     # All 4 points within 0.1
    ]

    for threshold, expected in zip(thresholds, expected_results):
        # Create PrecisionRecall instance with current threshold
        precision_recall = CorrespondencePrecisionRecall(threshold=threshold)

        # Compute precision and recall using the metric class
        metric_result = precision_recall(source_torch, target_torch)

        # Check that the results match expected values
        assert isinstance(metric_result, dict), f"{type(metric_result)=}"
        assert metric_result.keys() == {'precision', 'recall'}, f"{metric_result.keys()=}"
        assert abs(metric_result['precision'].item() - expected['precision']) < 1e-5, \
            f"Threshold {threshold}: Expected precision {expected['precision']}, got {metric_result['precision'].item()}"
        assert abs(metric_result['recall'].item() - expected['recall']) < 1e-5, \
            f"Threshold {threshold}: Expected recall {expected['recall']}, got {metric_result['recall'].item()}"


def test_edge_cases():
    """Test edge cases like empty correspondences."""
    # Test with empty correspondences
    empty_pred = torch.zeros((0, 2, 3), dtype=torch.float32)
    empty_true = torch.zeros((0, 2, 3), dtype=torch.float32)

    # Create CorrespondencePrecisionRecall instance
    precision_recall = CorrespondencePrecisionRecall(threshold=0.02)

    # Compute precision, recall, and F1 score using the metric class
    metric_result = precision_recall(empty_pred, empty_true)

    # For empty correspondences, precision, recall, and F1 score should all be 0.0
    assert abs(metric_result["precision"].item() - 0.0) < 1e-5, f"Expected precision 0.0, got {metric_result['precision'].item()}"
    assert abs(metric_result["recall"].item() - 0.0) < 1e-5, f"Expected recall 0.0, got {metric_result['recall'].item()}"
    assert abs(metric_result["f1_score"].item() - 0.0) < 1e-5, f"Expected F1 score 0.0, got {metric_result['f1_score'].item()}"

    # Test with empty ground truth but non-empty predictions
    non_empty_pred = create_correspondence_tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.01, 0.01, 0.01], [1.01, 0.01, 0.01]]
    )

    # Compute precision, recall, and F1 score using the metric class
    metric_result = precision_recall(non_empty_pred, empty_true)

    # When ground truth is empty but predictions are not, precision should be 0.0
    assert abs(metric_result["precision"].item() - 0.0) < 1e-5, f"Expected precision 0.0, got {metric_result['precision'].item()}"
    assert abs(metric_result["recall"].item() - 0.0) < 1e-5, f"Expected recall 0.0, got {metric_result['recall'].item()}"
    assert abs(metric_result["f1_score"].item() - 0.0) < 1e-5, f"Expected F1 score 0.0, got {metric_result['f1_score'].item()}"


def test_precision_recall_batch():
    """Test precision and recall with batched inputs."""
    # Create batch of correspondences
    batch_size = 3

    # Create predicted correspondence pairs for each batch
    pred_source_points_batch = [
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ]
    ]

    pred_target_points_batch = [
        [
            [0.01, 0.01, 0.01],  # Matches ground truth
            [1.01, 0.01, 0.01],  # Matches ground truth
            [0.01, 1.01, 0.01],  # Matches ground truth
            [1.5, 1.5, 0.01]     # Does not match any ground truth
        ],
        [
            [0.01, 0.01, 0.01],  # Matches ground truth
            [1.01, 0.01, 0.01],  # Matches ground truth
            [0.01, 1.01, 0.01],  # Matches ground truth
            [1.01, 1.01, 0.01]   # Matches ground truth (perfect match)
        ],
        [
            [5.0, 5.0, 5.0],     # No match
            [6.0, 6.0, 6.0],     # No match
            [7.0, 7.0, 7.0],     # No match
            [8.0, 8.0, 8.0]      # No match
        ]
    ]

    # Create ground truth correspondence pairs for each batch
    gt_source_points_batch = [
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]      # Does not match any prediction
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]      # Matches prediction
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]      # No match with predictions
        ]
    ]

    gt_target_points_batch = [
        [
            [0.01, 0.01, 0.01],
            [1.01, 0.01, 0.01],
            [0.01, 1.01, 0.01],
            [0.51, 0.51, 0.01]
        ],
        [
            [0.01, 0.01, 0.01],
            [1.01, 0.01, 0.01],
            [0.01, 1.01, 0.01],
            [1.01, 1.01, 0.01]
        ],
        [
            [0.01, 0.01, 0.01],
            [1.01, 0.01, 0.01],
            [0.01, 1.01, 0.01],
            [1.01, 1.01, 0.01]
        ]
    ]

    # Create correspondence tensors for each batch
    y_pred_batch = []
    y_true_batch = []

    for i in range(batch_size):
        y_pred_batch.append(create_correspondence_tensor(pred_source_points_batch[i], pred_target_points_batch[i]))
        y_true_batch.append(create_correspondence_tensor(gt_source_points_batch[i], gt_target_points_batch[i]))

    # Create CorrespondencePrecisionRecall instance
    precision_recall = CorrespondencePrecisionRecall(threshold=0.02)

    # Compute precision, recall, and F1 score for each batch
    batch_results = []
    for i in range(batch_size):
        batch_results.append(precision_recall(y_pred_batch[i], y_true_batch[i]))

    # Expected results for each batch
    expected_results = [
        {"precision": 0.75, "recall": 0.75, "f1_score": 0.75},  # Partial match
        {"precision": 1.0, "recall": 1.0, "f1_score": 1.0},     # Perfect match
        {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}      # No match
    ]

    # Check that the results match expected values
    for i in range(batch_size):
        assert abs(batch_results[i]["precision"].item() - expected_results[i]["precision"]) < 1e-5, \
            f"Batch {i}: Expected precision {expected_results[i]['precision']}, got {batch_results[i]['precision'].item()}"
        assert abs(batch_results[i]["recall"].item() - expected_results[i]["recall"]) < 1e-5, \
            f"Batch {i}: Expected recall {expected_results[i]['recall']}, got {batch_results[i]['recall'].item()}"
        assert abs(batch_results[i]["f1_score"].item() - expected_results[i]["f1_score"]) < 1e-5, \
            f"Batch {i}: Expected F1 score {expected_results[i]['f1_score']}, got {batch_results[i]['f1_score'].item()}"
