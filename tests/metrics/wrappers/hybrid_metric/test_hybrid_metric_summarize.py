import pytest
import torch
from metrics.wrappers.hybrid_metric import HybridMetric


def test_summarize_basic_functionality(metrics_cfg, sample_tensor, sample_target):
    """Test basic summarize functionality."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Generate some scores
    for _ in range(3):
        hybrid_metric(y_pred=sample_tensor, y_true=sample_target)

    # Wait for buffer processing
    hybrid_metric._buffer_queue.join()

    # Test summarize
    summary = hybrid_metric.summarize()

    # Verify summary structure
    assert isinstance(summary, dict)
    assert 'aggregated' in summary
    assert 'per_datapoint' in summary

    # Verify aggregated results
    aggregated = summary['aggregated']
    assert isinstance(aggregated, dict)
    assert 'metric1' in aggregated
    assert 'metric2' in aggregated

    # Verify per-datapoint results
    per_datapoint = summary['per_datapoint']
    assert isinstance(per_datapoint, dict)
    assert 'metric1' in per_datapoint
    assert 'metric2' in per_datapoint

    # Check that per-datapoint has correct number of entries
    assert len(per_datapoint['metric1']) == 3
    assert len(per_datapoint['metric2']) == 3


def test_summarize_with_different_score_values(metrics_cfg):
    """Test summarize with varying score values."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Generate scores with different input values to get varying results
    test_inputs = [
        (torch.zeros(2, 3, 4, 4, dtype=torch.float32), torch.zeros(2, 3, 4, 4, dtype=torch.float32)),
        (torch.ones(2, 3, 4, 4, dtype=torch.float32), torch.zeros(2, 3, 4, 4, dtype=torch.float32)),
        (torch.randn(2, 3, 4, 4, dtype=torch.float32), torch.randn(2, 3, 4, 4, dtype=torch.float32))
    ]

    expected_scores = []
    for input_pair in test_inputs:
        scores = hybrid_metric(y_pred=input_pair[0], y_true=input_pair[1])
        expected_scores.append(scores)

    # Wait for buffer processing
    hybrid_metric._buffer_queue.join()

    # Test summarize
    summary = hybrid_metric.summarize()

    # Verify that per-datapoint matches our expected scores
    per_datapoint = summary['per_datapoint']
    for i, expected in enumerate(expected_scores):
        for key in expected.keys():
            assert torch.allclose(per_datapoint[key][i], expected[key])

    # Verify aggregated is mean of per-datapoint
    aggregated = summary['aggregated']
    for key in per_datapoint.keys():
        expected_mean = per_datapoint[key].mean()
        assert torch.allclose(aggregated[key], expected_mean)


def test_summarize_empty_buffer(metrics_cfg):
    """Test that summarize fails with empty buffer."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Try to summarize without any scores in buffer
    with pytest.raises(AssertionError):
        hybrid_metric.summarize()


def test_summarize_after_reset(metrics_cfg, sample_tensor, sample_target):
    """Test summarize behavior after buffer reset."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Add some scores
    hybrid_metric(y_pred=sample_tensor, y_true=sample_target)
    hybrid_metric._buffer_queue.join()

    # Reset buffer
    hybrid_metric.reset_buffer()

    # Should not be able to summarize empty buffer
    with pytest.raises(AssertionError):
        hybrid_metric.summarize()

    # Add new scores after reset
    hybrid_metric(y_pred=sample_tensor, y_true=sample_target)
    hybrid_metric._buffer_queue.join()

    # Should now be able to summarize
    summary = hybrid_metric.summarize()
    assert isinstance(summary, dict)
    assert len(summary['per_datapoint']['metric1']) == 1


def test_summarize_thread_safety(metrics_cfg, sample_tensor, sample_target):
    """Test that summarize is thread-safe."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Add multiple scores rapidly
    num_scores = 10
    for i in range(num_scores):
        hybrid_metric(y_pred=sample_tensor, y_true=sample_target)

    # Wait for all buffer operations to complete
    hybrid_metric._buffer_queue.join()

    # Summarize should work correctly
    summary = hybrid_metric.summarize()

    # Verify all scores were captured
    assert len(summary['per_datapoint']['metric1']) == num_scores
    assert len(summary['per_datapoint']['metric2']) == num_scores


def test_summarize_output_file_writing(metrics_cfg, sample_tensor, sample_target, tmp_path):
    """Test that summarize can write to output file."""
    hybrid_metric = HybridMetric(metrics_cfg=metrics_cfg)

    # Generate some scores
    hybrid_metric(y_pred=sample_tensor, y_true=sample_target)
    hybrid_metric._buffer_queue.join()

    # Test writing to file
    output_file = tmp_path / "test_summary.json"
    summary = hybrid_metric.summarize(output_path=str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Verify return value is still correct
    assert isinstance(summary, dict)
    assert 'aggregated' in summary
    assert 'per_datapoint' in summary
