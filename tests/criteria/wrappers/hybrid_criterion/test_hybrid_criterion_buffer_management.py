import pytest
import torch
from criteria.wrappers.hybrid_criterion import HybridCriterion


def test_buffer_behavior(criteria_cfg, sample_tensor):
    """Test the buffer behavior of HybridCriterion."""
    # Create a criterion
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)

    # Test initialize
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')

    # Test update
    y_true = torch.randn_like(sample_tensor)
    loss = criterion(y_pred=sample_tensor, y_true=y_true)
    criterion._buffer_queue.join()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')

    # Test reset
    criterion.reset_buffer()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')


def test_disabled_buffer_initialization(criteria_cfg, sample_tensor):
    """Test HybridCriterion with disabled buffer."""
    # Create a criterion with disabled buffer
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg, use_buffer=False)

    # Test initialization
    assert criterion.use_buffer is False
    assert not hasattr(criterion, 'buffer')
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')

    # Test that losses are still computed correctly
    y_true = torch.randn_like(sample_tensor)
    loss = criterion(y_pred=sample_tensor, y_true=y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Test that summarize raises error when buffer is disabled
    with pytest.raises(AssertionError):
        criterion.summarize()


def test_summarize_functionality(criteria_cfg, sample_tensor):
    """Test summarize functionality."""
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    y_true = torch.randn_like(sample_tensor)

    # Generate some losses
    for _ in range(3):
        criterion(y_pred=sample_tensor, y_true=y_true)

    # Wait for buffer processing
    criterion._buffer_queue.join()

    # Test summarize
    summary = criterion.summarize()

    # Verify summary is a tensor of losses
    assert isinstance(summary, torch.Tensor)
    assert summary.ndim == 1
    assert len(summary) == 3
    assert torch.all(torch.isfinite(summary))


def test_buffer_operations_thread_safety(criteria_cfg, sample_tensor):
    """Test that buffer operations are thread-safe."""
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    y_true = torch.randn_like(sample_tensor)

    # Perform multiple operations to test thread safety
    for i in range(5):
        loss = criterion(y_pred=sample_tensor, y_true=y_true)
        assert isinstance(loss, torch.Tensor)

    # Wait for all operations to complete
    criterion._buffer_queue.join()

    # Verify buffer contains all losses
    assert len(criterion.buffer) == 5

    # Test that buffer can be safely accessed
    buffer_copy = criterion.get_buffer()
    assert len(buffer_copy) == 5
    assert all(isinstance(item, torch.Tensor) for item in buffer_copy)


def test_summarize_after_reset(criteria_cfg, sample_tensor):
    """Test summarize behavior after buffer reset."""
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    y_true = torch.randn_like(sample_tensor)

    # Add some losses
    criterion(y_pred=sample_tensor, y_true=y_true)
    criterion._buffer_queue.join()

    # Reset buffer
    criterion.reset_buffer()

    # Should not be able to summarize empty buffer
    with pytest.raises(AssertionError):
        criterion.summarize()

    # Add new losses after reset
    criterion(y_pred=sample_tensor, y_true=y_true)
    criterion._buffer_queue.join()

    # Should now be able to summarize
    summary = criterion.summarize()
    assert isinstance(summary, torch.Tensor)
    assert len(summary) == 1


def test_summarize_output_file_writing(criteria_cfg, sample_tensor, tmp_path):
    """Test that summarize can write to output file."""
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    y_true = torch.randn_like(sample_tensor)

    # Generate some losses
    criterion(y_pred=sample_tensor, y_true=y_true)
    criterion._buffer_queue.join()

    # Test writing to file
    output_file = tmp_path / "test_summary.pt"
    summary = criterion.summarize(output_path=str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Verify return value is still correct
    assert isinstance(summary, torch.Tensor)
    assert summary.ndim == 1
