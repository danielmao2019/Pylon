import torch
import pytest
from utils.point_cloud_ops.select import Select


def test_select_basic_list():
    """Test basic selection functionality using list indices."""
    pc = {
        'pos': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Point 0
                [2.0, 0.0, 0.0],  # Point 1
                [3.0, 0.0, 0.0],  # Point 2
                [4.0, 0.0, 0.0],  # Point 3
                [5.0, 0.0, 0.0],  # Point 4
            ],
            dtype=torch.float64,
        ),
        'rgb': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
            ],
            dtype=torch.float64,
        ),
        'classification': torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
    }

    select = Select([0, 2, 4])  # Select first, third, and fifth points
    result = select(pc)

    # Check that indices key is created
    assert 'indices' in result
    assert torch.equal(result['indices'], torch.tensor([0, 2, 4], dtype=torch.int64))

    # Check that positions are correctly selected
    expected_pos = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Point 0
            [3.0, 0.0, 0.0],  # Point 2
            [5.0, 0.0, 0.0],  # Point 4
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['pos'], expected_pos)

    # Check that other features are correctly selected
    expected_rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Red (Point 0)
            [0.0, 0.0, 1.0],  # Blue (Point 2)
            [1.0, 0.0, 1.0],  # Magenta (Point 4)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['rgb'], expected_rgb)

    expected_classification = torch.tensor([0, 2, 1], dtype=torch.long)
    assert torch.equal(result['classification'], expected_classification)


def test_select_basic_tensor():
    """Test basic selection functionality using tensor indices."""
    pc = {
        'pos': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Point 0
                [2.0, 0.0, 0.0],  # Point 1
                [3.0, 0.0, 0.0],  # Point 2
            ],
            dtype=torch.float64,
        ),
        'rgb': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
            ],
            dtype=torch.float64,
        ),
    }

    indices_tensor = torch.tensor([1, 2], dtype=torch.int64, device=pc['pos'].device)
    select = Select(indices_tensor)
    result = select(pc)

    # Check that indices key is created
    assert 'indices' in result
    assert torch.equal(result['indices'], indices_tensor)

    # Check that positions are correctly selected
    expected_pos = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # Point 1
            [3.0, 0.0, 0.0],  # Point 2
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['pos'], expected_pos)


def test_select_empty_indices():
    """Test selection with empty indices."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
        'rgb': torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        'classification': torch.tensor([0, 1], dtype=torch.long),
    }

    select = Select([])
    result = select(pc)

    # Check that indices key is created but empty
    assert 'indices' in result
    assert result['indices'].numel() == 0

    # Check that all tensors are empty but have correct shape
    assert result['pos'].shape == (0, 3)
    assert result['rgb'].shape == (0, 3)
    assert result['classification'].shape == (0,)


def test_select_single_point():
    """Test selection of a single point."""
    pc = {
        'pos': torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
        ),
        'rgb': torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        ),
    }

    select = Select([1])  # Select second point only
    result = select(pc)

    # Check that indices key is created
    assert 'indices' in result
    assert torch.equal(result['indices'], torch.tensor([1], dtype=torch.int64))

    # Check that position is correctly selected
    expected_pos = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(result['pos'], expected_pos)

    # Check shape is maintained
    assert result['pos'].shape == (1, 3)
    assert result['rgb'].shape == (1, 3)


def test_select_out_of_order():
    """Test selection with out-of-order indices."""
    pc = {
        'pos': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Point 0
                [2.0, 0.0, 0.0],  # Point 1
                [3.0, 0.0, 0.0],  # Point 2
                [4.0, 0.0, 0.0],  # Point 3
            ],
            dtype=torch.float64,
        ),
    }

    select = Select([3, 0, 2])  # Out of order: 3rd, 0th, 2nd
    result = select(pc)

    # Check that indices preserve the order
    assert torch.equal(result['indices'], torch.tensor([3, 0, 2], dtype=torch.int64))

    # Check that positions follow the specified order
    expected_pos = torch.tensor(
        [
            [4.0, 0.0, 0.0],  # Point 3 (first)
            [1.0, 0.0, 0.0],  # Point 0 (second)
            [3.0, 0.0, 0.0],  # Point 2 (third)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['pos'], expected_pos)


def test_select_duplicate_indices():
    """Test selection with duplicate indices."""
    pc = {
        'pos': torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
        ),
    }

    select = Select([1, 1, 2, 1])  # Point 1 appears three times
    result = select(pc)

    # Check that indices preserve duplicates
    assert torch.equal(result['indices'], torch.tensor([1, 1, 2, 1], dtype=torch.int64))

    # Check that positions are duplicated correctly
    expected_pos = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # Point 1 (1st copy)
            [2.0, 0.0, 0.0],  # Point 1 (2nd copy)
            [3.0, 0.0, 0.0],  # Point 2
            [2.0, 0.0, 0.0],  # Point 1 (3rd copy)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['pos'], expected_pos)


def test_select_index_chaining_basic():
    """Test that index chaining works correctly - this catches the bug we fixed."""
    # Start with a PC that already has indices [0, 2, 4]
    pc_with_indices = {
        'pos': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Original index 0
                [3.0, 0.0, 0.0],  # Original index 2
                [5.0, 0.0, 0.0],  # Original index 4
            ],
            dtype=torch.float64,
        ),
        'rgb': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 0.0, 1.0],  # Magenta
            ],
            dtype=torch.float64,
        ),
        'indices': torch.tensor([0, 2, 4], dtype=torch.int64),  # Maps to original PC
    }

    # Now select indices [0, 2] from this already-selected PC
    # This should result in final indices [0, 4] -> positions at [1,0,0], [5,0,0]
    select = Select([0, 2])  # Select 1st and 3rd of the already-selected points
    result = select(pc_with_indices)

    # Check that final indices are chained correctly
    expected_final_indices = torch.tensor([0, 4], dtype=torch.int64)
    assert torch.equal(result['indices'], expected_final_indices)

    # Check that positions correspond to the original PC at these indices
    expected_pos = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Original point 0
            [5.0, 0.0, 0.0],  # Original point 4
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['pos'], expected_pos)

    # Check that other features are chained correctly
    expected_rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Red (Original point 0)
            [1.0, 0.0, 1.0],  # Magenta (Original point 4)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result['rgb'], expected_rgb)


def test_select_index_chaining_complex():
    """Test multi-level index chaining."""
    pc = {
        'pos': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Point 0
                [2.0, 0.0, 0.0],  # Point 1
                [3.0, 0.0, 0.0],  # Point 2
                [4.0, 0.0, 0.0],  # Point 3
                [5.0, 0.0, 0.0],  # Point 4
            ],
            dtype=torch.float64,
        ),
    }

    # First selection: [1, 3, 4] -> indices [1, 3, 4]
    select1 = Select([1, 3, 4])
    result1 = select1(pc)

    # Check first level
    assert torch.equal(result1['indices'], torch.tensor([1, 3, 4], dtype=torch.int64))
    expected_pos1 = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # Point 1
            [4.0, 0.0, 0.0],  # Point 3
            [5.0, 0.0, 0.0],  # Point 4
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result1['pos'], expected_pos1)

    # Second selection: [0, 2] from result1 -> should give final indices [1, 4]
    select2 = Select([0, 2])  # Select 1st and 3rd from [1, 3, 4]
    result2 = select2(result1)

    # Check final chained indices
    expected_final_indices = torch.tensor([1, 4], dtype=torch.int64)
    assert torch.equal(result2['indices'], expected_final_indices)

    # Check final positions map back to original
    expected_final_pos = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # Original point 1
            [5.0, 0.0, 0.0],  # Original point 4
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result2['pos'], expected_final_pos)


def test_select_device_consistency_cpu():
    """Test that device consistency is enforced on CPU."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device='cpu'),
        'rgb': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device='cpu'),
    }

    # This should work fine
    indices_cpu = torch.tensor([0], dtype=torch.int64, device='cpu')
    select = Select(indices_cpu)
    result = select(pc)
    assert result['indices'].device == torch.device('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_select_device_consistency_cuda():
    """Test that device consistency is enforced on CUDA."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device='cuda'),
        'rgb': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device='cuda'),
    }

    # This should work fine - indices on same device
    indices_cuda = torch.tensor([0], dtype=torch.int64, device='cuda')
    select = Select(indices_cuda)
    result = select(pc)
    assert result['indices'].device.type == 'cuda'

    # This should fail - indices on different device
    indices_cpu = torch.tensor([0], dtype=torch.int64, device='cpu')
    select_wrong_device = Select(indices_cpu)
    with pytest.raises(AssertionError, match="device"):
        select_wrong_device(pc)


def test_select_dtype_validation():
    """Test that indices must be int64."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
    }

    # Wrong dtype should fail
    indices_float = torch.tensor([0, 1], dtype=torch.float32)
    select = Select(indices_float)
    with pytest.raises(AssertionError):
        select(pc)

    # int32 should also fail
    indices_int32 = torch.tensor([0, 1], dtype=torch.int32)
    select_int32 = Select(indices_int32)
    with pytest.raises(AssertionError):
        select_int32(pc)


def test_select_invalid_indices():
    """Test behavior with invalid indices."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
    }

    # Negative indices should fail (now explicitly checked)
    select_negative = Select([-1])
    with pytest.raises(AssertionError, match="Negative indices not allowed"):
        select_negative(pc)

    # Out of bounds indices should fail
    select_oob = Select([5])  # PC only has 2 points (0-1)
    with pytest.raises(IndexError):
        select_oob(pc)


def test_select_string_representation():
    """Test string representation."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    }

    # Short list
    select_short = Select([0, 1, 2])
    expected = "Select(indices=[0, 1, 2])"
    assert str(select_short) == expected

    # Long list
    long_list = list(range(10))
    select_long = Select(long_list)
    expected = "Select(indices=[...10 indices])"
    assert str(select_long) == expected

    # Short tensor (after calling)
    indices = torch.tensor([0], dtype=torch.int64)
    select_tensor = Select(indices)
    select_tensor(pc)  # Convert to tensor
    expected = "Select(indices=[0])"
    assert str(select_tensor) == expected


def test_select_list_to_tensor_conversion():
    """Test that list indices are properly converted to tensors."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
    }

    select = Select([0, 1])

    # Initially, indices should be a list
    assert isinstance(select.indices, list)

    # After calling, indices should be converted to tensor
    result = select(pc)
    assert isinstance(select.indices, torch.Tensor)
    assert select.indices.dtype == torch.int64
    assert torch.equal(select.indices, torch.tensor([0, 1], dtype=torch.int64))


def test_select_with_additional_tensor_keys():
    """Test that additional tensor keys are handled correctly."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
        'rgb': torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        'classification': torch.tensor([0, 1], dtype=torch.long),
        'intensity': torch.tensor([0.1, 0.2], dtype=torch.float32),
    }

    select = Select([0])
    result = select(pc)

    # All tensor values should be indexed
    assert torch.allclose(
        result['pos'], torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    )
    assert torch.allclose(
        result['rgb'], torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    )
    assert torch.equal(result['classification'], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(result['intensity'], torch.tensor([0.1], dtype=torch.float32))
    assert torch.equal(result['indices'], torch.tensor([0], dtype=torch.int64))
