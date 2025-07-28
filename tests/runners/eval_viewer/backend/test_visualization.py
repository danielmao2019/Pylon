import pytest
import numpy as np
from runners.eval_viewer.backend.visualization import (
    create_score_map_grid,
    create_overlaid_score_map,
    get_color_for_score
)


# ==========================================
# Tests for create_score_map_grid function
# ==========================================

def test_create_score_map_grid_with_list_input():
    """Test create_score_map_grid with list input."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = create_score_map_grid(scores)
    
    # Should create a 3x3 grid (since sqrt(5) ~ 2.24, ceil = 3)
    assert result.shape == (3, 3)
    assert result.flat[0] == 0.1
    assert result.flat[1] == 0.2
    assert result.flat[2] == 0.3
    assert result.flat[3] == 0.4
    assert result.flat[4] == 0.5
    # Remaining positions should be NaN
    assert np.all(np.isnan(result.flat[5:]))


def test_create_score_map_grid_with_numpy_array():
    """Test create_score_map_grid with numpy array input."""
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    result = create_score_map_grid(scores)
    
    # Should create a 2x2 grid (since sqrt(4) = 2)
    assert result.shape == (2, 2)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 2.0
    assert result[1, 0] == 3.0
    assert result[1, 1] == 4.0


def test_create_score_map_grid_perfect_square():
    """Test create_score_map_grid with perfect square number of elements."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = create_score_map_grid(scores)
    
    # Should create a 3x3 grid (since sqrt(9) = 3)
    assert result.shape == (3, 3)
    # No NaN values for perfect square
    assert not np.any(np.isnan(result))
    
    # Check all values are preserved
    flattened = result.flatten()
    for i, score in enumerate(scores):
        assert flattened[i] == score


def test_create_score_map_grid_single_element():
    """Test create_score_map_grid with single element."""
    scores = [0.5]
    result = create_score_map_grid(scores)
    
    # Should create a 1x1 grid
    assert result.shape == (1, 1)
    assert result[0, 0] == 0.5


def test_create_score_map_grid_large_array():
    """Test create_score_map_grid with larger array."""
    scores = list(range(50))  # 50 elements
    result = create_score_map_grid(scores)
    
    # Should create an 8x8 grid (since sqrt(50) ~ 7.07, ceil = 8)
    assert result.shape == (8, 8)
    
    # First 50 elements should match input
    flattened = result.flatten()
    for i in range(50):
        assert flattened[i] == i
    
    # Remaining 14 elements should be NaN
    assert np.sum(np.isnan(flattened)) == 64 - 50


@pytest.mark.parametrize("scores,expected_shape", [
    ([1], (1, 1)),
    ([1, 2], (2, 2)), 
    ([1, 2, 3], (2, 2)),
    ([1, 2, 3, 4], (2, 2)),
    ([1, 2, 3, 4, 5], (3, 3)),
    (list(range(16)), (4, 4)),
    (list(range(17)), (5, 5)),
])
def test_create_score_map_grid_shapes(scores, expected_shape):
    """Test create_score_map_grid produces correct shapes for various inputs."""
    result = create_score_map_grid(scores)
    assert result.shape == expected_shape


# Error case tests
def test_create_score_map_grid_none_input():
    """Test create_score_map_grid fails with None input."""
    with pytest.raises(AssertionError, match="scores must not be None"):
        create_score_map_grid(None)


def test_create_score_map_grid_empty_list():
    """Test create_score_map_grid fails with empty list."""
    with pytest.raises(AssertionError, match="scores must not be empty"):
        create_score_map_grid([])


def test_create_score_map_grid_empty_array():
    """Test create_score_map_grid fails with empty numpy array."""
    with pytest.raises(AssertionError, match="scores must not be empty"):
        create_score_map_grid(np.array([]))


def test_create_score_map_grid_invalid_type():
    """Test create_score_map_grid fails with invalid input type."""
    with pytest.raises(AssertionError, match="scores must be list or numpy array"):
        create_score_map_grid("invalid")


def test_create_score_map_grid_invalid_list_elements():
    """Test create_score_map_grid fails with non-numeric list elements."""
    with pytest.raises(AssertionError, match="All scores must be numeric"):
        create_score_map_grid([1, 2, "invalid", 4])


def test_create_score_map_grid_multidimensional_array():
    """Test create_score_map_grid fails with multidimensional array."""
    scores_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(AssertionError, match="scores must be 1D array"):
        create_score_map_grid(scores_2d)


# ==========================================
# Tests for create_overlaid_score_map function  
# ==========================================

def test_create_overlaid_score_map_basic(sample_score_maps):
    """Test create_overlaid_score_map with basic functionality."""
    result = create_overlaid_score_map(sample_score_maps, percentile=25)
    
    # Should have same shape as input maps
    assert result.shape == sample_score_maps[0].shape
    
    # Should be success rates (between 0 and 1)
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values >= 0.0)
    assert np.all(valid_values <= 1.0)
    
    # NaN positions in overlaid map: NaN < threshold is False, so treated as success (1.0)
    # This is the actual behavior of the function
    assert result[2, 2] == 1.0  # NaN positions become success (1.0)


def test_create_overlaid_score_map_different_percentiles(sample_score_maps):
    """Test create_overlaid_score_map with different percentile values."""
    result_10 = create_overlaid_score_map(sample_score_maps, percentile=10)
    result_50 = create_overlaid_score_map(sample_score_maps, percentile=50)
    result_90 = create_overlaid_score_map(sample_score_maps, percentile=90)
    
    # All should have same shape
    assert result_10.shape == result_50.shape == result_90.shape
    
    # Different percentiles should give different thresholds and results
    # Note: Since this is success rate (1 - failure_rate), we can't make simple comparisons
    # But we can verify they're all valid success rates
    for result in [result_10, result_50, result_90]:
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)


def test_create_overlaid_score_map_single_map():
    """Test create_overlaid_score_map with single score map."""
    single_map = np.array([
        [0.1, 0.5, 0.9],
        [0.3, 0.7, 0.2],
        [0.8, 0.4, np.nan]
    ])
    
    result = create_overlaid_score_map([single_map], percentile=50)
    
    # Should have same shape
    assert result.shape == single_map.shape
    
    # With single map, result should be binary (0 or 1) except for NaN
    valid_values = result[~np.isnan(result)]
    assert np.all(np.isin(valid_values, [0.0, 1.0]))
    
    # NaN positions in overlaid map: NaN < threshold is False, so treated as success (1.0)
    assert result[2, 2] == 1.0  # NaN position becomes success (1.0)


def test_create_overlaid_score_map_all_same_values():
    """Test create_overlaid_score_map when all score maps have same values."""
    # Create 3 identical maps
    same_map = np.array([[0.5, 0.5], [0.5, 0.5]])
    same_maps = [same_map.copy() for _ in range(3)]
    
    result = create_overlaid_score_map(same_maps, percentile=50)
    
    # When all values are the same and equal to the percentile threshold,
    # the result depends on whether they're below threshold
    # Since all values are 0.5 and percentile=50, threshold will be 0.5
    # Values equal to threshold are NOT considered failures (< threshold)
    # So success rate should be 1.0 everywhere
    assert np.allclose(result, 1.0)


def test_create_overlaid_score_map_with_nans():
    """Test create_overlaid_score_map handles NaN values correctly."""
    # Create maps with different NaN patterns
    map1 = np.array([[0.1, np.nan], [0.3, 0.4]])
    map2 = np.array([[np.nan, 0.2], [0.5, np.nan]])
    map3 = np.array([[0.6, 0.7], [np.nan, 0.8]])
    
    result = create_overlaid_score_map([map1, map2, map3], percentile=50)
    
    # Should have same shape
    assert result.shape == (2, 2)
    
    # All positions should have valid values (NaNs become 1.0 due to NaN < threshold = False)
    # Aggregation treats NaN positions as successes since NaN < threshold evaluates to False
    assert not np.any(np.isnan(result))
    
    # All values should be valid success rates
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


# ==========================================
# Tests for get_color_for_score function
# ==========================================

def test_get_color_for_score_min_score():
    """Test get_color_for_score at minimum score (should be red)."""
    color = get_color_for_score(0.0, 0.0, 1.0)
    assert color == 'rgb(255, 0, 0)'  # Pure red


def test_get_color_for_score_max_score():
    """Test get_color_for_score at maximum score (should be green)."""
    color = get_color_for_score(1.0, 0.0, 1.0)
    assert color == 'rgb(0, 255, 0)'  # Pure green


def test_get_color_for_score_middle_score():
    """Test get_color_for_score at middle score (should be yellow)."""
    color = get_color_for_score(0.5, 0.0, 1.0)
    assert color == 'rgb(255, 255, 0)'  # Pure yellow


def test_get_color_for_score_quarter_score():
    """Test get_color_for_score at quarter score (red-yellow transition)."""
    color = get_color_for_score(0.25, 0.0, 1.0)
    assert color == 'rgb(255, 127, 0)'  # Red with some green


def test_get_color_for_score_three_quarter_score():
    """Test get_color_for_score at three-quarter score (yellow-green transition)."""
    color = get_color_for_score(0.75, 0.0, 1.0)
    assert color == 'rgb(127, 255, 0)'  # Green with some red


def test_get_color_for_score_nan_input():
    """Test get_color_for_score with NaN input."""
    color = get_color_for_score(np.nan, 0.0, 1.0)
    assert color == '#808080'  # Gray for NaN


def test_get_color_for_score_custom_range():
    """Test get_color_for_score with custom min/max range."""
    # Test with range [10, 20]
    color_min = get_color_for_score(10.0, 10.0, 20.0)
    color_mid = get_color_for_score(15.0, 10.0, 20.0)
    color_max = get_color_for_score(20.0, 10.0, 20.0)
    
    assert color_min == 'rgb(255, 0, 0)'    # Red
    assert color_mid == 'rgb(255, 255, 0)'  # Yellow
    assert color_max == 'rgb(0, 255, 0)'    # Green


def test_get_color_for_score_negative_range():
    """Test get_color_for_score with negative value range."""
    # Test with range [-1, 1]
    color_min = get_color_for_score(-1.0, -1.0, 1.0)
    color_mid = get_color_for_score(0.0, -1.0, 1.0)
    color_max = get_color_for_score(1.0, -1.0, 1.0)
    
    assert color_min == 'rgb(255, 0, 0)'    # Red
    assert color_mid == 'rgb(255, 255, 0)'  # Yellow
    assert color_max == 'rgb(0, 255, 0)'    # Green


def test_get_color_for_score_same_min_max():
    """Test get_color_for_score when min equals max."""
    # When min == max, this causes division by zero
    # This is an edge case that should be handled by the function
    with pytest.raises(ZeroDivisionError):
        get_color_for_score(5.0, 5.0, 5.0)


@pytest.mark.parametrize("score,min_score,max_score,expected_color", [
    (0.0, 0.0, 1.0, 'rgb(255, 0, 0)'),     # Min -> Red
    (0.5, 0.0, 1.0, 'rgb(255, 255, 0)'),   # Mid -> Yellow  
    (1.0, 0.0, 1.0, 'rgb(0, 255, 0)'),     # Max -> Green
    (np.nan, 0.0, 1.0, '#808080'),         # NaN -> Gray
])
def test_get_color_for_score_parametrized(score, min_score, max_score, expected_color):
    """Parametrized test for get_color_for_score function."""
    result = get_color_for_score(score, min_score, max_score)
    assert result == expected_color


def test_get_color_for_score_color_format():
    """Test that get_color_for_score returns valid RGB color format."""
    color = get_color_for_score(0.3, 0.0, 1.0)
    
    # Should match rgb(r, g, b) format
    assert color.startswith('rgb(')
    assert color.endswith(')')
    
    # Extract RGB values
    rgb_part = color[4:-1]  # Remove 'rgb(' and ')'
    r, g, b = map(int, rgb_part.split(', '))
    
    # RGB values should be in valid range [0, 255]
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
