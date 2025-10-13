"""Test cases for deterministic hash utilities."""

from utils.determinism.hash_utils import deterministic_hash, convert_to_seed


def test_deterministic_hash_consistency():
    """Test that deterministic_hash produces consistent results."""
    test_objects = [
        42,
        3.14159,
        "hello world",
        True,
        False,
        None,
        [1, 2, 3],
        (1, 2, 3),
        {1, 2, 3},
        frozenset([1, 2, 3]),
        {"a": 1, "b": 2},
        {"nested": {"deep": [1, 2]}},
        [],
        {},
        set(),
    ]

    for obj in test_objects:
        # Test that same object produces same hash multiple times
        hashes = [deterministic_hash(obj) for _ in range(3)]
        assert all(h == hashes[0] for h in hashes), f"Inconsistent hash for {obj}"
        assert isinstance(hashes[0], int), f"Hash should be int, got {type(hashes[0])}"
        assert (
            0 <= hashes[0] < 2**32
        ), f"Hash should be 32-bit unsigned int, got {hashes[0]}"


def test_type_distinction():
    """Test that different types produce different hashes."""
    # These pairs should produce different hashes
    test_pairs = [
        (123, "123"),  # int vs string
        ([1, 2, 3], (1, 2, 3)),  # list vs tuple
        ({1, 2, 3}, [1, 2, 3]),  # set vs list
        (True, 1),  # bool vs int
        ({}, []),  # empty dict vs empty list
        (1.0, 1),  # float vs int
        ("", 0),  # empty string vs zero
    ]

    for obj1, obj2 in test_pairs:
        hash1, hash2 = deterministic_hash(obj1), deterministic_hash(obj2)
        assert (
            hash1 != hash2
        ), f"Different types should have different hashes: {obj1} vs {obj2}"


def test_order_independence():
    """Test that order doesn't matter for unordered collections."""
    # Dict order shouldn't matter
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"c": 3, "a": 1, "b": 2}
    assert deterministic_hash(dict1) == deterministic_hash(dict2)

    # Set order shouldn't matter
    set1 = {1, 2, 3}
    set2 = {3, 1, 2}
    assert deterministic_hash(set1) == deterministic_hash(set2)

    # But list order SHOULD matter
    list1 = [1, 2, 3]
    list2 = [3, 2, 1]
    assert deterministic_hash(list1) != deterministic_hash(list2)


def test_convert_to_seed():
    """Test convert_to_seed function."""
    # Test that integers are passed through (with range check)
    assert convert_to_seed(42) == 42
    assert convert_to_seed(2**32 + 5) == 5  # Should wrap around

    # Test that non-integers are converted
    hash_val = convert_to_seed("hello")
    assert isinstance(hash_val, int)
    assert 0 <= hash_val < 2**32

    # Test consistency
    assert convert_to_seed("hello") == convert_to_seed("hello")

    # Test complex objects
    complex_obj = {"config": [1, 2], "meta": (3, 4)}
    seed1 = convert_to_seed(complex_obj)
    seed2 = convert_to_seed(complex_obj)
    assert seed1 == seed2
    assert isinstance(seed1, int)


def test_nested_structures():
    """Test deeply nested data structures."""
    nested = {
        "level1": {"level2": [{"level3": (1, 2, 3)}, {"level3": {4, 5, 6}}]},
        "other": frozenset([7, 8, 9]),
    }

    # Should not crash and should be consistent
    hash1 = deterministic_hash(nested)
    hash2 = deterministic_hash(nested)
    assert hash1 == hash2
    assert isinstance(hash1, int)


def test_edge_cases():
    """Test edge cases and special values."""
    edge_cases = [
        None,
        "",
        0,
        0.0,
        False,
        [],
        {},
        set(),
        (),
        frozenset(),
        b"bytes",
    ]

    # Should handle all edge cases without crashing
    for obj in edge_cases:
        hash_val = deterministic_hash(obj)
        assert isinstance(hash_val, int)
        assert 0 <= hash_val < 2**32

        # Should be consistent
        assert deterministic_hash(obj) == hash_val


def test_non_negative_guarantee():
    """Test that deterministic_hash() always returns non-negative values.

    This guarantee is critical for color generation in segmentation visualization
    (data.viewer.utils.segmentation.get_color), where the hash value is used
    directly in the golden ratio color mapping algorithm without needing abs().

    The previous implementation used abs(hash(obj)) because Python's built-in
    hash() can return negative values. Our deterministic_hash() eliminates
    this need by guaranteeing non-negative output.
    """
    # Test a wide variety of objects that could be used as class identifiers
    test_objects = [
        # Primitive types
        -999,
        -1,
        0,
        1,
        999,
        -3.14159,
        0.0,
        3.14159,
        "",
        "negative_string",
        "class_name",
        True,
        False,
        None,
        # Collections with potentially negative elements
        [-1, -2, -3],
        (-5, -4, -3),
        {-10, -20, -30},
        {"negative": -1, "zero": 0},
        # Complex nested structures
        {"classes": [-1, "background"], "meta": {"version": -2}},
        [("negative_tuple", -1), ["negative_list", -2]],
        # Edge cases that might cause issues
        float('-inf'),
        float('inf'),
        "",  # empty string
        [],  # empty collections
        {},
        set(),
        # Objects with string representations that might be negative
        type(-1),  # <class 'int'>
        slice(-1, None, -1),  # slice(-1, None, -1)
    ]

    for obj in test_objects:
        hash_val = deterministic_hash(obj)

        # The core guarantee: always non-negative
        assert hash_val >= 0, f"deterministic_hash({obj}) = {hash_val} is negative!"

        # Additional constraints for segmentation color mapping
        assert isinstance(hash_val, int), f"Hash should be int, got {type(hash_val)}"
        assert hash_val < 2**32, f"Hash should fit in 32-bit range, got {hash_val}"

        # Verify consistency (same input -> same output)
        assert deterministic_hash(obj) == hash_val, f"Inconsistent hash for {obj}"
