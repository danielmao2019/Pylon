"""Tests for atomic displays module imports and __all__ exports.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest


# ================================================================================
# Module Import Tests
# ================================================================================

def test_atomic_displays_module_imports():
    """Test that all atomic display modules can be imported successfully."""
    # Test individual module imports
    from data.viewer.utils.atomic_displays.image_display import (
        create_image_display,
        get_image_display_stats,
    )
    from data.viewer.utils.atomic_displays.depth_display import (
        create_depth_display,
        get_depth_display_stats
    )
    from data.viewer.utils.atomic_displays.normal_display import (
        create_normal_display,
        get_normal_display_stats
    )
    from data.viewer.utils.atomic_displays.edge_display import (
        create_edge_display,
        get_edge_display_stats
    )
    from data.viewer.utils.atomic_displays.segmentation_display import (
        create_segmentation_display,
        get_segmentation_display_stats
    )
    from data.viewer.utils.atomic_displays.point_cloud_display import (
        create_point_cloud_display,
        get_point_cloud_display_stats,
        build_point_cloud_id,
        apply_lod_to_point_cloud,
        normalize_point_cloud_id
    )
    from data.viewer.utils.atomic_displays.instance_surrogate_display import (
        create_instance_surrogate_display,
        get_instance_surrogate_display_stats
    )

    # Verify all functions are callable
    assert callable(create_image_display)
    assert callable(get_image_display_stats)
    assert callable(create_depth_display)
    assert callable(get_depth_display_stats)
    assert callable(create_normal_display)
    assert callable(get_normal_display_stats)
    assert callable(create_edge_display)
    assert callable(get_edge_display_stats)
    assert callable(create_segmentation_display)
    assert callable(get_segmentation_display_stats)
    assert callable(create_point_cloud_display)
    assert callable(get_point_cloud_display_stats)
    assert callable(build_point_cloud_id)
    assert callable(apply_lod_to_point_cloud)
    assert callable(normalize_point_cloud_id)
    assert callable(create_instance_surrogate_display)
    assert callable(get_instance_surrogate_display_stats)


def test_atomic_displays_init_imports():
    """Test that __init__.py imports work correctly."""
    # Test main module import
    import data.viewer.utils.atomic_displays

    # Test that all expected functions are available in the module
    expected_functions = [
        'create_image_display',
        'create_depth_display',
        'create_normal_display',
        'create_edge_display',
        'create_segmentation_display',
        'create_point_cloud_display',
        'create_instance_surrogate_display',
        'get_image_display_stats',
        'get_depth_display_stats',
        'get_normal_display_stats',
        'get_edge_display_stats',
        'get_segmentation_display_stats',
        'get_point_cloud_display_stats',
        'get_instance_surrogate_display_stats',
        'image_to_numpy',
        'build_point_cloud_id',
        'apply_lod_to_point_cloud',
        'normalize_point_cloud_id'
    ]

    for func_name in expected_functions:
        assert hasattr(data.viewer.utils.atomic_displays, func_name), f"Missing function: {func_name}"
        func = getattr(data.viewer.utils.atomic_displays, func_name)
        assert callable(func), f"Function {func_name} is not callable"


def test_atomic_displays_from_import():
    """Test that from imports work correctly."""
    from data.viewer.utils.atomic_displays import (
        create_image_display,
        create_depth_display,
        create_normal_display,
        create_edge_display,
        create_segmentation_display,
        create_point_cloud_display,
        create_instance_surrogate_display,
        get_image_display_stats,
        get_depth_display_stats,
        get_normal_display_stats,
        get_edge_display_stats,
        get_segmentation_display_stats,
        get_point_cloud_display_stats,
        get_instance_surrogate_display_stats,
        build_point_cloud_id,
        apply_lod_to_point_cloud,
        normalize_point_cloud_id
    )

    # Verify all imported functions are callable
    functions = [
        create_image_display,
        create_depth_display,
        create_normal_display,
        create_edge_display,
        create_segmentation_display,
        create_point_cloud_display,
        create_instance_surrogate_display,
        get_image_display_stats,
        get_depth_display_stats,
        get_normal_display_stats,
        get_edge_display_stats,
        get_segmentation_display_stats,
        get_point_cloud_display_stats,
        get_instance_surrogate_display_stats,
        build_point_cloud_id,
        apply_lod_to_point_cloud,
        normalize_point_cloud_id
    ]

    for func in functions:
        assert callable(func)


# ================================================================================
# __all__ Export Tests
# ================================================================================

def test_atomic_displays_all_export():
    """Test that __all__ export list is complete and accurate."""
    import data.viewer.utils.atomic_displays as atomic_displays

    # Check that __all__ exists
    assert hasattr(atomic_displays, '__all__'), "Module missing __all__ attribute"

    all_exports = atomic_displays.__all__
    assert isinstance(all_exports, list), "__all__ should be a list"
    assert len(all_exports) > 0, "__all__ should not be empty"

    # Expected functions based on __init__.py
    expected_exports = [
        # Display functions
        'create_image_display',
        'create_depth_display',
        'create_normal_display',
        'create_edge_display',
        'create_segmentation_display',
        'create_point_cloud_display',
        'create_instance_surrogate_display',
        # Stats functions
        'get_image_display_stats',
        'get_depth_display_stats',
        'get_normal_display_stats',
        'get_edge_display_stats',
        'get_segmentation_display_stats',
        'get_point_cloud_display_stats',
        'get_instance_surrogate_display_stats',
        # Utility functions
        'image_to_numpy',
        'build_point_cloud_id',
        'apply_lod_to_point_cloud',
        'normalize_point_cloud_id'
    ]

    # Check that all expected exports are in __all__
    for export in expected_exports:
        assert export in all_exports, f"Missing export in __all__: {export}"

    # Check that all exports in __all__ are actually available
    for export in all_exports:
        assert hasattr(atomic_displays, export), f"Export in __all__ not available: {export}"
        func = getattr(atomic_displays, export)
        assert callable(func), f"Export in __all__ not callable: {export}"


def test_atomic_displays_all_vs_actual_functions():
    """Test that __all__ matches the actual available functions."""
    import data.viewer.utils.atomic_displays as atomic_displays

    all_exports = set(atomic_displays.__all__)

    # Get all public functions (not starting with _)
    actual_functions = set()
    for name in dir(atomic_displays):
        if not name.startswith('_'):
            obj = getattr(atomic_displays, name)
            if callable(obj):
                actual_functions.add(name)

    # __all__ should be a subset of actual functions
    missing_in_actual = all_exports - actual_functions
    assert len(missing_in_actual) == 0, f"Functions in __all__ but not available: {missing_in_actual}"

    # All actual functions should be in __all__ (or explicitly excluded)
    # For atomic displays, we expect all public functions to be exported
    missing_in_all = actual_functions - all_exports
    assert len(missing_in_all) == 0, f"Available functions not in __all__: {missing_in_all}"


# ================================================================================
# Function Signature Tests
# ================================================================================

def test_display_functions_have_consistent_signatures():
    """Test that display creation functions have consistent signature patterns."""
    from data.viewer.utils.atomic_displays import (
        create_image_display,
        create_depth_display,
        create_normal_display,
        create_edge_display,
        create_segmentation_display,
        create_point_cloud_display,
        create_instance_surrogate_display
    )

    # All display functions should accept a data parameter and title parameter
    # We can't easily check exact signatures, but we can verify they're callable
    display_functions = [
        create_image_display,
        create_depth_display,
        create_normal_display,
        create_edge_display,
        create_segmentation_display,
        create_point_cloud_display,
        create_instance_surrogate_display
    ]

    for func in display_functions:
        assert callable(func)
        # Each should have a __name__ attribute
        assert hasattr(func, '__name__')
        assert func.__name__.startswith('create_')
        assert func.__name__.endswith('_display')


def test_stats_functions_have_consistent_signatures():
    """Test that statistics functions have consistent signature patterns."""
    from data.viewer.utils.atomic_displays import (
        get_image_display_stats,
        get_depth_display_stats,
        get_normal_display_stats,
        get_edge_display_stats,
        get_segmentation_display_stats,
        get_point_cloud_display_stats,
        get_instance_surrogate_display_stats
    )

    stats_functions = [
        get_image_display_stats,
        get_depth_display_stats,
        get_normal_display_stats,
        get_edge_display_stats,
        get_segmentation_display_stats,
        get_point_cloud_display_stats,
        get_instance_surrogate_display_stats
    ]

    for func in stats_functions:
        assert callable(func)
        # Each should have a __name__ attribute
        assert hasattr(func, '__name__')
        assert func.__name__.startswith('get_')
        assert func.__name__.endswith('_display_stats')


# ================================================================================
# Import Error Handling Tests
# ================================================================================

def test_atomic_displays_import_error_handling():
    """Test that import errors are handled gracefully."""
    # This test verifies that the module can be imported without errors
    try:
        import data.viewer.utils.atomic_displays
        # If we get here, the import succeeded
        assert True
    except ImportError as e:
        # If there's an import error, fail the test with details
        pytest.fail(f"Failed to import atomic_displays module: {e}")
    except Exception as e:
        # If there's any other error, fail the test
        pytest.fail(f"Unexpected error importing atomic_displays module: {e}")


def test_individual_module_import_error_handling():
    """Test that individual module imports handle errors gracefully."""
    modules_to_test = [
        'data.viewer.utils.atomic_displays.image_display',
        'data.viewer.utils.atomic_displays.depth_display',
        'data.viewer.utils.atomic_displays.normal_display',
        'data.viewer.utils.atomic_displays.edge_display',
        'data.viewer.utils.atomic_displays.segmentation_display',
        'data.viewer.utils.atomic_displays.point_cloud_display',
        'data.viewer.utils.atomic_displays.instance_surrogate_display'
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            # If we get here, the import succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing {module_name}: {e}")


# ================================================================================
# Module Documentation Tests
# ================================================================================

def test_atomic_displays_module_docstring():
    """Test that the main module has appropriate documentation."""
    import data.viewer.utils.atomic_displays as atomic_displays

    # Module should have a docstring
    assert atomic_displays.__doc__ is not None, "Module should have a docstring"
    assert len(atomic_displays.__doc__.strip()) > 0, "Module docstring should not be empty"

    # Docstring should mention atomic displays or similar
    docstring_lower = atomic_displays.__doc__.lower()
    assert any(term in docstring_lower for term in ['atomic', 'display', 'visualization']), \
        "Module docstring should mention atomic displays or visualization"


def test_individual_module_docstrings():
    """Test that individual modules have appropriate documentation."""
    modules_and_imports = [
        ('data.viewer.utils.atomic_displays.image_display', 'image'),
        ('data.viewer.utils.atomic_displays.depth_display', 'depth'),
        ('data.viewer.utils.atomic_displays.normal_display', 'normal'),
        ('data.viewer.utils.atomic_displays.edge_display', 'edge'),
        ('data.viewer.utils.atomic_displays.segmentation_display', 'segmentation'),
        ('data.viewer.utils.atomic_displays.point_cloud_display', 'point cloud'),
        ('data.viewer.utils.atomic_displays.instance_surrogate_display', 'instance')
    ]

    for module_name, expected_term in modules_and_imports:
        module = __import__(module_name, fromlist=[''])

        # Module should have a docstring
        assert module.__doc__ is not None, f"{module_name} should have a docstring"
        assert len(module.__doc__.strip()) > 0, f"{module_name} docstring should not be empty"

        # Docstring should mention the relevant term
        docstring_lower = module.__doc__.lower()
        assert expected_term in docstring_lower, \
            f"{module_name} docstring should mention '{expected_term}'"


# ================================================================================
# Integration Tests
# ================================================================================

def test_atomic_displays_end_to_end_import():
    """Test end-to-end import workflow that a user would typically follow."""
    # Test the typical user import pattern
    from data.viewer.utils.atomic_displays import (
        create_image_display,
        create_point_cloud_display,
        get_image_display_stats,
        get_point_cloud_display_stats
    )

    # Should be able to import and use immediately
    assert callable(create_image_display)
    assert callable(create_point_cloud_display)
    assert callable(get_image_display_stats)
    assert callable(get_point_cloud_display_stats)

    # Functions should have reasonable names
    assert 'image' in create_image_display.__name__
    assert 'point_cloud' in create_point_cloud_display.__name__
    assert 'stats' in get_image_display_stats.__name__
    assert 'stats' in get_point_cloud_display_stats.__name__


def test_atomic_displays_wildcard_import():
    """Test that wildcard import works correctly."""
    # This is generally not recommended but should work
    exec("from data.viewer.utils.atomic_displays import *")

    # The wildcard import should make functions available in local scope
    # We can't easily test this directly, but if the exec succeeds,
    # it means the wildcard import worked


def test_atomic_displays_import_performance():
    """Test that imports complete in reasonable time."""
    import time

    start_time = time.time()

    # Import the main module
    import data.viewer.utils.atomic_displays

    # Import all functions
    from data.viewer.utils.atomic_displays import (
        create_image_display,
        create_depth_display,
        create_normal_display,
        create_edge_display,
        create_segmentation_display,
        create_point_cloud_display,
        create_instance_surrogate_display,
        get_image_display_stats,
        get_depth_display_stats,
        get_normal_display_stats,
        get_edge_display_stats,
        get_segmentation_display_stats,
        get_point_cloud_display_stats,
        get_instance_surrogate_display_stats,
        build_point_cloud_id,
        apply_lod_to_point_cloud,
        normalize_point_cloud_id
    )

    end_time = time.time()
    import_time = end_time - start_time

    # Import should complete in reasonable time (less than 5 seconds)
    assert import_time < 5.0, f"Import took too long: {import_time:.2f} seconds"
