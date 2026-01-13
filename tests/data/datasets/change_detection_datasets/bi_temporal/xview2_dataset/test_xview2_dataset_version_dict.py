"""Test version dict implementation for xView2Dataset."""

import inspect
from typing import Any, Dict

import pytest

from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import (
    xView2Dataset,
)
from utils.builders.builder import build_from_config


def test_xview2_dataset_has_version_dict_method():
    """Test that xView2Dataset has _get_cache_version_dict method."""
    assert hasattr(xView2Dataset, '_get_cache_version_dict')

    # Check method signature
    method = getattr(xView2Dataset, '_get_cache_version_dict')
    signature = inspect.signature(method)

    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']

    # Should return Dict[str, Any]
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_xview2_dataset_version_dict_with_train_config(xview2_dataset_train_config):
    """Test that xView2Dataset version dict method works correctly."""
    xview2_dataset_train = build_from_config(xview2_dataset_train_config)

    version_dict = xview2_dataset_train._get_cache_version_dict()

    # Should return a dictionary
    assert isinstance(version_dict, dict)

    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'xView2Dataset'

    # Should contain base parameters
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
