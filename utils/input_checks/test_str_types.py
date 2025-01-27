import pytest
from .str_types import check_read_file
import os
from tempfile import NamedTemporaryFile


# Helper function to create a temporary file for testing
def create_temp_file(suffix: str = "") -> str:
    temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name


def test_check_read_file_success():
    # Test with a valid file path and extension
    temp_file = create_temp_file(".txt")
    try:
        assert check_read_file(temp_file, ".txt") == temp_file
        assert check_read_file(temp_file, [".txt", ".log"]) == temp_file
        assert check_read_file(temp_file) == temp_file  # No extension check
    finally:
        os.remove(temp_file)


@pytest.mark.parametrize("invalid_path", [
    None,
    123,
    ["invalid", "path"],
    {},
])
def test_check_read_file_invalid_path_type(invalid_path):
    with pytest.raises(AssertionError, match="type\(path\)="):
        check_read_file(invalid_path)


def test_check_read_file_file_not_exist():
    # Test with a non-existent file path
    with pytest.raises(AssertionError, match="path="):
        check_read_file("/non/existent/file.txt")


@pytest.mark.parametrize("invalid_ext", [
    123,                 # Not a string or list
    {},                  # Invalid type
])
def test_check_read_file_invalid_ext_type_1(invalid_ext):
    temp_file = create_temp_file(".txt")
    try:
        with pytest.raises(AssertionError, match="type\(ext\)="):
            check_read_file(temp_file, invalid_ext)
    finally:
        os.remove(temp_file)


@pytest.mark.parametrize("invalid_ext", [
    [".txt", 123],       # List with a non-string element
])
def test_check_read_file_invalid_ext_type_2(invalid_ext):
    temp_file = create_temp_file(".txt")
    try:
        with pytest.raises(AssertionError, match="ext="):
            check_read_file(temp_file, invalid_ext)
    finally:
        os.remove(temp_file)


def test_check_read_file_invalid_extension():
    temp_file = create_temp_file(".txt")
    try:
        # Test with a valid file but invalid extension
        with pytest.raises(AssertionError, match="path=.*, ext="):
            check_read_file(temp_file, ".log")
    finally:
        os.remove(temp_file)


def test_check_read_file_empty_ext_list():
    temp_file = create_temp_file(".txt")
    try:
        # Test with an empty extension list
        with pytest.raises(AssertionError, match="path=.*, ext="):
            check_read_file(temp_file, [])
    finally:
        os.remove(temp_file)
