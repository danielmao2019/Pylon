import os
from unittest.mock import MagicMock, patch

# Create a mock for save_json that actually creates a file
def mock_save_json(obj, filepath):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Create an empty file
    with open(filepath, 'w') as f:
        f.write('{}')

# Mock specific functions to prevent file I/O during tests
patch('utils.io.save_json', mock_save_json).start()
patch('utils.io.load_image', MagicMock()).start()
patch('utils.io.load_point_cloud', MagicMock()).start()
patch('utils.input_checks.check_write_file', MagicMock()).start()
