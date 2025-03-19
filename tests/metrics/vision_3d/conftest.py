import sys
import os
from unittest.mock import MagicMock, patch

# Create more detailed mock for PIL
PIL = type('PIL', (), {})
PIL.Image = type('Image', (), {})
PIL.Image.open = MagicMock()
PIL.Image.new = MagicMock()
PIL.Image.NEAREST = 0
PIL.Image.BILINEAR = 1
PIL.Image.BICUBIC = 2
PIL.Image.LANCZOS = 3
sys.modules['PIL'] = PIL
sys.modules['PIL.Image'] = PIL.Image

# Create mock modules for other dependencies we don't need for our tests
mock_modules = [
    'rasterio',
    'plyfile',
    'cv2',
    'open3d',
    'matplotlib',
    'matplotlib.pyplot',
    'imageio',
    'torchvision',
    'torchvision.transforms',
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

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
