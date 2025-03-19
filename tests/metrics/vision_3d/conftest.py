import sys
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

# Mock specific functions to prevent file I/O during tests
patch('utils.io.save_json', MagicMock()).start()
patch('utils.io.load_image', MagicMock()).start()
patch('utils.io.load_point_cloud', MagicMock()).start()
patch('utils.input_checks.check_write_file', MagicMock()).start() 