import numpy
import torch
import cv2


class NormalizeImage:
    """Normalize the RGB and depth image.
    """

    def __init__(self, mean, std, to_rgb=True, depth_norm_type=None):
        """
        Args:
            mean (sequence): Mean values of 3 channels.
            std (sequence): Std values of 3 channels.
            to_rgb (bool): Whether to convert the image from BGR to RGB,
                default is true.
            depth_norm_type (str): `minmaxflip`, `minmax` or `uoais`,
                default is 'minmaxflip'.
        """
        self.mean = numpy.array(mean, dtype=numpy.float32)
        self.std = numpy.array(std, dtype=numpy.float32)
        self.to_rgb = to_rgb
        self.depth_norm_type = depth_norm_type

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        assert image.dtype != torch.uint8
        stdinv = 1 / (self.std.reshape(1, -1)).type(torch.float64)
        if self.to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # inplace
        cv2.subtract(image, self.mean, image)  # inplace
        cv2.multiply(image, stdinv, image)  # inplace
        return image
