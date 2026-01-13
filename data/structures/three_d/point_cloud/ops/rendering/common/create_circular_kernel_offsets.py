import torch


def create_circular_kernel_offsets(
    point_size: float, device: torch.device
) -> torch.Tensor:
    """Create offset positions for circular kernel.

    Args:
        point_size: Diameter of the circular kernel in pixels
        device: Device for the tensor

    Returns:
        Tensor of shape [num_pixels_in_circle, 2] with (y, x) offsets
    """
    kernel_size = int(torch.ceil(torch.tensor(point_size)))
    kernel_radius = point_size / 2.0

    y_kernel, x_kernel = torch.meshgrid(
        torch.arange(kernel_size, device=device) - kernel_size // 2,
        torch.arange(kernel_size, device=device) - kernel_size // 2,
        indexing='ij',
    )

    kernel_distances = torch.sqrt(x_kernel.float() ** 2 + y_kernel.float() ** 2)
    circular_mask = kernel_distances <= kernel_radius

    kernel_offsets = torch.stack(
        [y_kernel[circular_mask], x_kernel[circular_mask]], dim=1
    )

    return kernel_offsets
