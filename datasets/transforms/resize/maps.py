import torch
import torchvision


class ResizeMaps:

    def __init__(self, **kwargs) -> None:
        self.resize = torchvision.transforms.Resize(**kwargs)
        target_size = (self.resize.size,) * 2 if type(self.resize.size) == int else tuple(self.resize.size)
        assert type(target_size) == tuple, f"{type(target_size)=}"
        assert len(target_size) == 2, f"{len(target_size)=}"
        assert type(target_size[0]) == type(target_size[1]) == int, f"{type(target_size[0])=}, {type(target_size[1])=}"
        self.target_size = target_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert type(x) == torch.Tensor, f"{type(x)=}"
        assert x.dim() == 2, f"{x.shape=}"
        x = x.unsqueeze(0)
        x = self.resize(x)
        x = x.squeeze(0)
        assert x.shape == self.target_size, f"{x.shape=}, {self.target_size=}"
        return x
