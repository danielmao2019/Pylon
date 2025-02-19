import torch


class MixingBlock(torch.nn.Module):

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super(MixingBlock, self).__init__()
        self._convmix = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            torch.nn.PReLU(),
            torch.nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)
