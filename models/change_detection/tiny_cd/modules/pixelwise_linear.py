from typing import List
import torch


class PixelwiseLinear(torch.nn.Module):

    def __init__(
        self,
        fin: List[int],
        fout: List[int],
    ) -> None:
        assert len(fout) == len(fin)
        super(PixelwiseLinear, self).__init__()

        n = len(fin)
        self._linears = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    torch.nn.PReLU()
                )
                for i in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Processing the tensor:
        return self._linears(x)
