import torch
from .lwga_block import LWGA_Block

class BasicStage(torch.nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):

        super().__init__()

        blocks_list = [
            LWGA_Block(
                dim=dim,
                stage=stage,
                att_kernel=att_kernel,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
                 )
            for i in range(depth)
        ]

        self.blocks = torch.nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x