from typing import List
import torch
import timm

from .pa import PA
from .la import LA
from .mra import MRA
from .d_ga import D_GA
from .ga import GA
from .ga12 import GA12

class LWGA_Block(torch.nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = timm.models.layers.DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[torch.nn.Module] = [
            torch.nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            torch.nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = torch.nn.Sequential(*mlp_layer)

        self.PA = PA(self.dim_split, norm_layer, act_layer)     # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)     # LA is local attention
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention
        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)         # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)                       # GA4 is global attention (stage of 4)
            self.norm = norm_layer(self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)         # GA12 is global attention (stages of 1 and 2)
            self.norm = norm_layer(self.dim_split)
        self.norm1 = norm_layer(dim)
        self.drop_path = timm.models.layers.DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))
        x_att = torch.cat((x1, x2, x3, x4), 1)

        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return x