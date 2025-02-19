import torch

'''
Todo:
- add reference
'''

class Stem(torch.nn.Module):

    def __init__(self, in_chans, stem_dim, norm_layer):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_chans, stem_dim, kernel_size=4, stride=4, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(stem_dim)
        else:
            self.norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.proj(x))
        return x