import torch
from .ga import GA

class D_GA(torch.nn.Module):

    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA(dim)
        self.downpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = torch.nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x