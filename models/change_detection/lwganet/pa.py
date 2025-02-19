import torch

class PA(torch.nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim*4, 1, bias=False),
            norm_layer(dim*4),
            act_layer(),
            torch.nn.Conv2d(dim*4, dim, 1, bias=False)
        )
        self.gate_fn = torch.nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x
