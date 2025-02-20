import torch


class PIM(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.prob = torch.nn.Conv2d(dim, dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.b = 0.5

    def forward(self, x1t, x2t):
        P1, P2 = self.sigmoid(self.prob(x1t)), self.sigmoid(self.prob(x2t))
        P1 = P1 * (1 - self.b) + self.b
        x = P1 * x1t + (1 - P1) * P2 * x2t + (1 - P1) * (1 - P2) * (x1t + x2t) / 2
        return x
