import torch


class PMFFM(torch.nn.Module):

    def __init__(self, dim, k, m):
        super().__init__()
        self.k = k
        self.m = m
        self.prob = torch.nn.Linear(dim // k, m)
        self.softmax = torch.nn.Softmax(dim=4)
        self.result = torch.nn.Conv2d(dim // k * m, dim // k * m, kernel_size=1, groups=m)
        self.d1line = torch.tensor(
            [1] * (self.m // 2) + [1] * (self.m // 4) + [-1] * (self.m // 4),
            device=torch.device("cuda"),
        ).reshape([1, 1, self.m, 1, 1, 1])
        self.d2line = torch.tensor(
            [1] * (self.m // 2) + [-1] * (self.m // 4) + [1] * (self.m // 4),
            device=torch.device("cuda"),
        ).reshape([1, 1, self.m, 1, 1, 1])

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        m = x1 + x2
        m = m.view([B, self.k, C // self.k, H, W])
        m = m.permute(0, 1, 3, 4, 2).contiguous()
        prob = self.prob(m)
        prob = self.softmax(prob)

        d1, d2 = x1, x2
        d1 = d1.reshape([B, self.k, 1, C // self.k, H, W]).repeat(1, 1, self.m, 1, 1, 1)
        d2 = d2.reshape([B, self.k, 1, C // self.k, H, W]).repeat(1, 1, self.m, 1, 1, 1)
        d = d1 * self.d1line + d2 * self.d2line

        d = d.reshape(B * self.k, self.m * C // self.k, H, W)
        result = self.result(d)
        result = result.reshape(B, self.k, self.m, C // self.k, H, W)

        prob = prob.unsqueeze(4)
        result = result.permute(0, 1, 4, 5, 3, 2).contiguous()
        out = result * prob
        out = torch.sum(out, dim=5)

        out = out.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        return out
