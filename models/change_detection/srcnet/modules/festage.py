import torch


class FEStage(torch.nn.Module):

    def __init__(self, dim, n1):
        super().__init__()
        self.n1 = n1
        self.blocks = torch.nn.ModuleList([SRCBlock(dim) for _ in range(n1)])
        self.checks = torch.nn.ModuleList([PIM(dim) for _ in range(n1)])

    def forward(self, x1, x2):
        diffList = torch.tensor(0.0, dtype=x1.dtype, device=x1.device)
        for num in range(0, self.n1):
            chk = self.checks[num]
            blk = self.blocks[num]
            x1, x2 = blk(x1), blk(x2)
            x1w = self.WindowMaskSimple(x1)
            diff = chk(x1, x1w) - x1
            diffList += torch.mean(diff * diff)
            x2w = self.NoiseSimple(x2)
            diff = chk(x2, x2w) - x2
            diffList += torch.mean(diff * diff)
            x1, x2 = chk(x1, x2), chk(x2, x1)
        return x1, x2, diffList / self.n1 / 2

    def WindowMaskSimple(self, x, drop_prob=0.5):
        # shape = x.shape
        keep_prob = 1 - drop_prob
        random_tensor = torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        random_tensor = random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor

        shape = (x.shape[0], 1, 1, x.shape[3])
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor + keep_prob
        random_tensor = random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def NoiseSimple(self, x, drop_prob=0.5):
        # shape = x.shape
        random_tensor = torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor * 2 - 1) * drop_prob + 1
        return x * random_tensor
