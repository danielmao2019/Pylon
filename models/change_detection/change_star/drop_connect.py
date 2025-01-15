"""Reference: https://github.com/Z-Zheng/ChangeStar/blob/master/core/head.py
"""
import torch


class DropConnect(torch.nn.Module):

    def __init__(self, drop_rate):
        super(DropConnect, self).__init__()
        self.p = drop_rate

    def forward(self, inputs):
        """Drop connect.
            Args:
                input (tensor: BCWH): Input of this structure.
                p (float: 0.0~1.0): Probability of drop connection.
                training (bool): The running mode.
            Returns:
                output: Output after drop connection.
        """
        p = self.p
        assert 0 <= p <= 1, 'p must be in range of [0,1]'

        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - p

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)

        output = inputs / keep_prob * binary_tensor
        return output
