import torch


def init_linear(linear):
    torch.nn.init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    torch.nn.init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()
