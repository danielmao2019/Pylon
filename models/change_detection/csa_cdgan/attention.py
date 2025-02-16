import torch

   
class csa_layer(torch.nn.Module):
        
    def __init__(self, channel):
        super(csa_layer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.query_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.key_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.value_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(-1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        query = self.query_conv(y).transpose(-1, -2)
        key = self.key_conv(y)
        attention = torch.bmm(query, key)
        attention = self.sm(attention)
        value = self.value_conv(y)
        out = torch.bmm(value, attention).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        x = x * out.expand_as(x)
        return x
