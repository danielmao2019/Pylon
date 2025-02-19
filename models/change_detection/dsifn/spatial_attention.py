import torch


class SpatialAttention(torch.nn.Module):

    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = torch.nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
