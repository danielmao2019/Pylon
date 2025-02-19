import torch
from torchvision.models import vgg16


class vgg16_base(torch.nn.Module):

    def __init__(self):
        super(vgg16_base,self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22,29}:
                results.append(x)
        return results
