from typing import Tuple
import torch
import utils


class Generator(torch.nn.Module):

    def __init__(self, latent_dim: int, img_shape: Tuple[int]) -> None:
        super(Generator, self).__init__()
        self.img_shape = torch.Tensor(img_shape)
        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(torch.prod(self.img_shape))),
            torch.nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(torch.nn.Module):

    def __init__(self, img_shape: Tuple[int]) -> None:
        super(Discriminator, self).__init__()
        self.img_shape = torch.Tensor(img_shape)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(torch.prod(self.img_shape)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class GAN(torch.nn.Module):

    def __init__(self, generator_cfg: dict, discriminator_cfg: dict) -> None:
        super(GAN, self).__init__()
        self.generator = utils.builder.build_from_config(generator_cfg)
        self.discriminator = utils.builder.build_from_config(discriminator_cfg)

    def forward(self):
        raise NotImplementedError("GAN.forward not implemented.")
