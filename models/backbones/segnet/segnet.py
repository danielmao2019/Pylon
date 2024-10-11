import torch


class SegNet(torch.nn.Module):
    __doc__ = r"""Reference: https://github.com/Cranial-XIX/CAGrad/blob/main/cityscapes/model_segnet_split.py
    """

    def __init__(self, wide: bool, deep: bool):
        super().__init__()
        assert type(wide) == type(deep) == bool
        self.wide = wide
        self.deep = deep
        if self.wide:
            self.filters = [64, 128, 256, 512, 1024]
        else:
            self.filters = [64, 128, 256, 512, 512]
        self._init_encoder_decoder_()
        self._init_conv_layers_()
        self._init_pool_layers_()

    def _init_encoder_decoder_(self):
        self.encoder_block = torch.nn.ModuleList([self.conv_layer(in_channels=3, out_channels=self.filters[0])])
        self.decoder_block = torch.nn.ModuleList([self.conv_layer(in_channels=self.filters[0], out_channels=self.filters[0])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer(in_channels=self.filters[i], out_channels=self.filters[i + 1]))
            self.decoder_block.append(self.conv_layer(in_channels=self.filters[i + 1], out_channels=self.filters[i]))

    def _init_conv_layers_(self):
        self.conv_block_enc = torch.nn.ModuleList([self.conv_layer(in_channels=self.filters[0], out_channels=self.filters[0])])
        self.conv_block_dec = torch.nn.ModuleList([self.conv_layer(in_channels=self.filters[0], out_channels=self.filters[0])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer(in_channels=self.filters[i + 1], out_channels=self.filters[i + 1])
                )
                self.conv_block_dec.append(
                    self.conv_layer(in_channels=self.filters[i], out_channels=self.filters[i])
                )
            else:
                self.conv_block_enc.append(torch.nn.Sequential(
                    self.conv_layer(in_channels=self.filters[i + 1], out_channels=self.filters[i + 1]),
                    self.conv_layer(in_channels=self.filters[i + 1], out_channels=self.filters[i + 1]),
                ))
                self.conv_block_dec.append(torch.nn.Sequential(
                    self.conv_layer(in_channels=self.filters[i], out_channels=self.filters[i]),
                    self.conv_layer(in_channels=self.filters[i], out_channels=self.filters[i]),
                ))

    def _init_pool_layers_(self):
        self.down_sampling = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)

    def conv_layer(self, in_channels: int, out_channels: int):
        if self.deep:
            conv_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.ReLU(inplace=True),
            )
        else:
            conv_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.ReLU(inplace=True)
            )
        return conv_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # initialize
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        # forward pass of encoder
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        # forward pass of decoder
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        return g_decoder[i][1]
