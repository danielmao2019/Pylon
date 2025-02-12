import torch
from models.change_detection.change_former.modules.mlp import MLP
from models.change_detection.change_former.utils.resize import resize


class TDecV2(torch.nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                    in_channels = [64, 128, 256, 512], embedding_dim= 256, output_nc=2,
                    feature_strides=[4, 8, 16, 32]):
        super(TDecV2, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.linear_fuse = torch.nn.Conv2d(in_channels=self.embedding_dim*4, out_channels=self.embedding_dim,
                                        kernel_size=1)

        #Pixel Shiffle
        self.pix_shuffle_conv   = torch.nn.Conv2d(in_channels=self.embedding_dim, out_channels=16*output_nc, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pix_shuffle        = torch.nn.PixelShuffle(4)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_1 = resize(_c4_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4_2 = resize(_c4_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_1 = resize(_c3_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3_2 = resize(_c3_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_1 = resize(_c2_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2_2 = resize(_c2_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])

        _c = self.linear_fuse(torch.cat([torch.abs(_c4_1-_c4_2), torch.abs(_c3_1-_c3_2), torch.abs(_c2_1-_c2_2), torch.abs(_c1_1-_c1_2)], dim=1))

        x  = self.relu(self.pix_shuffle_conv(_c))
        cp = self.pix_shuffle(x)

        return cp
