import torch
import torch.nn.functional as F
from models.change_detection.change_former.modules.conv_layer import ConvLayer
from models.change_detection.change_former.modules.upsample_conv_layer import UpsampleConvLayer
from models.change_detection.change_former.modules.residual_block import ResidualBlock
from models.change_detection.change_former.modules.mlp import MLP
from models.change_detection.change_former.utils.conv_diff import conv_diff
from models.change_detection.change_former.utils.make_prediction import make_prediction
from models.change_detection.change_former.utils.resize import resize


class DecoderTransformer_x2(torch.nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3, 4], align_corners=True, 
                    in_channels = [32, 64, 128, 256, 512], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16, 32]):
        super(DecoderTransformer_x2, self).__init__()
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

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c5_in_channels = self.in_channels

        self.linear_c5 = MLP(input_dim=c5_in_channels, embed_dim=self.embedding_dim)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #Convolutional Difference Modules
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)

        #Taking outputs from middle of the encoder
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        self.linear_fuse = torch.nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = torch.nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = torch.nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = torch.nn.Sequential(ResidualBlock(self.embedding_dim))

        #Final prediction
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = torch.nn.Sigmoid() 

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
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2,1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2,1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1, c5_1 = x_1
        c1_2, c2_2, c3_2, c4_2, c5_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c5_1.shape

        outputs = [] #Multi-scale outputs adding here
        
        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        _c5   = self.diff_c5(torch.cat((_c5_1, _c5_2), dim=1)) #Difference of features at x1/32 scale
        p_c5  = self.make_pred_c5(_c5) #Predicted change map at x1/32 scale
        outputs.append(p_c5) #x1/32 scale
        _c5_up= resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((F.interpolate(_c5, scale_factor=2, mode="bilinear"), _c4_1, _c4_2), dim=1)) #Difference of features at x1/16 scale
        p_c4  = self.make_pred_c4(_c4) #Predicted change map at x1/16 scale
        outputs.append(p_c4) #x1/16 scale
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((F.interpolate(_c4, scale_factor=2, mode="bilinear"), _c3_1, _c3_2), dim=1)) #Difference of features at x1/8 scale
        p_c3  = self.make_pred_c3(_c3) #Predicted change map at x1/8 scale
        outputs.append(p_c3) #x1/8 scale
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((F.interpolate(_c3, scale_factor=2, mode="bilinear"), _c2_1, _c2_2), dim=1)) #Difference of features at x1/4 scale
        p_c2  = self.make_pred_c2(_c2) #Predicted change map at x1/4 scale
        outputs.append(p_c2) #x1/4 scale
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((F.interpolate(_c2, scale_factor=2, mode="bilinear"), _c1_1, _c1_2), dim=1)) #Difference of features at x1/2 scale
        p_c1  = self.make_pred_c1(_c1) #Predicted change map at x1/2 scale
        outputs.append(p_c1) #x1/2 scale

        _c = self.linear_fuse(torch.cat((_c5_up, _c4_up, _c3_up, _c2_up, _c1), dim=1))

        x = self.convd2x(_c)
        x = self.dense_2x(x)
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
