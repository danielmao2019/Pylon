import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmseg.models.utils import SelfAttentionBlock


class Class_Token_Seg3(nn.Module):
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, num_classes=150, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.zeros(1, num_classes, dim))
        self.prop_token = nn.Parameter(torch.zeros(1, num_classes, dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.prop_token, std=.02)

    def forward(self, x):  # , x1):
        b, c, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        prop_tokens = self.prop_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x[:, 0:self.num_classes]).unsqueeze(1).reshape(B, self.num_classes, self.num_heads,
                                                                  C // self.num_heads).permute(0, 2, 1, 3)

        k = k * self.scale
        attn = (k @ q.transpose(-2, -1)).squeeze(1).transpose(-2, -1)
        attn = attn[:, self.num_classes:]
        x_cls = attn.permute(0, 2, 1).reshape(b, -1, h, w)
        return x_cls, prop_tokens


class TransformerClassToken3(nn.Module):

    def __init__(self, dim, num_heads=2, num_classes=150, depth=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_cfg=None, norm_cfg=None, sr_ratio=1, trans_with_mlp=True, att_type="SelfAttention"):
        super().__init__()
        self.trans_with_mlp = trans_with_mlp
        self.depth = depth
        print("TransformerOriginal initial num_heads:{}; depth:{}, self.trans_with_mlp:{}".format(num_heads, depth,
                                                                                                  self.trans_with_mlp))
        self.num_classes = num_classes

        self.attn = SelfAttentionBlock(
            key_in_channels=dim,
            query_in_channels=dim,
            channels=dim,
            out_channels=dim,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.cross_attn = SelfAttentionBlock(
            key_in_channels=dim,
            query_in_channels=dim,
            channels=dim,
            out_channels=dim,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, cls_tokens, out_cls_mid):
        b, c, h, w = x.size()
        out_cls_mid = out_cls_mid.flatten(2).transpose(1, 2)

        # within images attention
        x1 = self.attn(x, x)

        # cross images attention
        out_cls_mid = out_cls_mid.softmax(dim=-1)
        cls = out_cls_mid @ cls_tokens  # bxnxc

        cls = cls.permute(0, 2, 1).reshape(b, c, h, w)
        x2 = self.cross_attn(x, cls)

        x = x + x1 + x2

        return x


def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


norm_cfg = dict(type='BN', requires_grad=True)
act_cfg=dict(type='ReLU')


class ChangeNeXtDecoder(nn.Module):
    """The all mlp Head of segformer.
    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True, trans_depth=1,
                 att_type="XCA",in_channels=[64, 128, 320, 512],in_index=[0, 1, 2, 3],channels=256,
        dropout_ratio=0.1,num_classes=2,input_transform='multiple_select',align_corners=False,feature_strides=[2, 4, 8, 16],embedding_dim=256,output_nc=2,decoder_softmax=False):
        super(ChangeNeXtDecoder, self).__init__()
        self.in_channels =in_channels
        self.in_index = in_index
        self.channels = 256,
        self.dropout_ratio = 0.1
        self.num_classes = 150
        self.input_transform = 'multiple_select'
        self.align_corners = False
        self.norm_cfg=norm_cfg
        self.act_cfg = act_cfg
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        self.num_heads = num_heads
        assert num_inputs == len(self.in_index)

        self.class_token = Class_Token_Seg3(dim=channels, num_heads=1, num_classes=num_classes)
        self.trans = TransformerClassToken3(dim=channels, depth=trans_depth, num_heads=num_heads,
                                            trans_with_mlp=trans_with_mlp, att_type=att_type, norm_cfg=norm_cfg,
                                            act_cfg=act_cfg)

        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        self.fusion_conv = ConvModule(
            in_channels=self.embedding_dim * num_inputs,
            out_channels=self.embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        # Final prediction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

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

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output

    def forward(self, input1,input2,gt_semantic_seg=None):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        x_1 = self._transform_inputs(input1)
        x_2 = self._transform_inputs(input2)
        outputs = []
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2
        n, _, h, w = c4_1.shape

        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1)) #[2, 256, 8, 8]
        p_c4 = self.make_pred_c4(_c4)                #[2, 2, 8, 8]]
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")  #[2, 256, 16, 16]

        p_c3 = self.make_pred_c3(_c3)                 #[2, 2, 16, 16]
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear") #[2, 256, 32, 32]
        p_c2 = self.make_pred_c2(_c2)   #[2, 2, 32, 32]
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")  #[2, 256, 64, 64]
        p_c1 = self.make_pred_c1(_c1)   #[2, 2, 64, 64]
        outputs.append(p_c1)
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        out_cls_mid, cls_tokens = self.class_token(_c)
        out_new = self.trans(_c, cls_tokens, out_cls_mid)

        # Linear Fusion of difference image from all scales
        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(out_new)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class PagFM2(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM2, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        # Spatial features processing
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,bias=False),
            BatchNorm(mid_channels)
        )
        # Frequency features processing
        self.f_freq = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1,bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        # Process spatial features
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        # Extract and process frequency features
        x_freq = torch.fft.fft2(x)
        y_freq = torch.fft.fft2(y)
        freq_features = torch.abs(x_freq - y_freq)  # Difference in frequency domain
        freq_features = self.f_freq(freq_features)
        freq_features = F.interpolate(freq_features, size=[input_size[2], input_size[3]], mode='bilinear',
                                      align_corners=False)

        # Combine spatial and frequency attention
        if self.with_channel:
            spatial_sim_map = torch.sigmoid(self.up(x_k * y_q))
            freq_sim_map = torch.sigmoid(self.up(freq_features))
            sim_map = spatial_sim_map * freq_sim_map
        else:
            spatial_sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
            freq_sim_map = torch.sigmoid(torch.sum(freq_features, dim=1).unsqueeze(1))
            sim_map = spatial_sim_map * freq_sim_map

        # Interpolate and fuse features based on combined attention
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x
