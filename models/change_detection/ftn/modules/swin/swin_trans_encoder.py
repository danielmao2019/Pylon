import math
import torch
from timm.models.layers import trunc_normal_
from models.change_detection.ftn.modules.patch.patch_embed import PatchEmbed
from models.change_detection.ftn.modules.patch.patch_merging import PatchMerging
from models.change_detection.ftn.modules.basic_layer import BasicLayer
from models.change_detection.ftn.modules.attention.channel_attention import ChannelAttention_1


class SwinTransEncoder(torch.nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (torch.nn.Module): Normalization layer. Default: torch.nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=192, depths=[2, 2, 18, 2], depths_decoder=[4, 4, 4, 4], num_heads=[6, 12, 24, 48],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        self.deal = torch.nn.ModuleList()
        self.deal.append(torch.nn.LayerNorm(128))
        self.deal.append(torch.nn.LayerNorm(256))
        self.deal.append(torch.nn.LayerNorm(512))
        self.deal.append(torch.nn.LayerNorm(1024))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.fusion = ChannelAttention_1(1024)

    # Encoder and Bottleneck
    def transpose(self, x):
        B, HW, C = x.size()
        H = int(math.sqrt(HW))
        x = x.transpose(1, 2)
        x = x.view(B, C, H, H)
        return x

    def transpose_verse(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        return x

    def forward(self, x1, x2):
        # input [1,3,224,224]
        x1 = self.patch_embed(x1)  # [1,56x56,96(embedding dim)]
        x2 = self.patch_embed(x2)  # [1,56x56,96(embedding dim)]
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        x1_downsample = []
        x2_downsample = []

        for inx, layer in enumerate(self.layers):
            if inx != 3:
                x1_downsample.append(self.deal[inx](x1))  # self.deal[inx](x))  #self.deal[inx]
                x2_downsample.append(self.deal[inx](x2))
                x1 = layer(x1)  # ??norm(  mlp(layer_norm(self-attention(x)))
                x2 = layer(x2)
            else:
                x1_downsample.append(self.deal[inx](x1))  # self.deal[inx](x))  #self.deal[inx]
                x2_downsample.append(self.deal[inx](x2))
                x_mid = self.transpose_verse(self.fusion(self.transpose(torch.cat([x1, x2], dim=2))))
                x_mid = layer(x_mid)
        x_mid = self.norm(x_mid)  # B L C --1 49 768]

        return x_mid, x1_downsample, x2_downsample
