import math
import torch
from models.change_detection.ftn.modules.patch.patch_expand import PatchExpand
from models.change_detection.ftn.modules.basic_layer_up import BasicLayer_up
from models.change_detection.ftn.modules.patch.patch_expand import FinalPatchExpand_X4, FinalPatchExpand_X4_1
from models.change_detection.ftn.modules.attention.attention import ppattention_wan, DFE
from models.change_detection.ftn.modules.attention.channel_attention import ChannelAttention


class SwinTransDecoder(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=128, depths=[4, 4, 4, 4], depths_decoder=[2, 2, 2, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.concat_linear0 = torch.nn.Linear(1024, 1024)

        self.norm = torch.nn.ModuleList()
        self.norm.append(torch.nn.LayerNorm(512))
        self.norm.append(torch.nn.LayerNorm(256))
        self.norm.append(torch.nn.LayerNorm(128))
        self.norm.append(torch.nn.LayerNorm(128))

        self.layers_up = torch.nn.ModuleList()
        self.concat_back_dim = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = torch.nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else torch.nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == "expand_first":
            # print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim, patchsize=patch_size)
            self.up_0 = FinalPatchExpand_X4_1(input_resolution=(56, 56), dim_scale=1, dim=128, patchsize=1)
            self.up_1 = FinalPatchExpand_X4_1(input_resolution=(28, 28), dim_scale=1, dim=256, patchsize=1)
            self.up_2 = FinalPatchExpand_X4_1(input_resolution=(14, 14), dim_scale=1, dim=512, patchsize=1)
            self.output = torch.nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_0 = torch.nn.Conv2d(in_channels=128, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_1 = torch.nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_2 = torch.nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.ppattention = torch.nn.ModuleList()
        self.ppattention.append(ppattention_wan(1024))
        self.ppattention.append(ppattention_wan(512))
        self.ppattention.append(ppattention_wan(256))
        self.ppattention.append(ppattention_wan(128))

        self.DFE = torch.nn.ModuleList()
        self.DFE.append(DFE(1024))
        self.DFE.append(DFE(512))
        self.DFE.append(DFE(256))
        self.DFE.append(DFE(128))

        self.norm_bn = torch.nn.ModuleList()
        self.norm_bn.append(torch.nn.BatchNorm2d(2048))
        self.norm_bn.append(torch.nn.BatchNorm2d(1024))
        self.norm_bn.append(torch.nn.BatchNorm2d(512))
        self.norm_bn.append(torch.nn.BatchNorm2d(256))

        self.channelattention = ChannelAttention(1024)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

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

    def forward_up_features(self, x_mid, x_downsample1, x_downsample2):  # 1/4,1/8,1/16,1/32,     1/32
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                t1 = self.transpose(x_downsample1[3])  # B,C,H,W    mlp
                t2 = self.transpose(x_downsample2[3])
                diff_feature = t1 - t2  # 1024
                add_feature = t1 + t2

                x_mid = self.transpose(x_mid)
                x_mid = torch.cat([x_mid, x_mid], dim=1)
                # hidden = torch.cat([x_mid,t1,t2],dim=1)
                hidden = torch.cat([diff_feature, add_feature], dim=1)
                hidden = x_mid + hidden
                # hidden = self.norm_bn[inx](hidden)
                x = self.transpose_verse(self.ppattention[inx](hidden))  # B,HW,C

                x = self.concat_linear0(x)  # C,C  1024
                y1 = layer_up(x)  # C/2  UP SAMPLE  512
                y2 = y1  # B,HW,C/2     512
                x = torch.cat([y1, y2], dim=2)  # C    B,HW,C     1024
                x_upsample.append(self.norm[0](y1))  # B,HW,C/2
            else:
                t1 = self.transpose(x_downsample1[3 - inx])  # 512
                t2 = self.transpose(x_downsample2[3 - inx])
                diff_feature = t1 - t2
                add_feature = t1 + t2

                x = self.transpose(x)  # 1024
                # hidden = torch.cat([x,t1,t2],dim=1)
                hidden = torch.cat([diff_feature, add_feature], dim=1)
                hidden = x + hidden
                # hidden = self.norm_bn[inx](hidden)
                x = self.ppattention[inx](hidden)
                x = self.transpose_verse(x)  # B,HW,C
                x = self.concat_back_dim[inx](x)  ######

                y1 = layer_up(x)  # layer up 初始层有norm,up  norm,norm,up
                y2 = y1
                x = torch.cat([y1, y2], dim=2)  # C1024
                norm = self.norm[inx]
                x_upsample.append((norm(y1)))

        x = self.norm_up(y1)  # B L C   最终预测结果

        return x, x_upsample

    def up_x4(self, x, pz):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def up_x4_1(self, x, pz):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up_0(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output_0(x)

        return x

    def up_x8(self, x, pz):
        H, W = (28, 28)
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            # x = self.up(x,patchsize=pz)
            x = self.up_1(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output_1(x)

        return x

    def up_x16(self, x, pz):
        H, W = (14, 14)
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            # x = self.up(x,patchsize=pz)
            x = self.up_2(x)
            x = x.view(B, pz * H, pz * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output_2(x)

        return x

    def forward(self, x, x_down1, x_down2):
        x, x_upsample = self.forward_up_features(x, x_down1, x_down2)

        x_p = self.up_x4(x, self.patch_size)
        x_pre2 = self.up_x4_1(x_upsample[2], 1)
        x_pre3 = self.up_x8(x_upsample[1], 1)
        x_pre4 = self.up_x16(x_upsample[0], 1)

        return x_p, x_pre2, x_pre3, x_pre4
