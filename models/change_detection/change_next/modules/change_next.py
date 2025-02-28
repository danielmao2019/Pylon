import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2,
                      out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rate = 2
        self.num_scales = 4
        self.freq_adapt_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.freq_conv_weight_real = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.freq_conv_weight_imag = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.freq_conv_bias = None
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_3 = Partial_conv3(dim, 4, 'split_cat')
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim,
                                 dim, (1, 21),
                                 padding=(0, 10),
                                 groups=dim)
        self.conv3_2 = nn.Conv2d(dim,
                                 dim, (21, 1),
                                 padding=(10, 0),
                                 groups=dim)

        self.conv4 = nn.Conv2d(dim, dim, 1)

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.rate) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p). \
            repeat_interleave(scale, axis=1). \
            repeat_interleave(scale, axis=2)

    def frequency_mask(self, x, scale):

        fft_x = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))

        _, _, H, W = fft_shifted.shape
        center_h, center_w = H // 2, W // 2
        h_start, h_end = center_h - scale, center_h + scale
        w_start, w_end = center_w - scale, center_w + scale
        mask = torch.zeros_like(fft_shifted)
        mask[:, :, h_start:h_end, w_start:w_end] = 1

        masked_freq = self.dwconv_in_freq(fft_shifted * mask)

        ifft_shifted = torch.fft.ifftshift(masked_freq, dim=(-2, -1))
        mask = torch.fft.ifft2(ifft_shifted)
        mask = torch.abs(mask)

        return mask

    def dwconv_in_freq(self, x_freq):
        groups = x_freq.shape[1]

        weight_real = self.freq_conv_weight_real.repeat(groups, 1, 1, 1)
        weight_imag = self.freq_conv_weight_imag.repeat(groups, 1, 1, 1)

        x_freq_real = F.conv2d(x_freq.real, weight_real, self.freq_conv_bias, groups=groups)
        x_freq_imag = F.conv2d(x_freq.imag, weight_imag, self.freq_conv_bias, groups=groups)

        x_freq_processed = torch.complex(x_freq_real, x_freq_imag)
        return x_freq_processed

    def frequency_mask2(self, x, scale):

        fft_x = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))

        _, _, H, W = fft_shifted.shape
        center_h, center_w = H // 2, W // 2
        freq_mask = torch.zeros_like(fft_shifted)
        for i in range(H):
            for j in range(W):
                dist = math.sqrt((i - center_h) ** 2 + (j - center_w) ** 2)
                freq_mask[:, :, i, j] = self.gaussian_weight(dist, scale)

        masked_freq = fft_shifted * freq_mask
        ifft_shifted = torch.fft.ifftshift(masked_freq, dim=(-2, -1))
        mask = torch.fft.ifft2(ifft_shifted)
        mask = torch.abs(mask)

        return mask

    def gaussian_weight(self, dist, scale):
        base_sigma = 0.1
        sigma = base_sigma * (2 ** scale)
        if sigma == 0:
            return 1
        return math.exp(-dist ** 2 / (2 * sigma ** 2))

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        masks = [self.frequency_mask(x, scale) for scale in range(0, self.num_scales + 1)]
        freq_mask = sum(masks) / self.num_scales

        freq_mask = self.freq_adapt_conv(freq_mask)
        attn *= freq_mask

        attn_0 = self.conv0_3(attn)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)

        attn = attn + attn_0 + attn_1 + attn_2 + attn_3

        attn = self.conv4(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,

                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MSCAN(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4):
        super(MSCAN, self).__init__()

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
               ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = nn.ModuleList([
                Block(dim=embed_dims[i],
                      mlp_ratio=mlp_ratios[i],
                      drop=drop_rate,
                      drop_path=dpr[cur + j]) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    nn.init.normal_(m,
                                    mean=0,
                                    std=math.sqrt(2.0 / fan_out),
                                    bias=0)
        elif isinstance(pretrained, str):
            self.load_parameters(torch.load(pretrained))

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs
