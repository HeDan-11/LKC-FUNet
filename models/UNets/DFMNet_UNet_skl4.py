import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from models.UNets.replknet import RepLKBlock, RepLKBlock_in_out
from models.UNets.Largekernel import DilatedReparamBlock, get_conv2d
from models.UNets.masag import MultiScaleGatedAttn
import matplotlib.pyplot as plt
import os

class BatchChannelNorm(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.Batchh = nn.BatchNorm2d(self.num_channels)
        self.layeer = nn.GroupNorm(1, self.num_channels)
        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var.view([1, self.num_channels, 1, 1]) * X + (
                1 - self.BCN_var.view([1, self.num_channels, 1, 1])) * Y
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0, conv_size=3, GN=True):
        # groups = 4
        super().__init__()
        self.block = nn.Sequential(
            # BatchChannelNorm(dim),
            # nn.BatchNorm2d(dim),
            # nn.GroupNorm(1, dim),
            nn.GroupNorm(groups, dim),
            nn.ReLU(),
            # Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, conv_size, padding=conv_size//2)
            # nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32, conv_size=3, GN=True):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups, conv_size=conv_size, GN=GN)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout, conv_size=conv_size, GN=GN)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, conv_size=3, is_res = 1):
        super().__init__()
        # conv_size = 15
        self.with_attn = with_attn
        
        if dim == dim_out and is_res!=0:
        # if dim == dim_out:
            self.res_block = RepLKBlock(in_channels=dim, dw_ratio=1.5, kernel_size=conv_size, small_kernel=None, drop_path=dropout, small_kernel_merged=False)
            # self.res_block = ResnetBlock(dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, conv_size=conv_size, GN=False)
        else:
            self.res_block = ResnetBlock(dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, conv_size=conv_size)
            # RepLKBlock_in_out(in_channels=dim, dw_ratio=1.5, kernel_size=5, small_kernel=None, drop_path=0.2, small_kernel_merged=False)

    def forward(self, x, time_emb):
        x = self.res_block(x)
        return x


class UNet(nn.Module):
    def __init__(
            self,
            image_size=128,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(4, 4, 4),
            conv_sizes=[3, 3, 3],
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            save_fea_path=None
    ):
        super().__init__()
        num_mults = len(channel_mults)  # (4, 4, 4)
        pre_channel = inner_channel*channel_mults[0]  # 32
        feat_channels = [inner_channel]
        now_res = image_size

        self.init_conv = nn.Conv2d(in_channels=in_channel, out_channels=inner_channel, kernel_size=3, padding=1)
        self.IN_conv = nn.Sequential(
            # nn.GroupNorm(inner_channel, inner_channel),
            # nn.GroupNorm(8, inner_channel),
            nn.BatchNorm2d(inner_channel),
            # BatchChannelNorm(inner_channel),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(inner_channel, pre_channel, 15, padding=7)
        )
        downs = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            use_attn = False
            channel_mult = inner_channel * channel_mults[ind]
            conv_size = conv_sizes[ind]
            for l_i in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=None, norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn, conv_size=conv_size, is_res = l_i))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)
        conv_size = conv_sizes[-1]
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=None, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, conv_size=conv_size), # True
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=None, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, conv_size=conv_size)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            use_attn = False
            channel_mult = inner_channel * channel_mults[ind]
            conv_size = conv_sizes[ind]
            for i in range(0, res_blocks + 1):
                if i == 0 or (is_last and i==res_blocks):
                    cat_c = feat_channels.pop()
                else:
                    cat_c = 0
                    feat_channels.pop()
                if (cat_c == pre_channel) and (i == 0):
                    ups.append(MultiScaleGatedAttn(pre_channel))
                    # cat_c = 0
                ups.append(
                    ResnetBlocWithAttn(pre_channel + cat_c, channel_mult, noise_level_emb_dim=None, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn, conv_size=conv_size)
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2
                

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        self.save_fea_path = save_fea_path

    def forward(self, x, time=None, feat_need=False):
        x = self.init_conv(x)
        # feats = []
        feats = [x]
        x = self.IN_conv(x)
        if self.save_fea_path is not None:
            new_path = os.path.join(self.save_fea_path, 'IN_conv')
            os.makedirs(new_path, exist_ok=True)
            for i in range(round(x.shape[1]/36)):
                plt.figure(figsize=(10, 10))
                for j in range(32):
                    data_x = x[:, 36*i+j, :, :].squeeze(0).cpu().numpy()
                    ax = plt.subplot(6, 6, j+1)
                    ax.imshow(data_x, cmap="gray")
                    plt.axis('off')
                plt.subplots_adjust(wspace=0.02, hspace=0.02)
                plt.savefig('{}/hb_{}.jpg'.format(new_path, str(i)),dpi=300)
                plt.close()

        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, time)
            else:
                feats.append(x)
                x = layer(x)
        feats.append(x)

        if feat_need:
            fe = feats.copy()

        # Passing through middle layer
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, time)
            else:
                x = layer(x)

        # Saving decoder features for CD Head
        if feat_need:
            fd = []
        # Diffiusion decoder
        # x = torch.cat((x, feats.pop()), dim=1)
        total_num = len(self.ups)
        for layer in self.ups:
            total_num = total_num -1
            if total_num == 0:
                x = torch.cat((x, feats.pop()), dim=1)
            if isinstance(layer, MultiScaleGatedAttn):
                y = feats.pop()
                # print(x.shape, y.shape)
                # x = layer(y, x)
                x = layer(x, y)
            elif isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, time)
                if feat_need:
                    # fd.append(x.clone().detach())
                    fd.append(x)
            else:
                x = layer(x)
                # x = torch.cat((x, feats.pop()), dim=1)

        # Final Diffusion layer
        x = self.final_conv(x)
        return x


class Model(nn.Module):
    def __init__(self, save_fea_path=None):
        super(Model, self).__init__()

        self.model = UNet(image_size=64,
                          in_channel=2,
                          out_channel=1,
                          inner_channel=32,
                          norm_groups=32,
                          channel_mults=(4, 4, 4),
                          conv_sizes=(15, 11, 5, 5),
                          attn_res=[16],
                          res_blocks=2,
                          dropout=0.2,
                          save_fea_path=save_fea_path
                          )

    def forward(self, Other_img, MRI_img, img_cat):
        out = self.model(img_cat)
        return out