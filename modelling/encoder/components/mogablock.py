import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()

    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    """
    Learnable per-channel scaling.
    """

    def __init__(self, embed_dims, init_value=1e-5):
        super().__init__()

        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)),
            requires_grad=True
        )

    def forward(self, x):
        return x * self.scale


def compute_splits(embed_dims, ratios=(1, 3, 4)):
    """
    Dynamically compute channel splits.

    Works for very small channel counts.
    """
    #print(f"embed_dims: {embed_dims}")

    if embed_dims < 3:
        return [embed_dims, 0, 0]

    total = sum(ratios)

    splits = [
        int(embed_dims * r / total)
        for r in ratios
    ]

    # distribute remaining channels
    while sum(splits) < embed_dims:
        splits[splits.index(max(splits))] += 1
    # print(f"splits: {splits}")

    return splits


class ChannelAggregationFFN(nn.Module):

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        kernel_size=3,
        act_type='GELU',
        ffn_drop=0.
    ):
        super().__init__()

        self.fc1 = nn.Conv3d(
            embed_dims,
            feedforward_channels,
            kernel_size=1
        )

        self.dwconv = nn.Conv3d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=feedforward_channels
        )

        self.act = build_act_layer(act_type)

        self.fc2 = nn.Conv3d(
            feedforward_channels,
            embed_dims,
            kernel_size=1
        )

        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv3d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=1
        )

        self.sigma = ElementScale(feedforward_channels)

        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):

        x = x + self.sigma(
            x - self.decompose_act(self.decompose(x))
        )

        return x

    def forward(self, x):

        x = self.fc1(x)

        x = self.dwconv(x)

        x = self.act(x)

        x = self.drop(x)

        x = self.feat_decompose(x)

        x = self.fc2(x)

        x = self.drop(x)

        return x


class MultiOrderDWConv(nn.Module):

    def __init__(
        self,
        embed_dims,
        dw_dilation=(1, 2, 3),
        channel_split=(1, 3, 4),
    ):
        super().__init__()

        self.embed_dims = embed_dims

        (
            self.embed_dims_0,
            self.embed_dims_1,
            self.embed_dims_2
        ) = compute_splits(embed_dims, channel_split)

        # TODO change the kernel sizes to be 5, 5, 7 instead of 3, 3, 3
        # shared DW conv
        self.DW_conv0 = nn.Conv3d(
            embed_dims,
            embed_dims,
            kernel_size=5,
            padding=(2* dw_dilation[0]),
            dilation=dw_dilation[0],
            groups=embed_dims,
            bias=True
        )

        # branch 1
        if self.embed_dims_1 > 0:
            self.DW_conv1 = nn.Conv3d(
                self.embed_dims_1,
                self.embed_dims_1,
                kernel_size=5,
                padding=(2* dw_dilation[1]),
                dilation=dw_dilation[1],
                groups=self.embed_dims_1,
                bias=True
            )
        else:
            self.DW_conv1 = None

        # branch 2
        if self.embed_dims_2 > 0:
            self.DW_conv2 = nn.Conv3d(
                self.embed_dims_2,
                self.embed_dims_2,
                kernel_size=7,
                padding=(3* dw_dilation[2]),
                dilation=dw_dilation[2],
                groups=self.embed_dims_2,
                bias=True
            )
        else:
            self.DW_conv2 = None

        self.PW_conv = nn.Conv3d(
            embed_dims,
            embed_dims,
            kernel_size=1
        )

    def forward(self, x):

        # print("splits:", self.embed_dims_0, self.embed_dims_1, self.embed_dims_2)
        # print("sum:", self.embed_dims_0 + self.embed_dims_1 + self.embed_dims_2)

        x0 = self.DW_conv0(x)

        outputs = []

        idx = 0

        branch_feats = {}

        # split 0
        if self.embed_dims_0 > 0:

            b0 = x0[:, idx:idx+self.embed_dims_0]

            outputs.append(
                b0
            )

            idx += self.embed_dims_0

            branch_feats["d1"] = b0

        # split 1
        if self.embed_dims_1 > 0:

            b1 = x0[:, idx:idx+self.embed_dims_1]
            x1 = self.DW_conv1(
                b1
            )

            outputs.append(x1)

            idx += self.embed_dims_1

            branch_feats["d2"] = b1

        # split 2
        if self.embed_dims_2 > 0:

            b2 = x0[:, idx:idx+self.embed_dims_2]

            x2 = self.DW_conv2(
                b2
            )

            outputs.append(x2)

            branch_feats["d3"] = b2

        x = torch.cat(outputs, dim=1)

        x = self.PW_conv(x)

        self.branch_feats = branch_feats

        return x


class MultiOrderGatedAggregation(nn.Module):

    def __init__(
        self,
        embed_dims,
        attn_dw_dilation=(1, 2, 3),
        attn_channel_split=(1, 3, 4),
        attn_act_type='SiLU',
        attn_force_fp32=False,
    ):
        super().__init__()

        self.attn_force_fp32 = attn_force_fp32

        self.proj_1 = nn.Conv3d(
            embed_dims,
            embed_dims,
            kernel_size=1
        )

        self.gate = nn.Conv3d(
            embed_dims,
            embed_dims,
            kernel_size=1
        )

        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split
        )

        self.proj_2 = nn.Conv3d(
            embed_dims,
            embed_dims,
            kernel_size=1
        )

        self.act_gate = build_act_layer(attn_act_type)

        self.act_value = build_act_layer(attn_act_type)

        self.sigma = ElementScale(embed_dims)

    def feat_decompose(self, x):

        x = self.proj_1(x)

        # FIXED: adaptive_avg_pool3d
        x_d = F.adaptive_avg_pool3d(x, output_size=1)

        x = x + self.sigma(x - x_d)

        x = self.act_value(x)

        return x

    def forward_gating(self, g, v):

        with torch.autocast(device_type='cuda', enabled=False):

            g = g.float()

            v = v.float()

            return self.proj_2(g * v)

    def forward(self, x):

        shortcut = x

        x = self.feat_decompose(x)

        g = self.gate(x)

        v = self.value(x)

        g = self.act_gate(g)

        v = self.act_gate(v)

        if self.attn_force_fp32:
            x = self.forward_gating(g, v)
        else:
            x = self.proj_2(g * v)

        x = x + shortcut

        return x


class MogaBlock(nn.Module):

    def __init__(
        self,
        embed_dims,
        ffn_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.,
        act_type='GELU',
        init_value=1e-5,
        attn_dw_dilation=(1, 2, 3),
        attn_channel_split=(1, 3, 4),
        attn_act_type='SiLU',
        attn_force_fp32=False,
    ):
        super().__init__()

        self.norm1 = nn.InstanceNorm3d(embed_dims)

        self.attn = MultiOrderGatedAggregation(
            embed_dims=embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32
        )

        self.drop_path = DropPath(
            drop_path_rate
        ) if drop_path_rate > 0 else nn.Identity()

        self.norm2 = nn.InstanceNorm3d(embed_dims)

        mlp_hidden_dim = int(embed_dims * ffn_ratio)

        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate
        )

        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1))
        )

        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1))
        )

    def forward(self, x):

        # print("norm input shape:", x.shape)
        # print("expected channels:", self.norm1.num_features)

        # spatial block
        identity = x

        x = self.layer_scale_1 * self.attn(
            self.norm1(x)
        )

        x = identity + self.drop_path(x)

        # channel block
        identity = x

        x = self.layer_scale_2 * self.mlp(
            self.norm2(x)
        )

        x = identity + self.drop_path(x)

        return x
