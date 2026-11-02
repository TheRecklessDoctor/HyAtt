import torch
from torch import nn
# from hydra.utils import instantiate
# from omegaconf import DictConfig
# from typing import Optional
# from copy import deepcopy

class ResNetBlock(nn.Module):
    def __init__(
            self,
            n_filters_in: int,
            n_filters_out: int,
            padding:int
    ):
        super().__init__()


        self.conv_layer_1 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)
        self.conv_layer_2 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)

        self.norm_layer_1 = nn.InstanceNorm3d(n_filters_out)
        self.norm_layer_2 = nn.InstanceNorm3d(n_filters_out)

        self.activation_1 = nn.SiLU(inplace=False)
        self.activation_2 = nn.SiLU(inplace=False)


    def forward(self, x):
        identity = x

        out = self.conv_layer_1(x)
        out = self.norm_layer_1(out)
        out = self.activation_1(out)

        out = self.conv_layer_2(out)
        out = self.norm_layer_2(out)
        out = self.activation_2(out)

        # return out + identity
        return out

class SegResNetBlock(nn.Module):
    def __init__(
            self, 
            n_filters_in: int, 
            n_filters_out: int,
            padding: int
    ):
        super().__init__()

        self.conv_layer_1 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)
        self.conv_layer_2 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)

        self.norm_layer_1 = nn.BatchNorm3d(n_filters_out)
        self.norm_layer_2 = nn.BatchNorm3d(n_filters_out)

        self.activation_1 = nn.ReLU(inplace=False)
        self.activation_2 = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = x

        out = self.norm_layer_1(x)
        out = self.activation_1(out)
        out = self.conv_layer_1(out)

        out = self.norm_layer_2(out)
        out = self.activation_2(out)
        out = self.conv_layer_2(out)

        return out + identity


class UnetBlock(nn.Module):
    def __init__(
            self, 
            n_filters_in: int, 
            n_filters_out: int,
            padding: int
    ):
        super().__init__()

        self.conv_layer_1 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)
        self.conv_layer_2 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)

        self.norm_layer_1 = nn.BatchNorm3d(n_filters_out)
        self.norm_layer_2 = nn.BatchNorm3d(n_filters_out)

        self.activation_1 = nn.ReLU(inplace=False)
        self.activation_2 = nn.ReLU(inplace=False)
    
    def forward(self, x):

        out = self.conv_layer_1(x)
        out = self.norm_layer_1(out)
        out = self.activation_1(out)

        out = self.conv_layer_2(out)
        out = self.norm_layer_2(out)
        out = self.activation_2(out)

        return out 



class MultiScaleConvBlock(nn.Module):
    """
    Ungated two-scale 3D convolution block.

    Branch 1 uses one 3x3x3 convolution.
    Branch 2 uses two stacked 3x3x3 convolutions, giving a 5x5x5
    effective receptive field.

    Branch outputs are concatenated and fused through a 1x1x1
    convolution. A residual path preserves the input representation.
    """

    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.local_branch = nn.Sequential(
            nn.Conv3d(
                n_filters_in,
                n_filters_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                n_filters_out,
                affine=True,
            ),
            nn.SiLU(inplace=False),
        )

        self.context_branch = nn.Sequential(
            nn.Conv3d(
                n_filters_in,
                n_filters_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                n_filters_out,
                affine=True,
            ),
            nn.SiLU(inplace=False),

            nn.Conv3d(
                n_filters_out,
                n_filters_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                n_filters_out,
                affine=True,
            ),
            nn.SiLU(inplace=False),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(
                2 * n_filters_out,
                n_filters_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm3d(
                n_filters_out,
                affine=True,
            ),
        )

        if n_filters_in == n_filters_out:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv3d(
                n_filters_in,
                n_filters_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

        self.activation = nn.SiLU(inplace=False)
        self.dropout = nn.Dropout3d(
            p=dropout,
            inplace=False,
        )

    def forward(self, x):
        local_features = self.local_branch(x)
        context_features = self.context_branch(x)

        multi_scale_features = torch.cat(
            [local_features, context_features],
            dim=1,
        )

        fused_features = self.fusion(
            multi_scale_features
        )

        output = fused_features + self.residual(x)
        output = self.activation(output)

        return self.dropout(output)
    


class DownsamplingBlock(nn.Module):

    def __init__(
            self, 
            n_filters_in:int, 
            n_filters_out:int, 
            stride:int, 
            padding:int
    ):
        super().__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
        ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.SiLU(inplace=False))

        self.conv = nn.Sequential(*ops)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class DownSamplingMedNeXT(nn.Module):

    def __init__(
            self,
            n_filters_in: int,
            n_filters_out: int
    ):
        super().__init__()

        self.inc = n_filters_in
        self.outc = n_filters_out

        self.res_conv = nn.Conv3d(
            in_channels = n_filters_in, 
            out_channels = n_filters_out, 
            kernel_size=1, 
            stride=2, 
            padding=0,
            groups=n_filters_in
        )

        self.conv1 = nn.Conv3d(
            in_channels=n_filters_in,
            out_channels = n_filters_in,
            kernel_size=3,
            padding=1,
            stride=2
        )

        self.norm1 = nn.InstanceNorm3d(n_filters_in)
        self.act = nn.SiLU(inplace=False)

        self.expansion = nn.Conv3d(
            in_channels = n_filters_in,
            out_channels = n_filters_in * 4,
            kernel_size=1,
            padding=0,
            stride=1
        )

        self.contraction = nn.Conv3d(
            in_channels = n_filters_in * 4,
            out_channels = n_filters_out,
            kernel_size=1,
            stride=1,
            padding=0
        )
    def forward(self, x):

        identity = x

        x1 = self.conv1(x)
        x1 = self.act(self.expansion(self.norm1(x1)))
        x1 = self.contraction(x1)

        return x1 + self.res_conv(identity)


        
