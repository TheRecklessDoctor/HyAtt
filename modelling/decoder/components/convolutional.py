import torch
from torch import nn
from hydra.utils import instantiate
from copy import deepcopy

class ResNetBlock(nn.Module):
    def __init__(self,
                 n_filters_in: int,
                 n_filters_out: int,
                 padding:int,
                 normalisation: nn.Module,
                 act: nn.Module
                 ):
        super().__init__()


        self.conv_layer_1 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding)
        self.norm_layer_1 = nn.BatchNorm3d(n_filters_out)
        self.activation_1 = nn.ReLU(inplace=False)

        self.conv_layer_2 = nn.Conv3d(n_filters_out, n_filters_out, 3, padding=padding)
        self.norm_layer_2 = nn.BatchNorm3d(n_filters_out)
        self.activation_2 = nn.ReLU(inplace=False)

        # self.filter_mod = nn.Conv3d(n_filters_in, n_filters_out, 1)

        # self.dropout = nn.Dropout(0.2, inplace=False)

    def forward(self, x):

        out = self.conv_layer_1(x)
        out = self.norm_layer_1(out)
        out = self.activation_1(out)

        out = self.conv_layer_2(out)
        out = self.norm_layer_2(out)
        out = self.activation_2(out)
        out = out + x

        return out

class ConvolutionalBlock(nn.Module):

    def __init__(self,
                  n_stages:int, 
                  n_filters_in:int, 
                  n_filters_out:int, 
                  padding: int, 
                  normalisation: nn.Module, 
                  act:nn.Module,
                  kernel_size: int
                  ):
        super().__init__()

        ops = []

        for i in range(n_stages):
            if i == 0:
                input_channels = n_filters_in
            else:
                input_channels = n_filters_out
            
            ops.append(nn.Conv3d(input_channels, n_filters_out, kernel_size, padding=padding))
            ops.append(nn.InstanceNorm3d(n_filters_out))
            ops.append(nn.SiLU(inplace=False))
        
        self.conv = nn.Sequential(*ops)
    
    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingBlock(nn.Module):

    def __init__(self, 
                 n_filters_in: int, 
                 n_filters_out:int, 
                 stride:int, 
                 output_padding: int,
                 padding:int,
                 normalisation: nn.Module, 
                 act:nn.Module, 
                 mode_upsampling:int):
        super().__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride = stride, output_padding=output_padding))
    
        elif mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=3, padding=1 , output_padding=output_padding))
        
        ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.SiLU(inplace=False))

        self.conv = nn.Sequential(*ops)
        # self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, x):

            x = self.conv(x)
            # x = self.dropout(x)
            return x


class ConvUp(nn.Module):
    def __init__(self, conv_cfg, up_cfg):
        super().__init__()

        self.conv = instantiate(conv_cfg)
        self.up = instantiate(up_cfg)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        
        return x