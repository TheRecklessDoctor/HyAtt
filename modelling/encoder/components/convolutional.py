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

        self.norm_layer_1 = nn.BatchNorm3d(n_filters_out)
        self.norm_layer_2 = nn.BatchNorm3d(n_filters_out)

        self.activation_1 = nn.ReLU(inplace=False)
        self.activation_2 = nn.ReLU(inplace=False)


    def forward(self, x):
        identity = x

        out = self.conv_layer_1(x)
        out = self.norm_layer_1(out)
        out = self.activation_1(out)

        out = self.conv_layer_2(out)
        out = self.norm_layer_2(out)
        out = self.activation_2(out)

        return out + identity

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



class MultiscaleConvBlock(nn.Module):

    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        padding:int,
    ):
        super().__init__()

        self.conv_layer_1 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)
        self.conv_layer_2 = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=padding, stride=1)
        self.conv_layer_3 = nn.Conv3d(n_filters_out, n_filters_out, 3, padding=padding)
        # self.conv_layer_4 = nn.Conv3d(n_filters_out, n_filters_out, 3, padding=padding)

        self.norm_layer_1 = nn.InstanceNorm3d(n_filters_out)
        self.norm_layer_2 = nn.InstanceNorm3d(2*n_filters_out)

        self.activation_1 = nn.SiLU(inplace=False)
        self.activation_2 = nn.SiLU(inplace=False)

        self.dropout = nn.Dropout3d(inplace=False, p=0.1)
    
    def forward(self, x):

        branch1 = self.conv_layer_1(x)
        branch2 = self.conv_layer_2(x)

        branch1 = self.norm_layer_1(branch1)
        branch1 = self.activation_1(branch1)

        #this is cause the effective receptive field of two 3x3 conv layers is equivalent to a 5x5 one. 
        branch2 = self.conv_layer_3(branch2)
        # branch2 = self.conv_layer_4(branch2)

        concat = torch.cat((branch1, branch2), 1)
        concat = self.norm_layer_2(concat)
        result = self.activation_2(concat)
        return self.dropout(result)
    

    


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




        
