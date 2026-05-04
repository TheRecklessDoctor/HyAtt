from copy import deepcopy
from typing import List

import torch
from torch import nn
from omegaconf import DictConfig
from hydra.utils import instantiate




def conv_layer(
        in_dim:int, 
        out_dim:int, 
        kernel_size:int, 
        padding:int, 
        stride:int,
        normalisation: nn.Module,
        act:nn.Module):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        normalisation(out_dim), 
        act)

def dwconv_layer(
        in_dim:int, 
        out_dim:int, 
        kernel_size:int, 
        padding:int, 
        stride:int,
        normalisation: nn.Module,
        act:nn.Module):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias=False, groups=in_dim),
        normalisation(out_dim), 
        act)


def deconv_layer(
        in_dim:int, 
        out_dim:int, 
        kernel_size:int, 
        padding:int,
        stride:int,
        normalisation: nn.Module,
        act:nn.Module):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        normalisation, 
        act)

class PrototypeA(nn.Module):

    """
    Class containing code for the bottleneck. A separate class is used alongside
    the bottleneck in order to allow for rapid prototyping while ensuring code remains
    relatively clean.
    """

    def __init__(
            self,
            conv: callable,
            depthwise: callable,
            weight_initialiser:DictConfig
    ):
        super().__init__()
        # self.test_conv_a = conv(128, 128, 1, 0, 1)
        # self.test_conv_a = conv(64, 64, 1, 0, 1)
        # self.test_conv_b = conv(128, 64, 1, 0, 1)
        self.test_conv_c = conv(128, 64, 1, 0, 1)

        self.dwconv3 = depthwise(128, 128, 3, 1, 1)
        self.dwconv5 = depthwise(128, 128, 5, 2, 1)
        # self.dwconv7 = depthwise(128, 128, 7, 3, 1)

        # self.dwconv3b = depthwise(128, 128, 3, 1, 1)
        # self.dwconv5b = depthwise(128, 128, 5, 2, 1)
        # self.dwconv7b = depthwise(128, 128, 7, 3, 1)
        self.weight_initialiser = weight_initialiser

        self.norm1 = nn.InstanceNorm3d(128)
        self.norm2 = nn.InstanceNorm3d(128)
        self.norm3 = nn.InstanceNorm3d(128)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()

        # self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        # self.beta  = nn.Parameter(torch.ones(1) * 0.5)


        self._initialise_weights()

    def _initialise_weights(self):
        if self.weight_initialiser == "kaiming":
            weight_initialiser = torch.nn.init.kaiming_normal_
        else:
            weight_initialiser = torch.nn.init.xavier_normal_
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                weight_initialiser(m.weight)
    
    def forward(self, out_a, out_b):

        v5_a = out_a
        v5_b = out_b

        concat = torch.cat((v5_a, v5_b), 1)

        dw3 = self.act1(self.norm1(self.dwconv3(concat)))
        dw5 = self.act2(self.norm2(self.dwconv5(concat)))
        concat = self.norm3(concat)


        # print(f"concat mag: {concat.abs().mean():.4f}")
        # print(f"dw3 mag:    {dw3.abs().mean():.4f}")
        # print(f"dw5 mag:    {dw5.abs().mean():.4f}")


        # This section might be inefficient.
        multi = dw3 + dw5 + concat
        multi = self.test_conv_c(multi)

        # multi = self.test_conv_a(multi)

        fq = multi

        return fq




# Performs fusion of the dual-encoders outputs
class Bottleneck(nn.Module):

    """
    Wrapper class for the bottleneck.
    """
    def __init__(self,
                #  bridger: nn.Module,
                 in_channels:int,
                 out_channels:List[int],
                 num_enc_blocks: int,
                 stride:List[int], # [1, 1, 1] for vit
                 conv: callable,
                 depthwise: callable,
                 deconv: callable,
                 print_sizes: bool,
                 weight_initialiser: DictConfig
                ):
        super().__init__()

        self._enc_blocks = num_enc_blocks
        self._print_sizes = print_sizes

        input_channels = in_channels
        for _ in range(self._enc_blocks-1):
            input_channels *= 2


        # self.fusion5 = deepcopy(bridger)

        #kernel sizes used, 3, 4
        # self.f5_proj = conv(input_channels, out_channels[0], 2, 0, stride[2])
        # self.f5_deconv = deconv(out_channels[0], input_channels, 2, 0, 1)

        # self.test_conv_a = conv(128, 128, 1, 0, 1)
        # # self.test_conv_a = conv(64, 64, 1, 0, 1)
        # self.test_conv_b = conv(128, 64, 1, 0, 1)
        # self.test_conv_c = conv(128, 64, 1, 0, 1)

        # self.dwconv3 = depthwise(128, 128, 3, 1, 1)
        # self.dwconv5 = depthwise(128, 128, 5, 2, 1)
        # # self.dwconv7 = depthwise(128, 128, 7, 3, 1)

        # self.dwconv3b = depthwise(128, 128, 3, 1, 1)
        # self.dwconv5b = depthwise(128, 128, 5, 2, 1)
        # self.dwconv7b = depthwise(128, 128, 7, 3, 1)

        # self.normalisation1 = normalisation
        # self.normalisation2 = normalisation
        # self.activation = activation

        self.proA = PrototypeA(
            conv=conv, 
            depthwise=depthwise, 
            weight_initialiser = weight_initialiser
        )

        # self._initialise_weights()
    
    # def _initialise_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             self._weight_initialiser(m.weight)
    
    def _print_shapes(self, 
                      act_a: torch.Tensor, 
                      act_b: torch.Tensor, 
                      stage: str
                    ) -> None:
        
        if self._print_sizes:
            print(f"{stage} A: {act_a.shape}, B: {act_b.shape}")
    

    def forward(self, enc_a, enc_b):

        v5_a = enc_a[self._enc_blocks-1]
        v5_b = enc_b[self._enc_blocks-1]

        fq = self.proA(v5_a, v5_b)

        self._print_shapes(v5_a, v5_b, "bottleneck start")

        self._print_shapes(fq, fq, "bottleneck after zoom out")
        return fq
    





