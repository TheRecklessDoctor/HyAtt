from typing import List

import torch
from torch import nn
from omegaconf import DictConfig
import torch.nn.functional as F




def conv_layer(
        in_dim:int, 
        out_dim:int, 
        kernel_size:int, 
        padding:int, 
        stride:int,
        optional:bool = True
    ):
    layers = []
    layers.append(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
    )
    if optional:
        layers.extend([
            nn.InstanceNorm3d(out_dim), 
            nn.SiLU(inplace=False)
        ])
    return nn.Sequential(*layers)

def dwconv_layer(
        in_dim:int, 
        out_dim:int, 
        kernel_size:int, 
        padding:int, 
        stride:int):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias=False, groups=in_dim),
        nn.InstanceNorm3d(out_dim), 
        # nn.SiLU(inplace=False)
    )


class PrototypeA(nn.Module):

    def __init__(
            self
    ):
        super().__init__()
        self.in_features = 64
        self.test_conv_c = conv_layer(self.in_features, self.in_features, 1, 0, 1)
        self.dwconv3 = dwconv_layer(self.in_features, self.in_features, 3, 1, 1)
        self.dwconv5 = dwconv_layer(self.in_features, self.in_features, 5, 2, 1)

        self.norm3 = nn.InstanceNorm3d(self.in_features)
    
    def forward(self, out_a):

        v5_a = out_a

        dw3 = self.dwconv3(v5_a)
        dw5 = self.dwconv5(v5_a)

        v5_a = self.norm3(v5_a)

        multi = dw3 + dw5 + v5_a
        multi = self.test_conv_c(multi) 

        fq = multi

        return fq
    


# Performs fusion of the dual-encoders outputs
class Bottleneck(nn.Module):
    def __init__(self,
                 num_enc_blocks: int,
                 print_sizes: bool,
                 weight_initialiser: DictConfig
                ):
        super().__init__()

        self._enc_blocks = num_enc_blocks
        self._print_sizes = print_sizes
        self.weight_initialiser = weight_initialiser

        self.proA = PrototypeA()
        # self.proB = PrototypeA()
        # self.proC = PrototypeA()

        # self.bottleneck_conv_ablation = conv_layer(64, 64, 1, 0, 1, False)

        self._initialise_weights()
    
    def _print_shapes(self, 
                      act_a: torch.Tensor,
                      stage: str
                    ) -> None:
        
        if self._print_sizes:
            print(f"{stage} A: {act_a.shape}")
    
    def _initialise_weights(self):
        if self.weight_initialiser == "kaiming":
            weight_initialiser = torch.nn.init.kaiming_normal_
        else:
            weight_initialiser = torch.nn.init.xavier_normal_
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                weight_initialiser(m.weight)

    def forward(self, enc_a):

        v5_a = enc_a[self._enc_blocks-1]

        self._print_shapes(v5_a, "bottleneck start")

        fq = self.proA(v5_a)
        # fq = self.proB(fq)
        # fq = self.proC(fq)
        # fq = v5_a
        # fq = self.bottleneck_conv_ablation(v5_a)

        self._print_shapes(fq,"bottleneck end")

        return fq
    





