import torch
from torch import nn
from omegaconf import DictConfig

from .components.convolutional import DownsamplingBlock
from .components.mogablock import MogaBlock
from .components.convolutional import ResNetBlock, MultiScaleConvBlock



class SinglePathEncoderStage(nn.Module):

    def __init__(
            self, 
            n_filters_in: int,
            n_filters_out: int,
            print_sizes: bool
    ):
        super().__init__()

        # self.conv_a = MogaBlock(
        #     embed_dims=n_filters_in
        # )

        # self.conv_a = ResNetBlock(
        #     n_filters_in=n_filters_in,
        #     n_filters_out=n_filters_in,
        #     padding=1
        # )
        self.conv_a = MultiScaleConvBlock(
            n_filters_in,
            n_filters_in
        )

        self.down_a = DownsamplingBlock(
            n_filters_in=n_filters_in, 
            n_filters_out=n_filters_out, 
            stride=2, 
            padding=0
        )
        
        self._print_sizes = print_sizes
    
    def _print_shapes(
            self, 
            act_a: torch.Tensor, 
            stage: str
    ) -> None:
        
        if self._print_sizes:
            print(f"{stage} A: {act_a.shape}")

    def forward(
            self, 
            a: torch.Tensor
    ) -> list[torch.Tensor]:
        
        # print("stage output shape:", a.shape)
        
        feat_a = self.conv_a(a)
        feat_a_down = self.down_a(feat_a)

        self._print_shapes(feat_a_down, "after downsampling")
        
        return feat_a_down




class DualEncoder(nn.Module):

    def __init__(
            self,
            weight_initialiser: DictConfig,
            num_blocks: int,
            n_filters_in_initial: int,
            n_filters_out_initial: int,
            print_sizes: bool
    ):
        super().__init__()

        self.weight_initialiser = weight_initialiser
        n_filters_in = n_filters_in_initial
        n_filters_out = n_filters_out_initial
        self.image_size = (80, 80, 80)
        # This was required to use patch embedding from Monai

        # Create encoder stages
        self.stages = nn.ModuleList()
        for i in range(num_blocks):

            stage = SinglePathEncoderStage(
                n_filters_in,
                n_filters_out,
                print_sizes[i]
            )         

            self.stages.append(stage)

            n_filters_in = n_filters_out
            n_filters_out *= 2

        self._print_sizes = print_sizes


        self._initialise_weights()
    
    def _initialise_weights(self):

        if self.weight_initialiser == "kaiming":
            weight_initialiser = torch.nn.init.kaiming_normal_
        else:
            weight_initialiser = torch.nn.init.xavier_normal_
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                weight_initialiser(m.weight)
    
    def _print_shapes(
            self, 
            act_a: torch.Tensor, 
            stage: str,
            print_at_stage
    ) -> None:
        
        if print_at_stage:
            print(f"{stage} {act_a.shape}\n")

    
    def forward(
            self, 
            x_a: torch.Tensor
    ) -> torch.Tensor:

        features_a = []
        
        for i, stage in enumerate(self.stages):

            x_a = stage(x_a)

            features_a.append(x_a)

            self._print_shapes(x_a, f"End stage {i}", self._print_sizes[i])

        return features_a




