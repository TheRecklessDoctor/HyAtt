from typing import List, Tuple
# import copy

import torch
from torch import nn
from omegaconf import DictConfig
# from monai.networks.blocks import PatchEmbeddingBlock
# from monai.networks.nets.vit import ViT
# from monai.networks.blocks.transformerblock import TransformerBlock
# import torch.nn.functional as F

from .components.convolutional import MultiscaleConvBlock, DownsamplingBlock
from .components.bridger import Bridger, SinglePathBridger

class SpatialToSequenceProjector(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            embed_dim: int
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W, D] -> [B, H*W*D, embed_dim]
        """
        B, C, H, W, D = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)  
        return x


class SequenceToSpatialProjector(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            out_channels: int
    ):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)
    
    def forward(
            self, 
            x: torch.Tensor, 
            spatial_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        x: [B, H*W*D, embed_dim] -> [B, C, H, W, D]
        """
        B, H, W, D = spatial_shape
        x = x.permute(1, 2, 0).view(B, -1, H, W, D)  
        return x


class DualPathEncoderStage(nn.Module):

    def __init__(
            self, 
            n_filters_in: int,
            n_filters_out: int,
            bridger_embedding: int,
            bridger_feedforward_embedding: int,
            print_sizes: bool,
            use_bridger: bool = True,
            bridger: Bridger = None,
    ):
        super().__init__()

        self.conv_a = MultiscaleConvBlock(
            n_filters_in=n_filters_in, 
            n_filters_out=n_filters_in, 
            padding=1
        )
        self.conv_b = MultiscaleConvBlock(
            n_filters_in=n_filters_in, 
            n_filters_out=n_filters_in, 
            padding=1
        )

        self.down_a = DownsamplingBlock(
            n_filters_in=2*n_filters_in, 
            n_filters_out=n_filters_out, 
            stride=2, 
            padding=0
        )
        self.down_b = DownsamplingBlock(
            n_filters_in=2*n_filters_in, 
            n_filters_out=n_filters_out, 
            stride=2, 
            padding=0
        )
        

        self.use_bridger = use_bridger 
        if self.use_bridger:
            self.zoom_in_a = nn.Conv3d(
                n_filters_out, 
                bridger_embedding, 
                kernel_size=2, 
                stride=2, 
                bias=True
            )

            self.zoom_in_b = nn.Conv3d(
                n_filters_out, 
                bridger_embedding, 
                kernel_size=2, 
                stride=2, 
                bias=True
            )

            self.zoom_out_a = nn.ConvTranspose3d(
                bridger_embedding, 
                n_filters_out, 
                kernel_size=2, 
                stride=2, 
                bias=True
            )

            self.zoom_out_b = nn.ConvTranspose3d(
                bridger_embedding, 
                n_filters_out, 
                kernel_size=2, 
                stride=2, 
                bias=True
            )

            out_channels_a = n_filters_out
            out_channels_b = n_filters_out
            embed_dim = bridger_embedding

            self.bridger = Bridger(
                embed_dim = embed_dim, 
                feedforward_dim=bridger_feedforward_embedding,
                n_heads = 2,
                dropout = 0.1
            )

            self.to_seq_a = SpatialToSequenceProjector(out_channels_a, embed_dim)
            self.to_seq_b = SpatialToSequenceProjector(out_channels_b, embed_dim)
            self.to_spatial_a = SequenceToSpatialProjector(embed_dim, out_channels_a)
            self.to_spatial_b = SequenceToSpatialProjector(embed_dim, out_channels_b)

            self.dwconv = nn.Conv3d(
                        in_channels = 2*n_filters_out,
                        out_channels=2*n_filters_out,
                        groups=2*n_filters_out,
                        stride=1,
                        padding=1,
                        kernel_size=3
                )

            self.pwconv = nn.Conv3d(
                2*n_filters_out, 
                n_filters_out, 
                1
            )
        
        self.dropout = nn.Dropout(0.1, inplace=False)
        self._print_sizes = print_sizes

    
    def _print_shapes(
            self, 
            act_a: torch.Tensor, 
            act_b: torch.Tensor, 
            stage: str
    ) -> None:
        
        if self._print_sizes:
            print(f"{stage} A: {act_a.shape}, B: {act_b.shape}")
    

    def forward(
            self, 
            a: torch.Tensor,
            b: torch.Tensor
    ) -> list[torch.Tensor]:
        
        feat_a = self.conv_a(a)
        feat_b = self.conv_b(b)

        feat_a_down = self.down_a(feat_a)
        feat_b_down = self.down_b(feat_b)


        self._print_shapes(feat_a_down, feat_b_down, "after downsampling")
        

        if not self.use_bridger:
            return [feat_a_down, feat_b_down]
        

        feat_a_zoom_in = self.zoom_in_a(feat_a_down)
        feat_b_zoom_in = self.zoom_in_b(feat_b_down)

        self._print_shapes(feat_a_zoom_in, feat_b_zoom_in, "after zoom in")
        B, _, H, W, D = feat_a_zoom_in.shape

        seq_a= self.to_seq_a(feat_a_zoom_in)
        seq_b= self.to_seq_b(feat_b_zoom_in)

        enh_seq_a, enh_seq_b = self.bridger(seq_a, seq_b)

        enh_a = self.to_spatial_a(enh_seq_a, (B, H, W, D))
        enh_b = self.to_spatial_b(enh_seq_b, (B, H, W, D))
        
        enh_a_zoom_out = self.zoom_out_a(enh_a)
        enh_b_zoom_out = self.zoom_out_b(enh_b)

        self._print_shapes(enh_a_zoom_out, enh_b_zoom_out, "after zoom out")

        concatenated = torch.cat([feat_a_down, feat_b_down], dim=1)
        concatenated = self.dwconv(concatenated)
        concatenated = self.pwconv(concatenated)

        # changed this from * to +
        feat_a = feat_a_down + (enh_a_zoom_out + concatenated)
        feat_b = feat_b_down + (enh_b_zoom_out + concatenated)

        feat_a = self.dropout(feat_a)
        feat_b = self.dropout(feat_b)

        return [feat_a, feat_b]

class DualEncoderStageWithSinglePathBridger(nn.Module):

    def __init__(
      self,
      n_filters_in:int,
      n_filters_out:int,
      bridger_embedding:int,
      bridger_feedforward_embedding:int,
      print_sizes:bool,
      use_bridger:bool = True      
    ):
        super().__init__()

        self.conv_a = MultiscaleConvBlock(
            n_filters_in = n_filters_in,
            n_filters_out = n_filters_in,
            padding = 1
        )

        self.conv_b = MultiscaleConvBlock(
            n_filters_in = n_filters_in,
            n_filters_out = n_filters_in,
            padding = 1
        )

        self.down_a = DownsamplingBlock(
            n_filters_in = 2*n_filters_in,
            n_filters_out = n_filters_out,
            stride = 2,
            padding = 0
        )

        self.down_b = DownsamplingBlock(
            n_filters_in = 2*n_filters_in,
            n_filters_out = n_filters_out,
            stride = 2,
            padding = 0
        )

        self.use_bridger = use_bridger
        if self.use_bridger:

            self.zoom_in = nn.Conv3d(
                in_channels = n_filters_out*2,
                out_channels = bridger_embedding,
                kernel_size = 2,
                stride = 2,
                bias = True
            )

            self.zoom_out = nn.ConvTranspose3d(
                in_channels = bridger_embedding,
                out_channels = 2*n_filters_out,
                kernel_size = 2,
                stride = 2,
                bias = True
            )

            out_channels = n_filters_out
            embed_dim = bridger_embedding

            self.bridger = SinglePathBridger(
                embed_dim = embed_dim,
                feedforward_dim = bridger_feedforward_embedding,
                n_heads = 2,
                dropout = 0.1
            )

            self.to_sequence = SpatialToSequenceProjector(
                in_channels = out_channels,
                embed_dim = embed_dim
            )
            self.to_spatial = SequenceToSpatialProjector(
                embed_dim = embed_dim,
                out_channels = out_channels
            )

            self.dwconv = nn.Conv3d(
                in_channels = 2*n_filters_out,
                out_channels=2*n_filters_out,
                groups=2*n_filters_out,
                stride=1,
                padding=1,
                kernel_size=3
                )

            self.pwconv = nn.Conv3d(
                2*n_filters_out, 
                n_filters_out, 
                1
            )
        else:
            self.merge = nn.Conv3d(
                in_channels = n_filters_out*2,
                out_channels = n_filters_out,
                kernel_size=1
            )

        self.dropout = nn.Dropout(0.1, inplace=False)
        self._print_sizes = print_sizes

    
    def print_shapes(
            self,
            path_a: torch.Tensor,
            path_b: torch.Tensor,
            stage: str
    ):
        if self._print_sizes:
            print(f"{stage} A:{path_a.shape}, B:{path_b.shape}")
    

    def forward(
            self,
            a: torch.Tensor,
            b: torch.Tensor
    ) -> list[torch.Tensor]:
    
        
        feat_a = self.conv_a(a)
        feat_b = self.conv_b(b)

        feat_a_down = self.down_a(feat_a)
        feat_b_down = self.down_b(feat_b)

        self.print_shapes(
            path_a = feat_a_down, 
            path_b = feat_b_down,
            stage = "after downsampling"
        )
        
        if not self.use_bridger:
            concat = torch.concat([feat_a_down, feat_b_down], dim=1)
            concat = self.merge(concat)

            feat_a_down = feat_a_down + concat
            feat_b_down = feat_b_down + concat

            return [feat_a_down, feat_b_down]
        
        concat = torch.concat([feat_a_down, feat_b_down], dim=1)
        zoom_in = self.zoom_in(concat)
        self.print_shapes(zoom_in, zoom_in, "after zoom in")
        B, _, H, W, D = zoom_in.shape

        seq = self.to_sequence(zoom_in)

        enh_seq = self.bridger(seq, seq)

        spatial = self.to_spatial(enh_seq, (B, H, W, D))

        zoom_out = self.zoom_out(spatial)
        self.print_shapes(zoom_out, zoom_out, "after zoom out")

        feat_a_chunk, feat_b_chunk = torch.chunk(zoom_out, 2, dim=1)

        helper2 = self.dwconv(concat)
        helper2 = self.pwconv(helper2)

        feat_a = feat_a_down + feat_a_chunk + helper2
        feat_b = feat_b_down + feat_b_chunk + helper2

        feat_a = self.dropout(feat_a)
        feat_b = self.dropout(feat_b)

        return [feat_a, feat_b]



class SinglePathEncoderStage(nn.Module):

    def __init__(
            self, 
            n_filters_in: int,
            n_filters_out: int,
            print_sizes: bool,
    ):
        super().__init__()

        self.conv_a = MultiscaleConvBlock(
            n_filters_in=n_filters_in, 
            n_filters_out=n_filters_in, 
            padding=1
        )

        self.down_a = DownsamplingBlock(
            n_filters_in=2*n_filters_in, 
            n_filters_out=n_filters_out, 
            stride=2, 
            padding=0
        )
        self._print_sizes = print_sizes
    
    def _print_shapes(
            self, 
            act_a: torch.Tensor, 
            act_b: torch.Tensor, 
            stage: str
    ) -> None:
        
        if self._print_sizes:
            print(f"{stage} A: {act_a.shape}, B: {act_b.shape}")
    
    def _to_spatial(self, x, img_size: tuple[int, int, int], patch_size:tuple[int, int, int]):

        B, N, Hid = x.shape
        feat_size = tuple(im // ps for im, ps in zip(img_size, patch_size))
        D_patch, H_patch, W_patch = feat_size

        x = x.view(B, D_patch, H_patch, W_patch, Hid)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x
    

    def forward(
            self, 
            a: torch.Tensor
    ) -> list[torch.Tensor]:
        
        feat_a = self.conv_a(a)

        feat_a_down = self.down_a(feat_a)


        self._print_shapes(feat_a_down, feat_a_down, "after downsampling")
        
        return feat_a_down




class DualEncoder(nn.Module):

    def __init__(
            self,
            weight_initialiser: DictConfig,
            bridger_at_stages: List[int],
            num_blocks: int,
            n_filters_in_initial: int,
            n_filters_out_initial: int,
            initial_bridger_embedding: int,
            initial_bridger_feedforward_embedding: int,
            print_sizes: bool
    ):
        super().__init__()

        self.weight_initialiser = weight_initialiser
        self.bridger_at_stages = bridger_at_stages
        n_filters_in = n_filters_in_initial
        n_filters_out = n_filters_out_initial
        bridger_embedding = initial_bridger_embedding
        bridger_feedforward_embedding = initial_bridger_feedforward_embedding
        self.image_size = (80, 80, 80)
        # This was required to use patch embedding from Monai

        
        # Create encoder stages
        self.stages = nn.ModuleList()
        for i in range(num_blocks):
            use_bridger = True if i in self.bridger_at_stages else False
            # self.bridger = bridger if use_bridger else None
            # This was required to use the patch embedding block from Monai

            if use_bridger:

                # stage = DualPathEncoderStage(
                #     n_filters_in,
                #     n_filters_out,
                #     bridger_embedding,
                #     initial_bridger_feedforward_embedding,
                #     print_sizes[i],
                #     use_bridger
                # )

                stage = DualEncoderStageWithSinglePathBridger(
                    n_filters_in,
                    n_filters_out,
                    bridger_embedding,
                    bridger_feedforward_embedding,
                    print_sizes[i],
                    use_bridger
                )
            else:
                stage = DualPathEncoderStage(
                    n_filters_in,
                    n_filters_out,
                    bridger_embedding,
                    initial_bridger_feedforward_embedding,
                    print_sizes[i]
                )

                # stage = DualEncoderStageWithSinglePathBridger(
                #     n_filters_in,
                #     n_filters_out,
                #     bridger_embedding,
                #     bridger_feedforward_embedding,
                #     print_sizes[i],
                #     use_bridger
                # )

            # stage = SinglePathEncoderStage(
            #     n_filters_in,
            #     n_filters_out,
            #     print_sizes[i]
            # )

            self.stages.append(stage)

            n_filters_in = n_filters_out
            n_filters_out *= 2
            # bridger_embedding *= 2
            # bridger_feedforward_embedding *= 2

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
            act_b: torch.Tensor, 
            stage: str,
            print_at_stage
    ) -> None:
        
        if print_at_stage:
            print(f"{stage} A: {act_a.shape}, B: {act_b.shape}\n\n")

    
    def forward(
            self, 
            x_a: torch.Tensor, 
            x_b: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        features_a = []
        features_b = []
        
        for i, stage in enumerate(self.stages):

            x_a, x_b = stage(x_a, x_b)
            # x_a = stage(x_a)

            features_a.append(x_a)
            features_b.append(x_b)

  
            self._print_shapes(x_a, x_b, f"End stage {i}", self._print_sizes[i])
            # self._print_shapes(x_a, x_a, f"End stage {i}", self._print_sizes[i])

        return features_a, features_b
        # return features_a, features_a




