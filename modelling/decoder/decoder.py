from .components.convolutional import ResNetBlock, UpsamplingBlock

import torch 
from torch import nn
from omegaconf import DictConfig

class Decoder(nn.Module):
      def __init__(self,
                 weight_initialiser:DictConfig,
                 num_blocks:int,
                 n_filters_in: int,
                 n_filters_out: int,
                 print_sizes: bool,
                 skip_connections: bool = False
                 ):
            super().__init__()
            self.blocks = nn.ModuleList()

            # so I don't have to change the configs for filter vals all
            # over the place. This way I can be a bit lazy with the configs 
            # during experiments.
            # self._filters_in = n_filters_in * num_blocks
            self._filters_in = n_filters_in 
            for _ in range(num_blocks - 1):
                  self._filters_in *= 2
            
            self._print_sizes = print_sizes

            self._filters_out = n_filters_out
            self._use_skip_connections = skip_connections
            self.use_deep_supervision = True
            self.weight_initialiser = weight_initialiser

            
            for i in range(num_blocks):
                if i == num_blocks - 1:
                      self.n_filters_upSamp = self._filters_in
                else:
                      self.n_filters_upSamp = self._filters_in // 2

                self.blocks.append(
                      nn.Sequential(
                        ResNetBlock(
                            n_filters_in = self._filters_in, 
                            n_filters_out = self._filters_in,
                            padding=1
                            ),
                        # MogaBlock(
                        #       embed_dims=self._filters_in
                        # ),
                        UpsamplingBlock(
                            n_filters_in = self._filters_in, 
                            n_filters_out = self.n_filters_upSamp,
                            padding=0,
                            output_padding=0,
                            stride=2,
                            mode_upsampling=0
                            )
                      )
                )
                  

                self._filters_in = self._filters_in // 2

            self.out_conv = nn.Conv3d(self.n_filters_upSamp, 2, 1, padding=0)

            self._initialise_weights()
            # self.dropout = nn.Dropout3d(0.5, inplace=False)
    
      def _initialise_weights(self):
            if self.weight_initialiser == "kaiming":
                  weight_initialiser = torch.nn.init.kaiming_normal_
            else:
                  weight_initialiser = torch.nn.init.xavier_normal_
            
            for m in self.modules():
                  if isinstance(m, nn.Conv3d):
                        weight_initialiser(m.weight)

            weight_initialiser(self.out_conv.weight)
      
      
    
      def _print_shapes(self, 
                        act: torch.Tensor, 
                        stage: str
                        ) -> None:
            
            if self._print_sizes:
                  print(f"{stage} Activation: {act.shape}")
    
      def forward(self, x, enc_outputs):
            num_blocks = len(enc_outputs)
            enc_a = enc_outputs
            for i, block in enumerate(self.blocks):
                  enc_a_out = enc_a[num_blocks - i - 1]
                  if self._use_skip_connections:
                        x = block[0](x) + enc_a_out
                        x = block[1](x)
                        
                  else:
                        x = block[0](x)
                        x = block[1](x)
                  self._print_shapes(x, f"decoder block {i}")                
                  
            x = self.out_conv(x)
            
            self._print_shapes(x, f"decoder end")
            return x
