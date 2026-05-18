from typing import List

from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch


class Model(nn.Module):
    """
    Wrapper class for the model
    """
    def __init__( 
            self,
            model_cfg: DictConfig,
            weight_initialiser: callable
    ) -> None:

        super().__init__()
        self.encoder = instantiate(model_cfg.encoder, weight_initialiser = weight_initialiser)
        self.bottleneck = instantiate(model_cfg.bottleneck, weight_initialiser = weight_initialiser)
        self.decoder = instantiate(model_cfg.decoder, weight_initialiser = weight_initialiser)
        self.printS = model_cfg.printS
    
    def print_size(self, **kwargs):
        if kwargs["printS"]:
            if "b" in kwargs:
                print(f'{kwargs["section"]}: {kwargs["a"].shape}, {kwargs["b"].shape}')
            else:
                print(f'{kwargs["section"]}: {kwargs["a"].shape}')
    
    def forward(
            self, 
            scan_a: List[torch.Tensor], 
            scan_b: List[torch.Tensor]
    ) -> list[torch.Tensor]:
        

        enc_a, enc_b  = self.encoder(
            scan_a,
            scan_b
        )

        self.print_size(printS=self.printS, section="encoder", a=enc_a[-1], b=enc_b[-1])


        latent = self.bottleneck(
            enc_a, 
            enc_b
        )

        self.print_size(printS=self.printS, section="bottleneck", a=latent)

        prediction, head2, head3 = self.decoder(
            latent,
            [enc_a, enc_b]
        )

        self.print_size(printS=self.printS, section="decoder", a=prediction)

        return [prediction, head2, head3]
    


if __name__ == "__main__":
    pass