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
                print(f'{kwargs["section"]}: {kwargs["a"].shape}')
    
    def forward(
            self, 
            scan: List[torch.Tensor]
    ) -> list[torch.Tensor]:
        

        enc_a = self.encoder(
            scan,
        )

        self.print_size(printS=self.printS, section="encoder", a=enc_a[-1])


        latent = self.bottleneck(
            enc_a
        )

        self.print_size(printS=self.printS, section="bottleneck", a=latent)

        prediction = self.decoder(
            latent,
            enc_a
        )

        self.print_size(printS=self.printS, section="decoder", a=prediction)

        return [prediction]
    


if __name__ == "__main__":
    pass