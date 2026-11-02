from monai.networks.nets import (
    UNETR, 
    UNet, 
    SegResNet, 
    SwinUNETR, 
    ViTAutoEnc, 
    AttentionUnet, 
    BasicUNetPlusPlus,
    DynUNet
)
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import hydra
from omegaconf import DictConfig

from src.modelling.model import Model



def get_monai_model(
            model_type:str = "unetr", 
            device: str = "cuda",
    ):
        model = None
        if model_type == "unetr":
            model = UNETR(
                in_channels=2,
                out_channels = 2,
                img_size = (80, 80, 80),
                feature_size = 16,
                hidden_size=32,
                norm_name = "instance",
                dropout_rate=0.1,
                num_heads = 2,
            )
        elif model_type == "unet":
            model = UNet(
                in_channels=2, 
                out_channels=2, 
                spatial_dims=3,
                channels=(16, 32, 64),
                strides=(2, 2, 2),
                dropout = 0.1,
                norm='instance',
                act='RELU'
            )
        elif model_type == 'segres':
            #test with blocks_down=(1,1,1) and blocks_up=(1,1,1,1)
            model = SegResNet(
                in_channels=2, 
                out_channels=2, 
                spatial_dims=3,
                dropout_prob=0.1,
                act=('ReLU', {'inplace':True},),
                norm='instance',
                blocks_down=(1,1,1,1)
            )
        elif model_type == "swin":
            model = SwinUNETR(
                in_channels=2,
                out_channels=2,
                feature_size=12,
                depths=(1,1,1,1),
                num_heads=(2,2,2,2)
            )
        elif model_type == "vit":
            model = ViTAutoEnc(
                in_channels=2,
                out_channels=2,
                hidden_size=32,
                dropout_rate=0.1,
                num_heads = 2,
                img_size=(80,80,80),
                patch_size=(16,16,16),
                num_layers=3,
                mlp_dim=512
            )
        elif model_type == "attnunet":
            model = AttentionUnet(
                spatial_dims=3,
                in_channels=2,
                out_channels=2,
                channels=[16, 32, 64],
                strides=[2,2,2],

            )
        elif model_type == "unetpp":
            model = BasicUNetPlusPlus(
                spatial_dims=3,
                features=(16,16,16,16,32,32),
                in_channels=2,
                out_channels=2
            )
        elif model_type == "dynunet":
            strides = [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
            ]

            kernel_size = [
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ]
            model = DynUNet(
                in_channels=2, 
                out_channels=2, 
                spatial_dims=3,
                kernel_size=kernel_size,
                filters=(16, 32, 64),
                strides=strides,
                upsample_kernel_size=strides[1:],
                dropout = 0.1,
                norm_name='instance'
            )

        return model

def measure_flops(model, input, name):

    model.eval()

    total_parameters = sum(p.numel() for p in model.parameters())

    analysis = FlopCountAnalysis(model, input)
    total_flops = analysis.total()

    print(f"\n{name}")
    print(f"Input shape:       {input.shape}")
    print(f"Total FLOPs:       {total_flops:,.0f}")
    print(f"GFLOPs: {total_flops / 1e9:.4f}")
    print(f"Total parameters:{total_parameters}")



@hydra.main(config_path="../config", config_name="main", version_base="1.3")
def main(
        cfg: DictConfig
):
    model_config = cfg.build
    weight_initialiser = cfg.learning_params.weight_initialiser.type

    model = Model(
                model_config,
                weight_initialiser=weight_initialiser
            )

    unet = get_monai_model(model_type = "unet")
    unetr = get_monai_model(model_type = "unetr")
    segres = get_monai_model(model_type = "segres")
    vit = get_monai_model(model_type = "vit")
    attnunet = get_monai_model(model_type = "attnunet")
    unetpp = get_monai_model(model_type = "unetpp")
    swin = get_monai_model(model_type = "swin")
    dyn = get_monai_model(model_type = "dynunet")

    input = torch.randn(1,2,80,80,80)
    measure_flops(model, input, "mine")
    measure_flops(unet, input, "unet")
    measure_flops(unetr, input, "unetr")
    measure_flops(segres, input, "segres")
    measure_flops(vit, input, "vit")
    measure_flops(attnunet, input, "attnunet")
    measure_flops(unetpp, input, "unetpp")
    measure_flops(dyn, input, "dynunet")

    input = torch.randn(1,2,96,96,96)
    measure_flops(swin, input, "swin")


if __name__ == "__main__":
    main()