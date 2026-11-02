import jsonlines

from src.modelling.model import Model
from .dilation_analysis import run_analysis
from src.data.dataloader import GenericDataSet

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch


def plot_validation_curve(file_path: str):
    epochs = []
    comp1 = []
    with jsonlines.open(file=file_path) as reader:
        for obj in reader:
            epoch = int(obj["epoch"])
            epochs.append(epoch)
            comp1.append(obj["dice_mean"])

    plt.figure(figsize=(7.5, 5))
    plt.plot(epochs, comp1)
    plt.title("Validation dice values over the epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.show()
    plt.close()


def plot_loss_curve(file_path: str):
    iters = []
    loss = []
    with jsonlines.open(file=file_path) as reader:
        for obj in reader:
            iters.append(obj["epoch"])
            loss.append(obj["Total loss"])

    window_size = 10
    smoothed_loss = pd.Series(loss).rolling(window=window_size, center=True).mean()
    plt.figure(figsize=(7.5, 5))
    plt.plot(iters, smoothed_loss)

    plt.title("Loss values over the epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.show()
    plt.close()


def print_model_tree(model: nn.Module):
    for name, module in model.named_modules():
        if "DW_conv" in name or "dwconv" in name.lower():
            print(f"{name}: {type(module).__name__}")


def lesion_activation(feature, lesion_mask):
    """
    feature: [B,C,D,H,W]
    lesion_mask: [B,1,D,H,W]
    """

    feat = feature.abs().mean(dim=1)

    lesion_act = feat[lesion_mask.squeeze(1) > 0].mean()

    background_act = feat[lesion_mask.squeeze(1) == 0].mean()

    return lesion_act.item(), background_act.item()


@hydra.main(config_path="../config", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    model_config = cfg.build
    model = Model(model_config, "kaiming")
    print_model_tree(model)

    dataloader = cfg.dataloader
    data_path = cfg.dataset
    transform = cfg.transform
    transform = instantiate(transform)

    dataset = GenericDataSet(
        data_path=data_path.isles_2022_test, transform=transform, mode="test"
    )

    dataloader = torch.utils.data.DataLoader(
        pin_memory=True, batch_size=1, dataset=dataset
    )

    checkpoint = "/home/dayan/Documents/PhD/Objective_1/checkpoints/training/experiment_higher557/BriAtt-Encoder_iter:43500.pth"
    # checkpoint = "/home/dayan/Documents/PhD/Objective_1/checkpoints/final_states/BriAtt-Encoder_19.pth"
    run_analysis(checkpoint_path=checkpoint, model=model, dataloader=dataloader)


if __name__ == "__main__":
    plot_loss_curve("metrics/lossValues/my_model_bottleneck_experimentation_22.jsonl")
    plot_validation_curve("metrics/validation/new_bottleneck.jsonl")
    # main()
