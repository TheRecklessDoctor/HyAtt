import os
from typing import List
import random

import h5py
import torch
import hydra
from torch.utils.data import Dataset
from hydra.utils import instantiate
import numpy as np


def get_images(path: str) -> List[dict[str, str]]:
    subjects = []
    subjects = sorted(entry.path for entry in os.scandir(path) if entry.is_dir())
    if not subjects:
        raise FileNotFoundError(f"No file found in {path}")

    return [{'subject path':subject} for subject in subjects]


class GenericDataSet(Dataset):
    def __init__(
            self, 
            data_path:str,
            transform = None, 
            mode:str = "train"
    ):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.path = data_path
        self.image_list = self.get_image_list()

    def get_image_list(self) -> List[str]:
        
        subjects = []
        
        subjects = sorted(entry.path for entry in os.scandir(self.path) if entry.is_dir())

        if not subjects:
            raise FileNotFoundError(f"Could not find any files in {self.path}")
        
        return subjects

    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, index:int) -> dict[str, h5py.File | str]:

        subject_path = self.image_list[index]
        subject_name = subject_path.split("/")[-1]
        image_path = subject_path + "/" + f"{subject_name}_norm.h5" 
        #image_path = subject_path + "/" + f"{subject_name}.h5"

        h5f = h5py.File(image_path, 'r')

        dwi = h5f["dwi"][:]
        adc  = h5f['adc'][:]
        label = h5f["label"][:]
        h5_path = image_path

        sample = {"dwi": dwi, "adc":adc, "label":label, "file_path": h5_path}

        if self.mode == "train":
            sample =  self.transform(sample)
        
        return sample

def set_seed() -> None:
    seed = torch.initial_seed()
    np.random.seed(seed)
    random.seed(seed)






# this is to test the dataset and dataloader, you won't use this otherwise.
def get_dataloader(cfg) -> torch.utils.data.DataLoader:

    transform = instantiate(cfg.transform)
    dataset = GenericDataSet(data_path="/home/dayan/Documents/PhD/Objective_1/data", transform=transform)
    dataloader = instantiate(cfg.dataloader_main, dataset)

    return dataloader



@hydra.main(config_path="../config", config_name="dataloader", version_base="1.3")
def main(cfg):

    
    dataloader = get_dataloader(cfg)
    return dataloader

if __name__ == "__main__":

    data_path = "/home/dayan/Documents/PhD/Objective_1/data"
    dataset = GenericDataSet(data_path, mode="train")
    subjects = dataset.get_image_list()
    # print(subjects)
    
    main()
