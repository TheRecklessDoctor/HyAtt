import os
import gc
from typing import Optional
from tqdm import tqdm
import random
from collections.abc import Callable

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.amp import autocast
from monai.networks.nets import UNETR, UNet, VNet
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.data.dataloader import GenericDataSet
from src.modelling.model import Model
from src.tools.loss_functions import diceLoss
from src.tools.logger import MetricLogger, SaveVals
from src.tools.test_patch import var_all_case, test_all_case




class Solver: 

    """Main training and testing engine."""

    def __init__(
            self,
            cfg:DictConfig
    ):
        learning_params = cfg.learning_params

        #overall
        self.seed = learning_params.experiment_var.seed
        
        #dataset variables
        self.patch_size = (learning_params.experiment_var.patch_size, learning_params.experiment_var.patch_size, learning_params.experiment_var.patch_size)
        self.dataloader = cfg.dataloader
        self.data_path = cfg.dataset
        self.transform = cfg.transform
        self.batch_size = cfg.dataloader.batch_size
        self.dataset_name = cfg.dataset.name

        #model variables
        self.weight_initialiser = learning_params.weight_initialiser.type
        self.model_config = cfg.build
        self.initial_filters = cfg.dualEncoder.n_filters_out_initial
        self.loss_hyperparams = learning_params.loss
        self.model = None

        #learning variables
        self.iter = 0
        self.epoch = 0
        self.save_model_path = None
        self.best_dice = 0
        self.use_lr_scheduler = learning_params.use_lr_scheduler
        self.use_warmup = learning_params.use_warmup
        self.scheduler = learning_params.scheduler
        self.warmup = learning_params.warmup
        self.scaler = learning_params.scaler
        self.optimizer = learning_params.optimizer
        self.val_at_iter = learning_params.experiment_var.val_at_iter
        self.n_epochs = learning_params.experiment_var.n_epochs
        self.mode = learning_params.experiment_var.mode


        #other configuration variables
        self.empty_cache = learning_params.experiment_var.empty_cache
        self._model_name = learning_params.experiment_var.model_name
        self._experiment_num = learning_params.experiment_var.experiment_num
        self.device = learning_params.experiment_var.device


    def make_dir(self, path:str) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)
    
    def save_checkpoint(
            self,
            savepath:str,
            model: torch.nn.Module,
            epoch:int,
            iter:int,
            optimizer:Optimizer,
            scaler:torch.GradScaler,
            scheduler:LRScheduler,
            config:DictConfig
    ) -> None:
        
        torch.save(
            {
                "epoch":epoch,
                "iter":iter,
                "model": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config
            },
            savepath
        )


    def get_dataset(
            self,
            dataset_name:str = "isles-24",
            mode: str = "train",
    ) -> torch.utils.data.DataLoader:

        transform = instantiate(self.transform)
        if dataset_name == "isles-24":
            dataset = GenericDataSet(
                data_path = self.data_path.train_data, 
                transform=transform, 
                mode=mode
            )
        else:
            dataset = GenericDataSet(
                data_path = self.data_path.isles_2022_train,
                transform = transform,
                mode = mode
            )

        dataloader = instantiate(self.dataloader, dataset)

        return dataloader
    

    def load_checkpoint(
            self,
            model:torch.nn.Module,
            optimizer:torch.nn.Module|None = None,
            scheduler:torch.nn.Module|None = None,
            scaler:torch.nn.Module|None = None,
            checkpoint_path: str = "",
            debug:bool = False,
    ) -> Optional[torch.nn.Module]:

        
        if self.mode == "restart":
            print(f"Loading state from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            self.epoch = checkpoint["epoch"]
            self.iter = checkpoint["iter"] + 1
        else:
            print(f"Loading state from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        
        if debug:
            print("Missing keys:")
            for k in missing_keys:
                print("  ", k)

            print("Unexpected keys:")
            for k in unexpected_keys:
                print("  ", k)
        if self.mode == "restart":
            return model, optimizer, scheduler, scaler
        else:
            return model


    def get_monai_model(
            self, 
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
                norm_name = "batch",
                dropout_rate=0.1,
                num_heads = 2,
            ).to(device)
        elif model_type == "unet":
            model = UNet(
                in_channels=2, 
                out_channels=2, 
                spatial_dims=3,
                channels=(16, 32, 64),
                strides=(2, 2, 2),
                dropout = 0.1,
                norm='batch',
                act='ReLU'
            ).to(device)
        elif model_type == 'vnet':
            model = VNet(
                in_channels=2, 
                out_channels=2, 
                spatial_dims=3,
                dropout_prob_down = 0.1,
                dropout_prob_up = (0.1, 0.1),
                act=('ReLU', {'inplace':True})
                )
        
        return model



    def train_epoch(
            self,
            dataloader:Callable,
            optimizer: Callable,
            logger: Callable,
            scaler: Optional[Callable],
            scheduler:Callable,
            device: str = "cuda",
            model_type: str = "pre-trained",
    ) -> None:
        
        # print(f"Iteration: {self.iter}")
        
        for _ , sampled_batch in enumerate(dataloader):
            dwi, adc, label = sampled_batch["dwi"], sampled_batch["adc"], sampled_batch["label"]
            
            # inputs = torch.cat([dwi, dwi], dim=1)
            
            if device == "cuda":
                dwi, adc, label = dwi.cuda(), adc.cuda(), label.cuda()
                # inputs, label = inputs.cuda(), label.cuda()

            self.model.train()
            
            with autocast(device_type="cuda", dtype=torch.float32):

                pred_outputs, _ = self.model(dwi, dwi)
                # pred_outputs = self.model(inputs)
                pred_segmentation = F.softmax(pred_outputs, dim=1)
                
                lesions = label[:self.batch_size, ...]
                # print(f"Predicted min/max: {pred_segmentation.min().item()} / {pred_segmentation.max().item()}")
                # print(f"Label sum: {lesions.sum().item()}")
                
                dice_loss = diceLoss(lesions == 1, pred_segmentation[:self.batch_size, 1,...])
                ce_loss  = F.cross_entropy(pred_outputs[:self.batch_size, ...], lesions)

                assert label.dtype == torch.long, f"Expected torch.long but got {label.dtype}"
                assert lesions.dtype == torch.long, f"Expected torch.long for lesions but got {label.dtype}"
                assert label.min() >= 0, f"Label has negative values: {label.min()}"
                assert label.max() <= 1, f"Label has value ≥ num_classes: {label.max()} >= {2}"

                loss_val = self.loss_hyperparams.alpha*dice_loss + self.loss_hyperparams.beta*ce_loss


            optimizer.zero_grad()
            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()
            self.iter += 1

            logger.add_vals(
                ["epoch", "iter","Total loss"],
                [self.epoch, self.iter, loss_val.item()]
            )

            print(str(logger))

            loss_saver = SaveVals(f"metrics/lossValues/{self._experiment_num:0>3d}_{self._model_name}_{model_type}.jsonl")

            if self.iter >= 0:
                for param_groups in optimizer.param_groups:
                    current_lr = param_groups["lr"]


                loss_saver.add_vals(
                    ["epoch", "iter", "dice", "CE", "Total loss", "current lr"],
                    [self.epoch, self.iter, dice_loss.item(), ce_loss.item(), loss_val.item(), current_lr]
                )

                loss_saver.save()

        val_saver = SaveVals(f"metrics/validation/{self._experiment_num:0>3d}_{self._model_name}_{model_type}.jsonl")
        
        # if self.iter  == 0 or self.iter % self.val_at_iter == 0:

        self.model.eval()

        # patch-based metric calculation
        dice_val, median_dice = var_all_case(
            model = self.model, 
            num_classes=2, 
            logger = logger,
            patch_size=self.patch_size, 
            stride_xy=20, 
            stride_z=20,
            iter = self.iter,
            dataset_name = self.dataset_name
            )
        
        
        val_saver.add_vals(
            ["epoch", "iter", "dice_mean", "dice_median"],
            [self.epoch, self.iter, dice_val, median_dice]
        )
        val_saver.save()

        dir_path = f"checkpoints/training/experiment_{self._experiment_num}"
        self.make_dir(dir_path)

        best_path = f"checkpoints/best_checkpoints/experiment_{self._experiment_num}"
        self.make_dir(best_path)

        if dice_val > self.best_dice:
            self.best_dice = dice_val
            self.save_model_path = f"checkpoints/best_checkpoints/experiment_{self._experiment_num}/{self._model_name}_best_model_{model_type}.pth"
            # torch.save(self.model.state_dict(), self.save_model_path)
            self.save_checkpoint(
                savepath=self.save_model_path,
                model=self.model,
                epoch=self.epoch,
                iter=self.iter,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                config=self.model_config
            )

        self.save_model_path = f"checkpoints/training/experiment_{self._experiment_num}/{self._model_name}_iter:{self.iter}_{model_type}.pth"
        # torch.save(self.model.state_dict(), self.save_model_path)
        self.save_checkpoint(
            savepath=self.save_model_path,
            model=self.model,
            epoch=self.epoch,
            iter=self.iter,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            config=self.model_config
        )
        

    
    def train(
            self,
            model_type: str = "pre-trained",
            checkpoint_path:str = ""
    ) -> None:

        gc.collect()
        torch.cuda.empty_cache()

        loader = self.get_dataset(
            dataset_name = self.dataset_name,
            mode = "train"
        )

        model = Model(
                self.model_config,
                weight_initialiser=self.weight_initialiser
        )

        # model = self.get_monai_model(model_type = "unet")

        if self.device == "cuda":
            self.model = model.cuda()
        else:
            self.model = model

        # self.model = model

        logger = MetricLogger()

        optimizer = instantiate(self.optimizer, params=model.parameters())

        if self.use_lr_scheduler:
            scheduler = instantiate(self.scheduler, optimizer = optimizer)
        if self.use_warmup:
            warmup = instantiate(self.warmup, optimizer = optimizer)

        scaler = instantiate(self.scaler)

        if self.mode == "restart":

            self.model, optimizer, scheduler, scaler = self.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler
            )

        iterator = tqdm(range(self.n_epochs), ncols=70)

        logger.add_vals(
            [" Total number of parameters"],
            [sum(p.numel() for p in self.model.parameters())]
        )
        print(str(logger))

        for _ in iterator:

            self.train_epoch(
                dataloader=loader,
                optimizer=optimizer,
                logger = logger,
                scaler = scaler,
                device = "cuda",
                model_type = model_type,
                scheduler=scheduler
            )
            
            self.epoch += 1

            if self.use_lr_scheduler:
                if self.use_warmup and self.epoch < 5:
                    warmup.step()
                else:
                    scheduler.step()


        self.save_model_path = f"checkpoints/final_states/{self._model_name}_{self._experiment_num}.pth"
        torch.save(
            {
                "model": self.model.state_dict()
            },
            self.save_model_path
        )
        logger.log_msg(self.save_model_path)


    def test(
            self,
            pre_trained_path: str = None,
            model_type: str = "pre-trained"
    ) -> None:
        
        self.mode = "test"

        model = Model(
                self.model_config,
                weight_initialiser=self.weight_initialiser
        )

        # model = self.get_monai_model(
        #     model_type = "unet"
        # )

        model = self.load_checkpoint(
            checkpoint_path=pre_trained_path,
            model=model,
            debug=True
        )

        if self.device == "cuda":
            model = model.cuda()


        # model = self.get_monai_model(mode = "test", state_dict = pre_trained_path)
        logger = MetricLogger()
        # saver = SaveVals(f"testing/{self._experiment_num}_{self._model_name}_fineTune.jsonl")


        dir_path = f"visualisation/images/experiment_{self._experiment_num}"
        self.make_dir(dir_path)

        avg_metric, median_metric, individual_metrics = test_all_case(
            model=model,
            num_classes=2,
            logger=logger,
            patch_size=(80, 80, 80),
            stride_xy=20, 
            stride_z=20,
            test_save_path="visualisation",
            experiment_num=self._experiment_num,
            nms=0,
            missing_holes=0,
            dataset_name = self.dataset_name
        )


        saver = SaveVals(f"metrics/test/{self._experiment_num}_{self._model_name}_{model_type}.jsonl")
        saver.add_vals(
            ["dice", "jc", "hd", "asd", "assd", "precision", "se", "sp", "F1"],
            avg_metric
        )
        saver.save()

        saver.add_vals(
            ["med_dice", "med_jc", "med_hd", "med_asd", "med_assd", "med_precision", "med_se", "med_sp", "med_F1"],
            median_metric
        )

        saver.save()

        # saver = SaveVals(f"metrics/test/{self._experiment_num}_{self._model_name}_{model_type}.jsonl")
        # saver.add_vals(
        #     ["dice", "f1","ald", "precision", "se", "sp"],
        #     avg_metric
        # )
        # saver.save()

        # saver.add_vals(
        #     ["med_dice", "f1", "ald", "med_precision", "med_se", "med_sp"],
        #     median_metric
        # )

        # saver.save()

        saver_2 = SaveVals(f"metrics/test/{self._experiment_num}_{self._model_name}_isles22_individual_metrics.jsonl")
        for i in individual_metrics:
            saver_2.add_vals(
                ["dice", "jc", "hd", "asd", "assd", "precision", "se", "sp", "F1"],
                i
            )
            saver_2.save()

        saver_2 = SaveVals(f"metrics/test/{self._experiment_num}_{self._model_name}_isles22_individual_metrics.jsonl")
        for i in individual_metrics:
            saver_2.add_vals(
                ["dice", "f1", "ald", "assd", "precision", "se", "sp"],
                i
            )
            saver_2.save()
        

    def fit(
            self,
            cfg: DictConfig
    ) -> None:
        
        #Reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        #use deterministic or randomised algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)

        print("*****Starting Training*****")
        # checkpoint_path = "checkpoints/training/experiment_8/BriAtt-Encoder_iter:43500_train.pth"
        self.train("train")

        print("*****Starting Testing*****")
        # path = "checkpoints/best_checkpoints/experiment_8/BriAtt-Encoder_best_model_train.pth"
        # print(self.save_model_path)
        self.test(pre_trained_path=self.save_model_path)

        print("*****Training and Testing Complete*****")



@hydra.main(config_path="../config", config_name="main", version_base="1.3")
def main(
        cfg: DictConfig
):
        solver = Solver(
            cfg = cfg
    )
        solver.fit(cfg)


if __name__ == "__main__":
    main()

          
    
