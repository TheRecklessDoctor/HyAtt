import matplotlib.pyplot as plt
import numpy as np
import jsonlines
import pandas as pd


class Visualiser():

    def __init__(
            self,
            experiment_num:int
    ):
        self.exp_num = experiment_num
    
    def plot_validation_curve(
            self,
            file_path: str
    ):
        iters = []
        comp1 = []
        with jsonlines.open(file=file_path) as reader:
            for obj in reader:
                epoch = int(obj["iter"])/100
                iters.append(epoch)
                comp1.append(obj["dice_mean"])

        
        plt.figure(figsize=(7.5, 5))
        plt.plot(iters, comp1)
        plt.title("Validation dice values over the epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        # plt.savefig(f"visualisation/plots/{self.exp_num}_validation.png")
        plt.show()
        plt.close()
    
    def plot_loss_curve(self,
                       file_path: str
                    ):
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
        # plt.savefig(f"visualisation/plots/{self.exp_num}_training.png")
        plt.show()
        plt.close()








if __name__ == "__main__":

    visu = Visualiser(experiment_num=7)
    visu.plot_loss_curve("metrics/lossValues/021_BriAtt-Encoder_train.jsonl")
    visu.plot_validation_curve("metrics/validation/021_BriAtt-Encoder_train.jsonl")
    # visu.plot_validation_curve("src/outputs/validation/003_BriAtt-Encoder.jsonl")
