import torch
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F

def tverskyIndex(
        y_true_flat: torch.Tensor,
        y_pred_flat: torch.Tensor,
        beta: float = 0.7
) -> float:
    
    smooth = 1
    true_positive = torch.sum(y_true_flat * y_pred_flat)
    false_positive = torch.sum((1 - y_true_flat) * y_pred_flat)
    false_negative = torch.sum(y_true_flat * (1 - y_pred_flat))

    numerator = true_positive + smooth
    denominator = true_positive + ((1 - beta) * false_negative) + \
                        (beta * false_positive) + smooth
    return numerator / denominator


def hyTver(
        config: DictConfig,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
) -> float:
    # print(y_true.shape)
    # print(y_pred.shape)
    
    y_true_flat = torch.flatten(y_true)
    y_pred_flat = torch.flatten(y_pred)

    tversky_index = tverskyIndex(y_true_flat, y_pred_flat, config.beta)
    
    # y_true_clp = torch.clip(y_true_flat, config.clip, 1.0 - config.clip)
    y_pred_clp = torch.clip(y_pred_flat, config.clip, 1.0 - config.clip)
    # print(y_true_flat)
    # print(y_pred_flat)

    ce_calc = - ((config.alpha * y_true_flat * torch.log(y_pred_clp)) + \
                        ((1 - config.alpha) * (1 - y_true_flat) * torch.log(1 - y_pred_clp)))
    
    # print(ce_calc)
    weighted_ce = torch.mean(ce_calc, axis = -1)
    print(weighted_ce)
    print(tversky_index)

    hy_tver = (config.gamma * weighted_ce) - ((1 - config.gamma) * tversky_index)

    return hy_tver
    
        
def diceLoss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        eps: float = 1e-8
 ) -> float:
    
    # print(y_true.shape)
    # print(y_pred.shape) 
    numerator = 2 * torch.sum(y_true * y_pred) + eps
    denominator = torch.sum(y_pred) + torch.sum(y_true) + eps

    return 1 - (numerator / denominator)


# class JSD(nn.Module):
#     def __init__(self):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction="none", log_target=True)
    
#     def forward(self,
#                 enc_a: torch.Tensor,
#                 enc_b: torch.Tensor
#             ):
        
