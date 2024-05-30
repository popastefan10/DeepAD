import torch

from torch import nn
from torch.nn import MSELoss
from torcheval.metrics import PeakSignalNoiseRatio
from typing import Callable


def define_mse_metric() -> MSELoss:
    return nn.MSELoss()


def define_rmse_metric() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    RMSE loss is the square root of MSE loss applied on the output and target tensors.
    """
    criterion = nn.MSELoss()

    def rmse_metric(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(criterion(output, target))

    return rmse_metric

def define_psnr_metric() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    metric = PeakSignalNoiseRatio()

    def psnr_metric(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        metric.update(output, target)
        return metric.compute()

    return psnr_metric
