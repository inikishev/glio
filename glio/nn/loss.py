"losses"
from collections.abc import Callable, Sequence
from typing import Optional
import torch
from .container import to_module
class CombinedLoss_Mean(torch.nn.Module):
    def __init__(self, losses:Sequence[torch.nn.Module | Callable], weights:Optional[Sequence[int|float]] = None):
        super().__init__()
        self.losses = torch.nn.ModuleList([to_module(i) for i in losses])
        if weights is not None: self.weights = [i / sum(weights) for i in weights]
        else: weights = [1/len(losses) for _ in range(len(losses))]
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_func, weight in zip(self.losses, self.weights):
            loss += loss_func(y_true, y_pred) * weight
        return loss # type:ignore

class CombinedLoss_Sum(torch.nn.Module):
    def __init__(self, losses:Sequence[torch.nn.Module | Callable], weights:Optional[Sequence[int|float]] = None):
        super().__init__()
        self.losses = torch.nn.ModuleList([to_module(i) for i in losses])
        if weights is None: self.weights = [1 for _ in range(len(losses))]
        else: self.weights = weights
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_func, weight in zip(self.losses, self.weights):
            loss += loss_func(pred, target) * weight
        return loss # type:ignore

class BinaryToMulticlassLoss(torch.nn.Module):
    """
    Computes a binary loss for a multiclass problem per each channels weighted by `weights`.
    Expects a `BC..` tensor, where `B` is the batch size, `C` is the channel in one-hot encoded array, and `.` is the number of dimensions.
    """
    def __init__(self, loss:torch.nn.Module | Callable, weights:Optional[Sequence[int | float]] = None):
        super().__init__()
        self.loss = loss
        if weights is None: self.weights = [1 for _ in range(len(self.base))]
        else: self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_channels = torch.unbind(pred, 1)
        target_channels = torch.unbind(target, 1)
        loss = 0
        for pch, tch, w in zip(pred_channels, target_channels, self.weights):
            loss += self.loss(pch, tch) * w
        return loss # type:ignore


class ConvertBCHWLossToBCXYZ(torch.nn.Module):
    """
    Converts a 2d loss to a 3d loss, where the last dimension is the channel dimension.
    Forward therefore expects a `BCXYZ` tensor, which will be unpacked into a `BCHW` tensor by unrolling X dim.
    """
    def __init__(self, loss:torch.nn.Module | Callable):
        super().__init__()
        self.loss = loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred.view(pred.shape[0],pred.shape[1],  pred.shape[2]*pred.shape[3], pred.shape[4]),
                         target.view(target.shape[0], target.shape[1],  target.shape[2]*target.shape[3], target.shape[4]))

