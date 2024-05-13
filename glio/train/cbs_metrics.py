"Метрики"
from collections.abc import Callable
from typing import Any
import statistics
import torch
from .learner import Learner
from . import metrics_base
from ..python_tools import type_str


class Metric_Loss(metrics_base.Metric):
    """Log loss every train batch and average loss on test split"""
    name = "loss"
    def _func(self, learner: Learner): return float(learner.loss.detach().cpu())

class Metric_Fn(metrics_base.Metric):
    """Log fn(preds, targets) every train batch and average loss on test split"""
    def __init__(self, fn:Callable[[Any, Any], Any]):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.fn = fn
        name = fn.__name__ if hasattr(fn, "__name__") else type_str(fn)
        self.name = f"fn - {name}"
    def _func(self, learner: Learner): return float(self.fn(learner.preds, learner.targets).detach().cpu())


class Metric_Accuracy(metrics_base.Metric):
    """Log accuracy every train batch and average loss on test split"""
    name = "accuracy"
    def _func(self, learner: Learner):
        if learner.preds.ndim == 2:
            return float(learner.preds.argmax(1).eq(learner.targets).detach().float().mean().cpu())
        else:
            return float(torch.round(learner.preds.eq(learner.targets)).detach().float().mean().cpu())
