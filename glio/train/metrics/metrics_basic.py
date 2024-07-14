"metrics"
from collections.abc import Callable, Mapping
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import torch, torch.nn.functional as F
from .metric_callback import MetricCallback
from ..Learner import Learner
from ...design.EventModel import Callback, EventModel
from ...python_tools import type_str, get__name__


__all__ = [
    "LogLossCB",
    "LogLearnerFnCB",
    "LogPredsTargetsFnCB",
    "MetricAccuracyCB",
]

class LogLossCB(MetricCallback):
    """loss"""
    metric = "loss"
    def __init__(self):
        super().__init__(train = True, test = True)
    def __call__(self, learner: Learner): 
        if isinstance(learner.loss, torch.Tensor): return float(learner.loss.detach().cpu())
        return learner.loss

class LogLearnerFnCB(MetricCallback):
    """called as: `fn(learner: Learner)`"""
    def __init__(self, fn:Callable[[Learner], Any], train=True, test=True, step=1, name=None):
        super().__init__(train = train, test = test, batch_cond=None if step<=1 else lambda _,i: i%step==0)
        self.fn = fn
        metric = fn.__name__ if hasattr(fn, "__name__") else type_str(fn)
        if name is None: self.metric = f"fn - {metric}"
        else: self.metric = name
    def __call__(self, learner: Learner): 
        metric = self.fn(learner)
        if isinstance(metric, torch.Tensor): return float(metric.detach().cpu())
        return metric

class LogPredsTargetsFnCB(MetricCallback):
    """called as: `fn(preds, targets)`"""
    def __init__(self, fn:Callable[[Any, Any], Any], train=True, test=True,  step=1, name=None):
        super().__init__(train = train, test = test, batch_cond=None if step<=1 else lambda _,i: i%step==0)
        self.fn = fn
        metric = fn.__name__ if hasattr(fn, "__name__") else type_str(fn)
        if name is None: self.metric = f"fn - {metric}"
        else: self.metric = name
    def __call__(self, learner: Learner): 
        metric = self.fn(learner.preds.detach(), learner.targets.detach())
        if isinstance(metric, torch.Tensor): return float(metric.detach().cpu())
        return metric

class MetricAccuracyCB(MetricCallback):
    """Accuracy. Expects preds to be BC* (probabilities of each class), targets - B* (class indexes)"""
    def __init__(self, step=1, name='accuracy'):
        super().__init__(train = True, test = True)
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        return float(learner.preds.amax(1).eq(learner.targets).detach().float().mean().cpu())

