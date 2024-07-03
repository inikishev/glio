"metrics"
from collections.abc import Callable
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import torch, torch.nn.functional as F
from .Learner import Learner
from ..design.EventModel import Callback, EventModel
from ..python_tools import type_str, get__name__
from torchzero.metrics.dice import dice
from torchzero.metrics.iou import iou

class CBMetric(Callback, ABC):
    metric:str

    def __init__(
        self,
        train=True,
        test=True,
        aggregate_func: Callable = np.mean,
        train_cond: Optional[Callable] = None,
        test_cond: Optional[Callable] = None,
    ):
        self.train, self.test = train, test
        self.aggregate_func = aggregate_func
        self.train_cond, self.test_cond = train_cond, test_cond

        self.test_metrics = []

    @abstractmethod
    def __call__(self, learner: "Learner"):
        raise NotImplementedError(f"{self.__class__.__name__} is missing `__call__`")

    def batch(self, learner: "Learner"):
        # training
        if learner.status == "train" and self.train:
            learner.log(f"train {self.metric}", self(learner))

        # testing
        elif learner.status == "test" and self.test:
            self.test_metrics.append(self(learner))

    def epoch(self, learner: "Learner"):
        if len(self.test_metrics) > 0:
            learner.log(f"test {self.metric}", self.aggregate_func(self.test_metrics))
            self.test_metrics = []

    def attach(self, learner:"EventModel"):
        learner.attach(event = "after_batch", fn = self.batch, cond = self.train_cond, name = get__name__(self), ID = id(self))
        if self.test:
            learner.attach(event =  "after_test_epoch", fn = self.epoch, cond = self.test_cond, name = get__name__(self), ID = id(self))

    def attach_default(self, learner:"EventModel"):
        learner.attach_default(event = "after_batch", fn = self.batch, cond = self.train_cond, name = get__name__(self), ID = id(self))
        if self.test:
            learner.attach_default(event =  "after_test_epoch", fn = self.epoch, cond = self.test_cond, name = get__name__(self), ID = id(self))


class Metric_Loss(CBMetric):
    """loss"""
    metric = "loss"
    def __init__(self):
        super().__init__(train = True, test = True, aggregate_func = np.mean)
    def __call__(self, learner: Learner): 
        if isinstance(learner.loss, torch.Tensor): return float(learner.loss.detach().cpu())
        return learner.loss

class Metric_LearnerFn(CBMetric):
    """called as: `fn(learner: Learner)`"""
    def __init__(self, fn:Callable[[Learner], Any], train=True, test=True, step=1, name=None):
        super().__init__(train = train, test = test, aggregate_func = np.mean, train_cond=None if step<=1 else lambda _,i: i%step==0)
        self.fn = fn
        metric = fn.__name__ if hasattr(fn, "__name__") else type_str(fn)
        if name is None: self.metric = f"fn - {metric}"
        else: self.metric = name
    def __call__(self, learner: Learner): 
        metric = self.fn(learner)
        if isinstance(metric, torch.Tensor): return float(metric.detach().cpu())
        return metric
class Metric_PredsTargetsFn(CBMetric):
    """called as: `fn(preds, targets)`"""
    def __init__(self, fn:Callable[[Any, Any], Any], train=True, test=True,  step=1, name=None):
        super().__init__(train = train, test = test, aggregate_func = np.mean, train_cond=None if step<=1 else lambda _,i: i%step==0)
        self.fn = fn
        metric = fn.__name__ if hasattr(fn, "__name__") else type_str(fn)
        if name is None: self.metric = f"fn - {metric}"
        else: self.metric = name
    def __call__(self, learner: Learner): 
        metric = self.fn(learner.preds.detach(), learner.targets.detach())
        if isinstance(metric, torch.Tensor): return float(metric.detach().cpu())
        return metric
class Metric_Accuracy(CBMetric):
    """accuracy"""
    def __init__(self, argmax_preds = True, argmax_targets = False, ignore_bg = False, step=1, name='accuracy'):
        super().__init__(train = True, test = True, aggregate_func = np.mean)
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.ignore_bg = ignore_bg
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds:
            if self.ignore_bg: preds = learner.preds[:, 1:].argmax(1)
            else: preds = learner.preds.argmax(1)
        else:
            if self.ignore_bg: raise NotImplementedError("cant ignore bg if preds arent one-hot one-hot.")
            preds = learner.preds
        if self.argmax_targets:
            if self.ignore_bg: targets = learner.targets[:, 1:].argmax(1)
            else: targets = learner.targets.argmax(1)
        else:
            if self.ignore_bg: raise NotImplementedError("cant ignore bg if targets arent one-hot.")
            targets = learner.targets
        return float(preds.eq(targets).detach().float().mean().cpu())
