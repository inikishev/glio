"""asdnjkhaoldloasdnasiujadklodigsdauhlasdasdaskdbasdbaskhdaskjhdasdhjkbaskdbhaskdhsakdhlasdlasdas."""
from typing import Callable
import torch
from ..design.EventModel import BasicCallback
from .Learner import Learner

__all__ = [
    "SetOptimizerCB", 
    "SetLossFnCB", 
    "SetSchedulerCB",
]
class SetOptimizerCB(BasicCallback):
    """Sets optimizer"""
    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.optimizer = optimizer

    def enter(self, learner: "Learner"):
        self.backup = learner.optimizer
        learner.optimizer = self.optimizer

    def exit(self, learner: "Learner"):
        learner.optimizer = self.backup

class SetLossFnCB(BasicCallback):
    """Sets loss function"""
    def __init__(self, loss_fn: Callable):
        super().__init__()
        self.loss_fn = loss_fn

    def enter(self, learner: "Learner"):
        self.backup = learner.loss_fn
        learner.loss_fn = self.loss_fn

    def exit(self, learner: "Learner"):
        learner.loss_fn = self.backup

class SetSchedulerCB(BasicCallback):
    """Sets scheduler"""
    def __init__(self, scheduler: torch.optim.lr_scheduler.LRScheduler):
        super().__init__()
        self.scheduler = scheduler

    def enter(self, learner: "Learner"):
        self.backup = learner.scheduler
        learner.scheduler = self.scheduler

    def exit(self, learner: "Learner"):
        learner.scheduler = self.backup

class SetStatusCB(BasicCallback):
    def __init__(self, status): 
        self.status = status
    def enter(self, learner: Learner):
        self.status_backup = learner.status
        learner.status = self.status
    def exit(self, learner: Learner):
        learner.status = self.status_backup