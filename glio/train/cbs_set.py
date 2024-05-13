"""Присваивание"""
from typing import Callable
import torch
from ..design.CallbackModel import Callback
from .learner import Learner
class Set_Optimizer(Callback):
    """Sets optimizer"""
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def enter(self, learner: "Learner"):
        self.backup = learner.optimizer
        learner.optimizer = self.optimizer

    def exit(self, learner: "Learner"):
        learner.optimizer = self.backup

class Set_LossFn(Callback):
    """Sets loss function"""
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn

    def enter(self, learner: "Learner"):
        self.backup = learner.loss_fn
        learner.loss_fn = self.loss_fn

    def exit(self, learner: "Learner"):
        learner.loss_fn = self.backup

class Set_Scheduler(Callback):
    """Sets scheduler"""
    def __init__(self, scheduler: torch.optim.lr_scheduler.LRScheduler):
        self.scheduler = scheduler

    def enter(self, learner: "Learner"):
        self.backup = learner.scheduler
        learner.scheduler = self.scheduler

    def exit(self, learner: "Learner"):
        learner.scheduler = self.backup
