"""Docstring """
import gc
from ..design.EventModel import CBCond, CBMethod
from .Learner import Learner

class Debug_PrintLoss(CBCond):
    """Print loss"""
    def __call__(self, learner: "Learner"):
        print(f"{learner.cur_epoch}:{learner.cur_batch} {float(learner.loss.detach().cpu()):.3f}", end='; ')

class Debug_GCCollect(CBCond):
    def __call__(self, learner:Learner):
        gc.collect()

class Debug_PrintFirstParam(CBMethod):
    def after_train_batch(self, learner:Learner):
        print(list(learner.model.parameters())[0])