"""Docstring """
import gc
from ..design.EventModel import ConditionCallback, MethodCallback
from .Learner import Learner

__all__ = [
    'CondPrintLossCB',
    "CondGCCOllectCB",
    "CondPrintFirstParamCB",
]

class CondPrintLossCB(ConditionCallback):
    """Print loss"""
    def __call__(self, learner: "Learner"):
        print(f"{learner.cur_epoch}:{learner.cur_batch} {float(learner.loss.detach().cpu()):.3f}", end='; ')

class CondGCCOllectCB(ConditionCallback):
    def __call__(self, learner:Learner):
        gc.collect()

class CondPrintFirstParamCB(MethodCallback):
    """Can help see if parameters update or not."""
    def after_train_batch(self, learner:Learner):
        print(list(learner.model.parameters())[0])