"""Docstring """
import gc
from ..design.event_model import ConditionCallback, MethodCallback
from .Learner import Learner

__all__ = [
    'CondPrintLossCB',
    "CondGCCOllectCB",
    "CondPrintFirstParamCB",
    "CondPrintStatusCB",
]

class CondPrintLossCB(ConditionCallback):
    """Print loss"""
    def __call__(self, learner: "Learner"):
        print(f"{learner.cur_epoch}:{learner.cur_batch} {float(learner.loss.detach().cpu()):.3f}", end='; ')

class CondGCCOllectCB(ConditionCallback):
    def __call__(self, learner:Learner):
        gc.collect()

class CondPrintFirstParamCB(ConditionCallback):
    """Can help see if parameters update or not."""
    def __call__(self, learner:Learner):
        print(list(learner.model.parameters())[0])
        
        
class CondPrintStatusCB(ConditionCallback):
    def __call__(self, learner: "Learner"):
        print(f'{learner.cur_batch = }; {learner.cur_epoch = }; {learner.status = }', end="; ")