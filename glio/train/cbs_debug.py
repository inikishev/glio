"""Docstring """
import gc
from ..design.CallbackModel import Callback
from .learner import Learner

class PrintLoss(Callback):
    """Print loss every train batch"""
    def after_batch(self, learner: "Learner"):
        print(f"{learner.cur_epoch}:{learner.cur_batch} {float(learner.loss.detach().cpu()):.3f}", end='; ')

class GCCollect(Callback):
    def __init__(self, step_batch = 16):
        self.step_batch = step_batch

    def after_batch(self, learner:Learner):
        if learner.cur_batch % self.step_batch == 0:
            gc.collect()
