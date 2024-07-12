import torch
from ..design.EventModel import MethodCallback
from .Learner import Learner
from ..torch_tools import copy_state_dict

__all__ = [
    "ResetOptimizerOnTestLossIncreaseCB",
]

class ResetOptimizerOnTestLossIncreaseCB(MethodCallback):
    order=10
    def before_fit(self, learner:Learner):
        self.empty_state_dict = copy_state_dict(learner.optimizer.state_dict()) # type:ignore
        self.previous_test_loss = float('inf')
    def after_test_epoch(self, learner:Learner):
        test_loss = learner.logger.last('test loss')
        if test_loss > self.previous_test_loss:
            learner.optimizer.load_state_dict(copy_state_dict(self.empty_state_dict)) # type:ignore
        self.previous_test_loss = test_loss