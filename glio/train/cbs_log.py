"""Docstring """
from time import perf_counter
from ..design.EventModel import ConditionCallback
from .Learner import Learner
from ..torch_tools import get_lr

__all__ = [
    "LogPredsCB",
    "LogPredsAndTargetsCB",
    "SavePredsToThisCB",
    "SavePredsToLearnerCB",
    "LogTimeCB",
    "LogLRCB",
    "LogOptimizerParamCB",
]

class SavePredsToThisCB(ConditionCallback):
    """Logs predictions to `preds`."""
    default_events = (("after_train_batch", None),)
    def __init__(self, inputs=None, preds=None, targets=None, log_inputs=False, log_preds=True, log_targets=False):
        super().__init__()
        self.log_inputs = log_inputs
        if log_inputs:
            if inputs is None: inputs = []
            self.inputs = inputs

        self.log_preds = log_preds
        if log_preds:
            if preds is None: preds = []
            self.preds = preds

        self.log_targets = log_targets
        if log_targets:
            if targets is None: targets = []
            self.targets = targets

    def __call__(self, learner: Learner):
        if self.log_inputs: self.inputs.append(learner.inputs)
        if self.log_preds: self.preds.append(learner.preds)
        if self.log_targets: self.targets.append(learner.targets)


class SavePredsToLearnerCB(ConditionCallback):
    """Logs predictions to learner attributes `train_preds_log`, 'test_preds_log`"""
    default_events = [("after_batch", None),]
    def enter(self, learner: Learner):
        learner.train_preds_log: list[list] = [] # type:ignore
        learner.test_preds_log: list[list] = [] # type:ignore
    def __call__(self, learner: Learner):
        if learner.status == "train": learner.train_preds_log.append([learner.preds, learner.targets]) # type:ignore
        elif learner.status == "test": learner.test_preds_log.append([learner.preds, learner.targets]) # type:ignore

class LogPredsCB(ConditionCallback):
    """Logs preds into logger, keys: `train preds`, `real preds`"""
    default_events = [("after_train_batch", None),]
    def __call__(self, learner: Learner):
        learner.log(f"{learner.status} preds / targets", [learner.preds, learner.targets])


class LogPredsAndTargetsCB(ConditionCallback):
    """Logs inputs and preds into logger, keys: `train preds`, `test preds`, `train targets`, `test targets`"""
    default_events = [("after_train_batch", None),]
    def __call__(self, learner: Learner):
        learner.log(f"{learner.status} preds", learner.preds)
        learner.log(f"{learner.status} targets", learner.targets)

class LogTimeCB(ConditionCallback):
    default_events = [("after_train_batch", None),]
    def enter(self, learner:Learner):
        self.start = perf_counter()

    def __call__(self, learner: Learner):
        learner.log("time", perf_counter() - self.start)

class LogLRCB(ConditionCallback):
    default_events = [("after_train_batch", None),]
    def __call__(self, learner: Learner):
        learner.log("lr", get_lr(learner.optimizer)) # type:ignore

class LogOptimizerParamCB(ConditionCallback):
    default_events = [("after_train_batch", None),]
    def __init__(self, param: str):
        super().__init__()
        self.param = param
    def __call__(self, learner: Learner):
        learner.log("lr", learner.optimizer.param_groups[0][self.param]) # type:ignore
