"""Docstring """
from time import perf_counter
from ..design.CallbackModel import Callback
from .learner import Learner

class Write_Preds(Callback):
    """Записывает предсказанные и реальные значения в атрибуты `train_preds_log`, 'test_preds_log`"""
    def enter(self, learner: Learner):
        learner.train_preds_log: list[list] = [] # type:ignore
        learner.test_preds_log: list[list] = [] # type:ignore
    def after_batch(self, learner: Learner):
        if learner.status == "train": learner.train_preds_log.append([learner.preds, learner.targets]) # type:ignore
        elif learner.status == "test": learner.test_preds_log.append([learner.preds, learner.targets]) # type:ignore

class Log_Preds(Callback):
    """Записывает предсказанные и реальные значения в одно поле под `train preds`, `real preds`"""
    def after_batch(self, learner: Learner):
        learner.log(f"{learner.status} preds / targets", [learner.preds, learner.targets])


class Log_PredsSep(Callback):
    """Записывает предсказанные и реальные значения под `train preds`, `test preds`, `train targets`, `test targets`"""
    def after_batch(self, learner: Learner):
        learner.log(f"{learner.status} preds", learner.preds)
        learner.log(f"{learner.status} targets", learner.targets)

class Log_Time(Callback):
    def __init__(self):
        self.start = perf_counter()

    def after_batch(self, learner: Learner):
        if learner.status == "train":
            learner.log("time", perf_counter() - self.start)
