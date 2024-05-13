"ds"
from abc import ABC, abstractmethod
from typing import Any
import statistics
from ..design.CallbackModel import Callback
from .learner import Learner


class Metric(ABC, Callback):
    """Log a metric"""
    name:str = ""
    def __init__(self, train = True, test = True, aggregate_func = statistics.mean):
        self.aggregate_func = aggregate_func
        self.train = train
        self.test = test
        self.test_metrics = []

    @abstractmethod
    def _func(self, learner: "Learner") -> Any: ...

    def after_batch(self, learner: "Learner"):
        # training
        if learner.status == "train":
            learner.log(f"train {self.name}", self._func(learner))

        # testing
        elif learner.status == "test" and self.test:
            self.test_metrics.append(self._func(learner))

    def after_epoch(self, learner: "Learner"):
        if learner.status == "test" and self.test and len(self.test_metrics) > 0:
            learner.log(f"test {self.name}", self.aggregate_func(self.test_metrics))
            self.test_metrics = []
