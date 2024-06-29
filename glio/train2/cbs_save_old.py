"""Docstring """
import os, shutil
from collections.abc import Mapping, Sequence
from ..design.EventModel import CBCond, CBEvent
from .Learner import Learner

class Save_Best(CBEvent):
    event = "after_test_epoch"
    def __init__(
        self,
        folder="checkpoints",
        metrics: Mapping[str, str] = {"test loss": "low", "test accuracy": "high"},
        serialize=False,
    ):  # pylint:disable=W0102
        super().__init__()
        self.folder = folder
        self.serialize = serialize
        if isinstance(metrics, Sequence): metrics = {m: ("low" if "loss" in m else "high") for m in metrics}
        if not isinstance(metrics, Mapping): metrics = {m: ("low" if "loss" in m else "high") for m in (metrics, )}

        self.metrics = {k:v.lower() for k,v in metrics.items()}
        self.best_metrics = {k:float("inf") if v == "low" else -float("inf") for k,v in metrics.items()}

    def __call__(self, learner: Learner):
        for met, target in self.metrics.items():
            if met in learner.logger:
                val = learner.logger.last(met)
                if target == "low":
                    if  val < self.best_metrics[met]:
                        if os.path.exists(self.folder):
                            shutil.rmtree(self.folder)
                            os.mkdir(self.folder)
                        learner.checkpoint(dir = self.folder, serialize=self.serialize)
                        self.best_metrics[met] = val
                else:
                    if val > self.best_metrics[met]:
                        if os.path.exists(self.folder):
                            shutil.rmtree(self.folder)
                            os.mkdir(self.folder)
                        learner.checkpoint(dir = self.folder, serialize=self.serialize)
                        self.best_metrics[met] = val

class Save_Last(CBEvent):
    event = "after_fit"
    def __init__(self, folder = "checkpoints", serialize=False): #pylint:disable=W0102
        super().__init__()
        self.folder = folder
        self.serialize = serialize

    def __call__(self, learner: Learner):
        learner.checkpoint(dir = self.folder, serialize=self.serialize)
