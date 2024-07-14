"""Docstring """
import torch
from ..design.EventModel import EventCallback
from .Learner import Learner

__all__ = [
    "PrintSummaryCB",
    "DisplayLoggerTableCB",
]

class PrintSummaryCB(EventCallback):
    event = "after_fit"
    order = -10
    def __init__(self, metrics = None):
        super().__init__()
        self.metrics = metrics

    def __call__(self, learner: "Learner"):
        if self.metrics is None:
            self.metrics = [i for i in learner.logger.get_keys_num() if '/' not in i]
        for m in self.metrics:
            print(f"{m}: min: {learner.logger.min(m):.4f}; max: {learner.logger.max(m):.4f}; last: {learner.logger.last(m):.4f}")


class DisplayLoggerTableCB(EventCallback):
    event = "after_fit"
    order = -10

    def __call__(self, learner: "Learner"):
        learner.logger.display_table()