from collections.abc import Sequence
import math
from ..design.EventModel import CBMethod
from .Learner import Learner
from ..progress_bar import PBar

_NEWLINE = '\n'
class SimpleProgressBar(CBMethod):
    order = 90
    def __init__(
        self,
        length = 50,
        step = 0.01,
        metrics:Sequence[str]=("train loss", "test loss", "train accuracy", "test accuracy"),
        metric_step = None,
        ):
        super().__init__()

        self.length = length
        self.step = step
        self.metrics = metrics
        self.metric_step = metric_step
        self.bar_epoch:PBar = PBar((0,1), self.length, 1)
        self.bar_batch:PBar = PBar((0,1), self.length, self.step)

    def before_fit(self, learner:Learner):
        self.bar_epoch.set_obj(learner.epochs_iterator)
        learner.epochs_iterator = self.bar_epoch

    def before_any_epoch(self, learner:Learner):
        self.bar_batch.set_obj(learner.dl)
        learner.dl = self.bar_batch
        if self.metric_step is None: self._metric_step_actual = self.bar_batch._actual_step
        elif self.metric_step < 1: self._metric_step_actual = int(math.ceil(len(self.bar_batch) * self.metric_step))
        else: self._metric_step_actual = int(self.metric_step)

    def _write(self, learner:Learner):
        metrics = [metric for metric in self.metrics if metric in learner.logger and len(learner.logger[metric]) > 0]
        text=''
        for metric in metrics:
            text += f"\n{(metric+':').ljust(40)} last = {learner.logger.last(metric):.3f}, min = {learner.logger.min(metric):.3f}, max = {learner.logger.max(metric):.3f}"
        self.bar_batch.write(text)

    def after_train_batch(self, learner: Learner):
        if learner.cur_batch % self._metric_step_actual == 0: self._write(learner)
    def after_train_epoch(self, learner: Learner):
        self._write(learner)
    def after_fit(self, learner: Learner):
        self._write(learner)
