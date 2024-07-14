"fastprogress"
from collections.abc import Sequence
import numpy as np
from scipy.signal import convolve
from fastprogress.fastprogress import master_bar, progress_bar
from ..design.EventModel import MethodCallback
from .Learner import Learner

__all__ = [
    "FastProgressBarCB",
]
class FastProgressBarCB(MethodCallback):
    order = 90
    def __init__(
        self,
        metrics=("train loss", "test loss"),
        plot=False,
        step_batch=16,
        step_epoch=None,
        fit=True,
        plot_max=4096,
        smooth:None | int | Sequence[None | int]=None,
        maxv=None,
    ):
        super().__init__()
        self.plot = plot
        if isinstance(metrics, str): metrics = [metrics]
        self.metrics = metrics
        self.plot_max = plot_max
        self.b = step_batch
        self.e = step_epoch
        self.fit = fit
        if isinstance(smooth, int): smooth = [smooth for _ in range(len(metrics))]
        self.smooth = smooth
        self.maxv= maxv

    def before_fit(self, learner:Learner):
        self.mbar = learner.epochs_iterator = master_bar(learner.epochs_iterator) # type:ignore

    def before_any_epoch(self, learner:Learner):
        learner.dl = progress_bar(learner.dl, leave=False, parent=self.mbar)

    def _plot(self, learner: Learner):
        if self.plot:
            metrics = [learner.logger[metric] for metric in self.metrics if metric in learner.logger]
            metrics = [(i if self.maxv is None else dict(zip(i.keys(), np.clip(list(i.values()), a_min=None, a_max=1)))) for i in metrics if len(i) > 0]
            if len(metrics) > 0:
                reduction = [max(int(len(metric) / self.plot_max), 1) for metric in metrics]
                metrics = [([list(m.keys())[::reduction[i]], list(m.values())[::reduction[i]]] if reduction[i]>1 else [list(m.keys()), list(m.values())]) for i, m in enumerate(metrics)]
                if self.smooth:
                    for i in range(len(metrics)):
                        if self.smooth[i] is not None and self.smooth[i] > 1 and len(metrics[i][1]) > self.smooth[i]: # type:ignore
                            metrics[i][1] = convolve(metrics[i][1], np.ones(self.smooth[i])/self.smooth[i], 'same') # type:ignore
                try:
                    self.mbar.update_graph(metrics)
                except AttributeError: pass
    def after_train_batch(self, learner: Learner):
        if self.b and learner.cur_batch % self.b == 0: self._plot(learner)
    def after_train_epoch(self, learner: Learner):
        if self.e and learner.cur_epoch % self.e == 0: self._plot(learner)
    def after_fit(self, learner: Learner):
        if self.fit: self._plot(learner)
