import hiddenlayer as hl, numpy as np
from ..design.EventModel import CBCond
from .Learner import Learner
from ..logger import Logger

class _Hiddenlayer_Logger_Metric():
    """Represents the history of a single metric."""
    def __init__(self, logger:Logger, metric):
        self.name = metric
        self.data = logger.toarray(metric)
        self.steps = None

    @property
    def formatted_steps(self):
        return None


class HLCanvas(CBCond):
    def __init__(self, plot=("train loss", "test loss", "train accuracy", "test accuracy"), image = (), hist = ()):
        super().__init__()
        self.canvas = hl.Canvas()
        self.plot = plot
        self.image = image
        self.hist = hist

    def __call__(self, learner:Learner):
        with self.canvas:
            if self.plot: self.canvas.draw_plot(_Hiddenlayer_Logger_Metric(learner.logger, self.plot), self.plot)
            if self.image: self.canvas.draw_image(_Hiddenlayer_Logger_Metric(learner.logger, self.image), self.image)
            if self.plot: self.canvas.draw_hist(_Hiddenlayer_Logger_Metric(learner.logger, self.hist), self.hist)