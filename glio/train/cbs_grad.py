"""Docstring """
import torch
from ..design.EventModel import CBEvent
from .Learner import Learner


class GradientClipNorm(CBEvent):
    event = "optimizer_step"
    order = -10
    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = max_norm

    def __call__(self, learner: "Learner"):
        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), self.max_norm) # type: ignore

class GradientClipValue(CBEvent):
    event = "optimizer_step"
    order = -9
    def __init__(self, clip_value: float):
        super().__init__()
        self.clip_value = clip_value

    def __call__(self, learner: "Learner"):
        torch.nn.utils.clip_grad_value_(learner.model.parameters(), self.clip_value) # type: ignore
