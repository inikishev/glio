"""Docstring """
import torch
from ..design.CallbackModel import Callback
from .learner import Learner


class GradientClipNorm(Callback):
    def __init__(self, max_norm: float): self.max_norm = max_norm
    def before_optimizer_step(self, learner: "Learner"):
        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), self.max_norm) # type: ignore

class GradientClipValue(Callback):
    def __init__(self, clip_value: float): self.clip_value = clip_value
    def before_optimizer_step(self, learner: "Learner"):
        torch.nn.utils.clip_grad_value_(learner.model.parameters(), self.clip_value) # type: ignore
