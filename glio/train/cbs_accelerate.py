"1"
from accelerate import Accelerator
from ..design.EventModel import EventCallback
from ..python_tools import type_str
from .Learner import Learner

__all__ = [
    'AccelerateCB'
]
class AccelerateCB(EventCallback):
    """https://huggingface.co/docs/accelerate/en/index"""
    order = 10
    event = "before_fit"
    def __init__(self, mixed_precision = None, cpu=False):
        super().__init__()
        self.accelerator = Accelerator(mixed_precision = mixed_precision, cpu=cpu)
        self.device = self.accelerator.device

    def enter(self, learner: "Learner"):
        self.backup_accelerator = learner.accelerator
        self.backup_device = learner.device
        learner.accelerator = self.accelerator
        learner.device = self.device

    def exit(self, learner: "Learner"):
        learner.accelerator = self.backup_accelerator
        learner.device = self.backup_device

    def __call__(self, learner: "Learner"):
        (
            learner.model,
            learner.optimizer,
            learner.scheduler,
            learner.dltrain,
            learner.dltest,
        ) = self.accelerator.prepare(
            learner.model,
            learner.optimizer,
            learner.scheduler,
            learner.dltrain,
            learner.dltest,
        )

    def __str__(self):
        return f"{type_str(self)}(precision = {self.accelerator.mixed_precision})"
