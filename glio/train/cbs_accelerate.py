"1"
from accelerate import Accelerator
from ..design.CallbackModel import Callback
from ..python_tools import type_str
from .learner import Learner

class Accelerate(Callback):
    """https://huggingface.co/docs/accelerate/en/index"""
    ORDER = 10
    def __init__(self, precision = 'fp16'):
        self.accelerator = Accelerator(mixed_precision = precision)
        self.device = self.accelerator.device

    def enter(self, learner: "Learner"):
        self.backup_accelerator = learner.accelerator
        self.backup_device = learner.device
        learner.accelerator = self.accelerator
        learner.device = self.device

    def exit(self, learner: "Learner"):
        learner.accelerator = self.backup_accelerator
        learner.device = self.backup_device

    def before_fit(self, learner: "Learner"):
        (
            learner.model,
            learner.optimizer,
            learner.scheduler,
            learner.dl_train,
            learner.dl_test,
        ) = self.accelerator.prepare(
            learner.model,
            learner.optimizer,
            learner.scheduler,
            learner.dl_train,
            learner.dl_test,
        )

    def __str__(self):
        return f"{type_str(self)}(precision = {self.accelerator.mixed_precision})"
