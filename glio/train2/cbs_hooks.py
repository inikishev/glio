"Методы, регистрирующиеся на прямой или обратный проход через каждый модуль при помощи `register_forward_hook` и `register_backward_hook`"
import torch
from .Learner import Learner
from ..design.EventModel import CBContext
from . hooks_base import LearnerForwardHook, LearnerBackwardHook
from ..torch_tools import is_container

class Log_LayerSignalDistribution(LearnerForwardHook, CBContext):
    def __init__(self, step: int = 1, mean = True, std = True, var = True, min = True, max = True, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.mean, self.std, self.var, self.min, self.max = mean, std, var, min, max
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            if self.mean: learner.log(f'{name} output mean', outputs.mean())
            if self.std: learner.log(f'{name} output std', outputs.std())
            if self.var: learner.log(f'{name} output var', outputs.var())
            if self.min: learner.log(f'{name} output min', outputs.min())
            if self.max: learner.log(f'{name} output max', outputs.max())

class Log_LayerSignalHistorgram(LearnerForwardHook, CBContext):
    def __init__(self, step: int = 1, range = 10, bins = 60, top_dead = 1, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.range, self.bins, self.top_dead = range, bins, top_dead
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            hist = outputs.float().histc(self.bins, 0 - self.range, 0+self.range)
            learner.log(f'{name} output histogram', hist)
            zero = int(len(hist)/2)
            if self.top_dead is not None: learner.log(f'{name} dead activations', hist[zero-self.top_dead:zero+self.top_dead].sum()/hist.sum())

class Log_LayerGradDistribution(LearnerBackwardHook, CBContext):
    def __init__(self, step: int = 1, mean = True, std = True, var = True, min = True, max = True, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.mean, self.std, self.var, self.min, self.max = mean, std, var, min, max
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            if grad_input[0] is not None:
                if self.mean: learner.log(f'{name} input grad mean', grad_input[0].mean())
                if self.std: learner.log(f'{name} input grad std', grad_input[0].std())
                if self.var: learner.log(f'{name} input grad var', grad_input[0].var())
                if self.min: learner.log(f'{name} input grad min', grad_input[0].min())
                if self.max: learner.log(f'{name} input grad max', grad_input[0].max())

            if grad_output[0] is not None:
                if self.mean: learner.log(f'{name} output grad mean', grad_output[0].mean())
                if self.std: learner.log(f'{name} output grad std', grad_output[0].std())
                if self.var: learner.log(f'{name} output grad var', grad_output[0].var())
                if self.min: learner.log(f'{name} output grad min', grad_output[0].min())
                if self.max: learner.log(f'{name} output grad max', grad_output[0].max())

class Log_LayerGradHistorgram(LearnerBackwardHook, CBContext):
    def __init__(self, step: int = 1, range = None, bins = 60, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.range, self.bins = range, bins
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            if len(grad_input) > 1: print(f"{name} градиент grad_input {len(grad_input)} тензоров")
            if len(grad_output) > 1: print(f"{name} градиент grad_output {len(grad_input)} тензоров")
            if grad_input[0] is not None:
                if self.range is None: r = float(grad_input[0].abs().max().cpu().detach())
                else: r = self.range
                hist = grad_input[0].float().histc(self.bins, 0 - r, 0+r)
                learner.log(f'{name} input grad histogram', hist)

            if grad_output[0] is not None:
                if self.range is None: r = float(grad_output[0].abs().max().cpu().detach())
                else: r = self.range
                hist = grad_output[0].float().histc(self.bins, 0 - r, 0+r)
                learner.log(f'{name} output grad histogram', hist)
