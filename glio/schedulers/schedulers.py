from collections.abc import Callable
from abc import ABC, abstractmethod
from typing import Any

from ..torch_tools import get_lr, set_lr_, copy_state_dict

class _DummyOptimizer:
    def __init__(self):
        self.param_groups = [{'lr':0}]
    def state_dict(self): return self.param_groups[0]
    def load_state_dict(self, state_dict:dict[str, Any]): self.lr = state_dict['lr']

__all__ = [
    "LRScheduler",
    "BatchLambdaLR",
    "LRLambdaLR",
    "SawLR",
]
class LRScheduler:
    optimizer: Any
    @abstractmethod
    def step(self): ...
    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state:dict[str, Any]):
        for k,v in state.items():
            setattr(self, k, v)
    def plot(self, steps=1000):
        from ..plot import qlinechart
        backup = copy_state_dict(self.optimizer.state_dict())
        lrs = []
        for i in range(steps):
            self.step()
            lrs.append(get_lr(self.optimizer))
        qlinechart(lrs, title=f"{self.__class__.__name__} LR over {steps} steps")
        self.optimizer.load_state_dict(backup)

class BatchLambdaLR(LRScheduler):
    """Get LR as a function of current batch index."""
    def __init__(self, optimizer, fn:Callable[[int], float]):
        self.optimizer = optimizer
        self.fn = fn
        self.batch = 0

    def step(self):
        set_lr_(self.optimizer, self.fn(self.batch))
        self.batch += 1

    def state_dict(self):
        return dict(fn = self.fn, batch=self.batch)

class LRLambdaLR(LRScheduler):
    """Get LR as a function of current LR."""
    def __init__(self, optimizer, fn:Callable[..., float]):
        self.optimizer = optimizer
        self.fn = fn

    def step(self):
        set_lr_(self.optimizer, self.fn)

    def state_dict(self):
        return dict(fn = self.fn)

class SawLR(LRScheduler):
    """Looks like this: /|/|/|/|, peaks go from `lrmin` to `lrmax`, and each peak is `length` wide.

    Args:
        optimizer (_type_): _description_
        lrmin (_type_): _description_
        lrmax (_type_): _description_
        length (_type_): _description_
    """
    def __init__(self, optimizer, lrmin, lrmax, length):

        self.optimizer = optimizer
        self.lrmin = lrmin
        self.lrmax = lrmax
        self.length = length
        self.batch = 0

    def step(self):
        set_lr_(self.optimizer, lr = ((self.batch % self.length) / self.length) * (self.lrmax - self.lrmin) + self.lrmin)
        self.batch += 1

    def state_dict(self):
        return dict(lrmin = self.lrmin, lrmax = self.lrmax, length = self.length, batch=self.batch)