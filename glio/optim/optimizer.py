from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from typing import Any
import torch
from torch.optim import Optimizer
def clone_params(params: Iterable[torch.Tensor | torch.nn.Parameter]):
    return torch.nn.ParameterList([p.clone(memory_format=torch.contiguous_format) for p in params])

def set_params_(params: Iterable[torch.Tensor | torch.nn.Parameter], values: Iterable[Any]):
    for p, v in zip(params, values):
        p.data.copy_(v)

class ClosureOptimizer(Optimizer,ABC):
    @abstractmethod
    def step(self, closure: Callable) -> int | float | torch.Tensor: ... # type:ignore #pylint:disable=W0222

