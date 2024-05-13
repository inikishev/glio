"""Абстрактные классы для хуков"""
from typing import Callable, Optional, final
from abc import ABC, abstractmethod
import functools
import torch, torch.utils.hooks
from ..design.EventModel import Callback
from .Learner import Learner
from ..python_tools import type_str

class ForwardHook(ABC):
    """`_hook` must be a callable with the following arguments
    ```python
    _hook(module: torch.nn.Module, input: tuple[torch.Tensor], output: torch.Tensor)
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def __call__(self, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor): ...
    @final
    def enter(self, learner: "Learner"):
        for _, mod in learner.model.named_modules(remove_duplicate=False):
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_forward_hook(self)) # type: ignore
    @final
    def exit(self, learner: "Learner"):
        for handle in self.handles: handle.remove()
        self.handles = []



class LearnerForwardHook(ABC):
    """`_hook` must be a callable with the following arguments
    ```python
    _hook(learner: Learner, name: str, module: torch.nn.Module, input: tuple[torch.Tensor], output: torch.Tensor)
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor): ...
    @final
    def enter(self, learner: "Learner"):
        for idx, (name, mod) in enumerate(learner.model.named_modules(remove_duplicate=False)):
            full_name = f"{idx}/{name}/{type_str(mod)}"
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_forward_hook(functools.partial(self, learner, full_name))) # type: ignore
    @final
    def exit(self, learner: "Learner"):
        for handle in self.handles: handle.remove()
        self.handles = []


class BackwardHook(ABC):
    """`_hook` must be a callable with the following arguments
    ```python
    _hook(module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor])
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def __call__(self, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]): ...
    @final
    def enter(self, learner: "Learner"):
        for _, mod in learner.model.named_modules(remove_duplicate=False):
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_full_backward_hook(self)) # type: ignore
    @final
    def exit(self, learner: "Learner"):
        for handle in self.handles: handle.remove()
        self.handles = []

class LearnerBackwardHook(ABC):
    """`_hook` must be a callable with the following arguments
    ```python
    _hook(learner: Learner, name: str, grad_input: torch.nn.Module, input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor])
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]): ...
    @final
    def enter(self, learner: "Learner"):
        for idx, (name, mod) in enumerate(learner.model.named_modules(remove_duplicate=False)):
            full_name = f"{idx}/{name}/{type_str(mod)}"
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_full_backward_hook(functools.partial(self, learner, full_name))) # type: ignore
    @final
    def exit(self, learner: "Learner"):
        for handle in self.handles: handle.remove()
        self.handles = []
