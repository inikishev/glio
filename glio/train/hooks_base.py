"""left hook, right hook"""
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

class LearnerRegisterForwardHook(ABC):
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
    def _register(self, learner: Learner):
        for idx, (name, mod) in enumerate(learner.model.named_modules(remove_duplicate=False)):
            full_name = f"{idx}/{name}/{type_str(mod)}"
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_forward_hook(functools.partial(self, learner, full_name))) # type: ignore

    @final
    def _unregister(self):
        for handle in self.handles: handle.remove()
        self.handles = []


    @final
    def exit(self, learner: "Learner"): self._unregister()



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
    _hook(learner: Learner, name: str, grad_input: torch.nn.Module, grad_output: tuple[torch.Tensor])
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

class LearnerRegisterBackwardHook(ABC):
    """`_hook` must be a callable with the following arguments
    ```python
    _hook(learner: Learner, name: str, grad_input: torch.nn.Module, grad_output: tuple[torch.Tensor])
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]): ...

    @final
    def _register(self, learner: "Learner"):
        for idx, (name, mod) in enumerate(learner.model.named_modules(remove_duplicate=False)):
            full_name = f"{idx}/{name}/{type_str(mod)}"
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_full_backward_hook(functools.partial(self, learner, full_name))) # type: ignore
    @final
    def _unregister(self):
        for handle in self.handles: handle.remove()
        self.handles = []

    def exist(self, learner: "Learner"): self._unregister()


class LearnerTensorBackwardHook(ABC):
    """Registers the backward hook on output tensors during forward pass.
    `_hook` must be a callable with the following arguments
    ```python
    _hook(learner: Learner, name: str, grad_output: tuple[torch.Tensor])
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_output: torch.Tensor): ...

    @abstractmethod
    def _forward_hook(self, learner:Learner, name: str, module: torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
        output.register_hook(functools.partial(self, learner, name, module))

    @final
    def enter(self, learner: "Learner"):
        for idx, (name, mod) in enumerate(learner.model.named_modules(remove_duplicate=False)):
            full_name = f"{idx}/{name}/{type_str(mod)}"
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_forward_hook(functools.partial(self._forward_hook, learner, full_name))) # type: ignore
    @final
    def exit(self, learner: "Learner"):
        for handle in self.handles: handle.remove()
        self.handles = []

class LearnerRegisterTensorBackwardHook(ABC):
    """Registers the backward hook on output tensors during forward pass.
    `_hook` must be a callable with the following arguments
    ```python
    _hook(learner: Learner, name: str, grad_output: tuple[torch.Tensor])
    ```"""
    def __init__(self, filt: Optional[Callable]):
        self.filt = filt
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.output_handles: list[torch.utils.hooks.RemovableHandle]  = []

    @abstractmethod
    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_output: torch.Tensor): ...

    @abstractmethod
    def _forward_hook(self, learner:Learner, name: str, module: torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
        self.output_handles.append(output.register_hook(functools.partial(self, learner, name, module)))

    @final
    def _register(self, learner: "Learner"):
        for idx, (name, mod) in enumerate(learner.model.named_modules(remove_duplicate=False)):
            full_name = f"{idx}/{name}/{type_str(mod)}"
            if self.filt is None or self.filt(mod):
                self.handles.append(mod.register_forward_hook(functools.partial(self._forward_hook, learner, full_name))) # type: ignore
    @final
    def _unregister(self):
        for handle in self.handles: handle.remove()
        self.handles = []
        for handle in self.output_handles: handle.remove()
        self.output_handles = []

    @final
    def exit(self, learner: "Learner"): self._unregister()