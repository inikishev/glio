from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    import torch, torch.utils.hooks

from functools import partial

import contextlib
class Hook:
    """Регистрирует `self.hook` каждый `step_batch` и/или `step_epoch` в модели, и удаляет после одного шага."""
    def __init__(self, model: "torch.nn.Module", filter):
        self.model = model
        self.hooks: list["torch.utils.hooks.RemovableHandle"] = []
        self.filter = filter
        self.has_hooks = False

    def hook(self, module: 'torch.nn.Module', input: 'torch.Tensor', output: 'torch.Tensor', name:str, ):...  
    def apply(self): ...
    def remove(self): 
        for h in self.hooks: h.remove()
        self.hooks: list["torch.utils.hooks.RemovableHandle"] = []
        
    @contextlib.contextmanager
    def context(self, enable = True, b = None, e = None):
        if enable:
            if b is None or self.model.total_batch % b == 0:
                if e is None or self.model.cur_epoch % e == 0:
                    self.has_hooks = True
                    self.apply()
        yield
        if self.has_hooks: 
            self.remove()
            self.has_hooks = False
        
        
class ForwardHook(Hook):
    """Регистрирует `self.hook` каждый `step_batch` и/или `step_epoch` в модели, и удаляет после одного шага. 
    
    `self.hook` должен принимать 4 аргумента: имя модуля, объект модуля, входной тензор, выходной тензор."""
    def apply(self): 
        for m in self.model.named_modules(remove_duplicate=False):
            if self.filter(m[1]):
                self.hooks.append(m[1].register_forward_hook(partial(self.hook, name = m[0]))) # pyright: ignore

class BackwardHook(Hook):
    """Регистрирует `self.hook` каждый `step_batch` и/или `step_epoch` в модели, и удаляет после одного шага. 
    
    `self.hook` должен принимать 2 аргумента: имя параметра, тензор градиента."""
    def apply(self): 
        for p in self.model.named_parameters(remove_duplicate=False):
            if self.filter(p[1]):
                self.hooks.append(p[1].register_hook(partial(self.hook, name = p[0])))