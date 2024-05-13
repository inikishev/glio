from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    import torch, torch.utils.hooks
    from . import Logger
    
from ..python_tools import type_str
from .hooks import ForwardHook
from .filters import does_something
class Hook_ActDist(ForwardHook):
    def __init__(self, model: 'torch.nn.Module', logger: 'Logger', filter = does_something, metrics = ['mean', 'std', 'var', 'min', 'max']):
        self.logger = logger
        self.metrics = metrics
        super().__init__(model, filter)
        
    def hook(self, module: 'torch.nn.Module', input: 'torch.Tensor', output: 'torch.Tensor', name:str, ):
        name = f'{name}|{type_str(type(module))}'
        if 'mean' in self.metrics: self.logger.add(f'{name} out_mean', output.mean(), self.model.total_batch)
        if 'std' in self.metrics: self.logger.add(f'{name} out_std', output.std(), self.model.total_batch)
        if 'var' in self.metrics: self.logger.add(f'{name} out_var', output.var(), self.model.total_batch)
        if 'min' in self.metrics: self.logger.add(f'{name} out_min', output.min(), self.model.total_batch)
        if 'max' in self.metrics: self.logger.add(f'{name} out_max', output.max(), self.model.total_batch)

class Hook_ActHist(ForwardHook):
    def __init__(self, model: 'torch.nn.Module', logger: 'Logger', filter = does_something, range = 10, bins = 100):
        self.logger = logger
        self.range = range
        self.bins = bins
        super().__init__(model, filter)
        
    def hook(self, module: 'torch.nn.Module', input: 'torch.Tensor', output: 'torch.Tensor', name:str, ):
        name = name = f'{name}|{type_str(type(module))}'
        hist = output.float().histc(self.bins, 0 - self.range, 0 + self.range)
        self.logger.add(f'{name} HIST_fwd', hist, self.model.total_batch)
        
        hist_center = int(len(hist)/2)
        self.logger.add(f'{name} dead_acts', hist[hist_center - 1 : hist_center + 1].sum() / hist.sum(), self.model.total_batch)