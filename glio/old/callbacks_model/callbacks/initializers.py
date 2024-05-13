from ..model import Callback, Cancel
from functools import partial
import torch
NEED_INIT = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, 
                      torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Linear)

def _needs_init(module):
    if hasattr(module, 'weight') and module.weight is not None and hasattr(module, 'bias') and module.bias is not None: return True
    return False

def _Init_KaimingLeaky(a, m): 
    if _needs_init(m): 
        torch.nn.init.kaiming_normal_(m.weight, a=a)
        
class Init_KaimingLeaky(Callback):
    order = 50
    def __init__(self, leaky = 0):
        self.leaky = leaky
    def enter(self, l): 
        l.model.apply(partial(_Init_KaimingLeaky, self.leaky))
        
def _tofloat(t: torch.Tensor): return float(t.detach().cpu())
from ...python_tools import type_str
def _Init_LSUV(allowance, enable_print, c, mod, inp, outp):
    mean = outp.mean()
    std = outp.std()
    has_bias = True if (hasattr(mod, 'bias') and mod.bias is not None) else False
    has_weight = True if (hasattr(mod, 'weight') and mod.weight is not None) else False
    if not (((has_bias is False) or (allowance - abs(mean) > 0)) and ((has_weight is False) or (abs(std -1 ) < allowance))) and std!= 0:
        # try:
            if has_bias: mod.bias.data -= mean
            if has_weight is not None: mod.weight.data /= std
            mean = _tofloat(mean)
            std = _tofloat(std)
            if enable_print: print(f'\r{c}: {type_str(type(mod))}: mean = {round(mean,5)}, std = {round(std,5)}. {round(allowance-abs(mean), 5)} > 0; {round(abs(std-1), 5)} < {allowance}        ', end='\r')
            raise Cancel('batch')
    elif enable_print: print('|', end = '')
        # except TypeError: pass
    
from .basic import StopFitOnTotalAnyBatches, FitStatus
class Init_LSUV(Callback):
    order = 100
    def __init__(self, dl, allowance = 0.02, max_batches = 10000, print = True):
        self.allowance = allowance
        self.max_batches = max_batches
        self.dl = dl
        self.handles = []
        self.print=print
        self.c = 0
        
    def _register(self, m:torch.nn.Module):
        if _needs_init(m): 
            self.handles.append(m.register_forward_hook(partial(_Init_LSUV, self.allowance, self.print, self.c)))
            self.c+=1
        
    def enter(self, l):
        l.model.apply(self._register)
        l.fit(dl_train = None, dl_test = self.dl, extra = [FitStatus('LSUV'), StopFitOnTotalAnyBatches(self.max_batches)], epochs = 1)
        for h in self.handles: h.remove()
        self.handles = []
        
        
def _Init_Uniform(min, max, m): 
    if _needs_init(m): 
        torch.nn.init.uniform_(m.weight, a=min, b = max)
        
class Init_Uniform(Callback):
    order = 50
    def __init__(self, min = -1, max = 1):
        self.min = min
        self.max = max
    def enter(self, l): 
        l.model.apply(partial(_Init_Uniform, self.min, self.max))