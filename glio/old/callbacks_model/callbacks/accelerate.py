from ..model import Callback
import torch
def _to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device)
    else: return [i.to(device) for i in x]
    
    
class Accelerate(Callback):
    order = 90
    description = 'Huggingface Accelerate'
    def __init__(self, precision = 'fp16'): 
        from accelerate import Accelerator
        self._Accelerator = Accelerator
        self.precision = precision
        self.opt = None
        self.scheduler = None
        self.dl_train = None
        self.dl_test = None
        
    def enter(self, l):
        l.acc = self._Accelerator(mixed_precision = self.precision)
        l.device = l.acc.device
        l.model = l.model.to(l.device)
        l.model = l.acc.prepare(l.model)
        
    def before_batch(self, l):
        l.batch = _to_device(l.batch, l.device)

    def before_fit(self, l):
        if self.scheduler != l.scheduler: 
            l.scheduler = l.acc.prepare(l.scheduler)
            self.scheduler = l.scheduler
        if self.opt != l.opt: 
            l.opt = l.acc.prepare(l.opt)
            self.opt = l.opt
        if self.dl_train != l.dl_train: 
            l.dl_train = l.acc.prepare(l.dl_train)
            self.dl_train = l.dl_train
        if self.dl_test != l.dl_test: 
            l.dl_test = l.acc.prepare(l.dl_test)
            self.dl_test = l.dl_test
        
    def backward(self, l): l.acc.backward(l.loss)
