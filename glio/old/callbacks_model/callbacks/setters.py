from ..model import Callback
import torch
from types import FunctionType

class Loss(Callback):
    def __init__(self, loss_fn, *args, **kwargs):
        self.loss_fn = loss_fn
        self.args = args
        self.kwargs = kwargs
    def enter(self, l): 
        if isinstance(self.loss_fn, FunctionType): l.loss_fn = self.loss_fn
        else: l.loss_fn = self.loss_fn(*self.args, **self.kwargs)

class Optimizer(Callback):
    order = 10
    def __init__(self, opt,*args, **kwargs):
        self.opt = opt
        self.args = args
        self.kwargs = kwargs
    def enter(self, l): l.opt = self.opt(l.model.parameters(), *self.args, **self.kwargs)
    
class Scheduler(Callback):
    """Sets model's scheduler and removes it on exit. Scheduler runs as defined in the default Learner.fit which is every epoch by default."""
    order = 20
    def __init__(self, scheduler, *args, **kwargs):
        self.scheduler = scheduler
        self.args = args
        self.kwargs = kwargs
        
    def enter(self, l): l.scheduler = self.scheduler(l.opt, *self.args, **self.kwargs)
    def exit(self, l): l.scheduler = None
        
        
class SchedulerStep(Callback):
    """This runs a scheduler step every step_batch and step_epoch. As this callback handles scheduler steps, it changes scheduler_step on Learner to not do anything."""
    order = 20
    def __init__(self, scheduler, step_batch = 0, step_epoch = 1, *args, **kwargs):
        self.scheduler = scheduler
        self.args = args
        self.kwargs = kwargs
        self.step_batch = step_batch
        self.step_epoch = step_epoch
        
    def enter(self, l): 
        l.scheduler = self.scheduler(l.opt, *self.args, **self.kwargs)
        
    def after_batch(self,l):
        if self.step_batch and l.train and l.cur_batch % self.step_batch == 0: 
            l.scheduler.step()
        
    def after_epoch(self, l):
        if self.step_epoch and l.train and l.cur_epoch % self.step_epoch == 0: 
            l.scheduler.step()
            
    def scheduler_step(self, l): pass

def _to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device)
    else: return [i.to(device) for i in x]
    
class Device(Callback):
    order = -100
    def __init__(self, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.device = device
    def enter(self, l): 
        l.device = self.device
        l.model.to(l.device)
    def before_batch(self, l): 
        l.batch = _to_device(l.batch, l.device)

    
class NoTarget(Callback):
    """
    Callback for when batch is the target, for example for an autoencoder.
    """
    def enter(self, l): l.has_target = False
    def predict(self, l, batch): l.y = l.model(batch)
    
    
class GradientAccumulation(Callback):
    def __init__(self, n = 2):
        self.n = n
    def before_zero_grad(self, l):
        if l.cur_batch % self.n != 0: l.do_zero_grad = False