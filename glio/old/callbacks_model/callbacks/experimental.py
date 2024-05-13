from ..model import Callback
# from .conditions import condition
from .initializers import _Init_LSUV, _needs_init
from .basic import StopFitOnBatch, FitStatus
from functools import partial

class _LSUVTraining:
    def __init__(self, allowance = 0.01, max_batches = 10000, print = True):
        self.allowance = allowance
        self.max_batches = max_batches
        self.print = print
        self.handles = []
        
    def _register(self, m):
        if _needs_init(m): 
            self.handles.append(m.register_forward_hook(partial(_Init_LSUV, self.allowance, self.print)))
        
    def enter(self, l):
        l.model.apply(self._register)
        l.fit(dl_train = None, dl_test = l.dl_train, epochs = 1, extra = [FitStatus('LSUV'), StopFitOnBatch(self.max_batches)])
        for h in self.handles: h.remove()
        self.handles = []
        
class LSUVTraining(Callback):
    def __init__(self, step_batch, step_epoch = None, allowance = 0.01, max_batches = 10000, print = True):
        self.LSUV = _LSUVTraining(allowance, max_batches, print) 
        self.b = step_batch
        self.e = step_epoch
        
    def _lsuv(self, l):
        with l.without('LSUVTraining'):
            self.LSUV.enter(l)
        
    def before_batch(self, l):
        if self.b and l.is_training and l.cur_batch % self.b == 0: 
            backups = l.batch, l.train, l.dl, l.dl_train, l.dl_test, l.epoch, l.n_epochs, l.cur_batch, l.cur_epoch
            self._lsuv(l)
            l.batch, l.train, l.dl, l.dl_train, l.dl_test, l.epoch, l.n_epochs, l.cur_batch, l.cur_epoch = backups

    def before_epoch(self, l):
        if self.e and l.is_training and l.cur_epoch % self.e == 0: 
            backups = l.batch, l.train, l.dl, l.dl_train, l.dl_test, l.epoch, l.n_epochs, l.cur_batch, l.cur_epoch
            self._lsuv(l)
            l.batch, l.train, l.dl, l.dl_train, l.dl_test, l.epoch, l.n_epochs, l.cur_batch, l.cur_epoch = backups