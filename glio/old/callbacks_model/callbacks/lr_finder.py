from ..model import Callback, Cancel

class _LRFinderCallback(Callback):
    order = 200
    def __init__(self, dl, lr_mult=1.3, lr_start = 1e-05, lr_end = 1, lr_increase = 3):
        self.dl = dl
        self.lr_mult = lr_mult
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_increase = lr_increase
        self.lrs, self.losses = [],[]
        self.min = float('inf')
    def enter(self, l):
        if self.lr_start is not None:
            for g in l.opt.param_groups: g['lr'] = self.lr_start
        l.fit(self.dl, None, 99999)
        
    def after_batch(self, l):
        last_lr = l.opt.param_groups[0]['lr']
        self.lrs.append(last_lr)
        loss = l.loss.cpu().detach()
        self.losses.append(loss)
        if loss<self.min: self.min = loss
        if self.lr_increase is not None:
            if loss > self.min * self.lr_increase: raise Cancel('fit')
        if self.lr_end is not None:
            if last_lr >= self.lr_end: raise Cancel('fit')
        for g in l.opt.param_groups: g['lr'] *= self.lr_mult

import matplotlib.pyplot as plt
def lr_finder(learner, dl, lr_mult=1.3, lr_start = 1e-05, lr_end = 1, lr_increase = 3):
    callback = _LRFinderCallback(dl, lr_mult,lr_start, lr_end, lr_increase)
    learner.add(callback)
    learner.remove(callback)
    plt.plot(callback.lrs, callback.losses)
    plt.xscale('log')
    plt.show()