import time
import math
import torch
from ..design.EventModel import CBMethod
from .Learner import Learner
from ..torch_tools import angle, seeded_rng, stepchunk
from ..random import randperm

__all__ = [
    "LogParamDistCB",
    "LogUpdateDistCB",
    "LogGradDistCB",
    "LogGradUpdateAngleCB",
    "LogLastGradsAngleCB",
    "LogLastUpdatesAngleCB",
    "LogParamsPathCB",
    "LogGradPathCB",
    "LogUpdatePathCB",
    
]
class LogParamDistCB(CBMethod):
    def __init__(self, step: int = 1, mean = True, std = False, var = True, min = True, max = True, argmin=False, argmax=False, median = False):#pylint:disable=W0622,W0621
        super().__init__()
        self.step = step
        self.mean, self.std, self.var, self.min, self.max, self.median, self.argmin, self.argmax = mean, std, var, min, max, median, argmin, argmax


    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            self.params = torch.nn.utils.parameters_to_vector(learner.model.parameters()) # type:ignore
            if self.mean: learner.log('param mean', self.params.mean())
            if self.std: learner.log('param std', self.params.std())
            if self.var: learner.log('param var', self.params.var())
            if self.min: learner.log('param min', self.params.min())
            if self.max: learner.log('param max', self.params.max())
            if self.argmin: learner.log('param argmin', self.params.argmin())
            if self.argmax: learner.log('param argmax', self.params.argmax())
            if self.median: learner.log('param median', self.params.median())

    def get_params(self, learner:Learner):
        """This method will become available in Learner so that other callbacks can use it."""
        return self.params


class LogUpdateDistCB(CBMethod):
    def __init__(self, step: int = 1, mean = False, std = False, var = True, min = False, max = False, argmin=False, argmax=False, median = False, angle=True, cosine=False):#pylint:disable=W0622,W0621
        super().__init__()
        self.step = step
        self.mean, self.std, self.var, self.min, self.max, self.median, self.argmin, self.argmax, self.angle, self.cosine = mean, std, var, min, max, median, argmin, argmax, angle, cosine

    def before_train_batch(self, learner: Learner):
        if learner.total_batch % self.step == 0:
            self.params = torch.nn.utils.parameters_to_vector(learner.model.parameters()) # type:ignore

    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            self.params_new = torch.nn.utils.parameters_to_vector(learner.model.parameters()) # type:ignore
            self.update = (self.params - self.params_new).abs()
            if self.mean: learner.log('update mean', (self.update).mean())
            if self.std: learner.log('update std', (self.update).std())
            if self.var: learner.log('update var', (self.update).var())
            if self.min: learner.log('update min', (self.update).min())
            if self.max: learner.log('update max', (self.update).max())
            if self.argmin: learner.log('update argmin', (self.update).argmin())
            if self.argmax: learner.log('update argmax', (self.update).argmax())
            if self.median: learner.log('update median', (self.update).median())
            if self.angle: learner.log('update angle', angle(self.params, self.params_new))
            if self.cosine: learner.log('update cosine', torch.nn.functional.cosine_similarity(self.params, self.params_new, dim=0)) # pylint:disable=E1102

    def get_update(self, learner:Learner):
        """This method will become available in Learner so that other callbacks can use it."""
        return self.update

class LogGradDistCB(CBMethod):
    def __init__(self, step: int = 1, mean = False, std = True, var = False, min = False, max = False, argmin=False, argmax=False, median = False, ):#pylint:disable=W0622
        super().__init__()
        self.step = step
        self.mean, self.std, self.var, self.min, self.max, self.median, self.argmin, self.argmax = mean, std, var, min, max, median, argmin, argmax


    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            self.grad = torch.cat([i.grad.ravel() for i in learner.model.parameters() if i.grad is not None], 0) # type:ignore

            if self.mean: learner.log('grad mean', self.grad.mean())
            if self.std: learner.log('grad std', self.grad.std())
            if self.var: learner.log('grad var', self.grad.var())
            if self.min: learner.log('grad min', self.grad.min())
            if self.max: learner.log('grad max', self.grad.max())
            if self.argmin: learner.log('grad argmin', self.grad.argmin())
            if self.argmax: learner.log('grad argmax', self.grad.argmax())
            if self.median: learner.log('grad median', self.grad.median())

    def get_grad(self, learner:Learner):
        """This method will become available in Learner so that other callbacks can use it."""
        return self.grad

class LogGradUpdateAngleCB(CBMethod):
    order = 10
    def __init__(self, step, mean = True, std = False, var = False, min = False, max = False, argmin=False, argmax=False, median = False, angle = True, cosine = False): #pylint:disable=W0621,W0622
        super().__init__()
        self.step = step
        self.mean, self.std, self.var, self.min, self.max, self.median, self.argmin, self.argmax, self.angle, self.cosine = mean, std, var, min, max, median, argmin, argmax, angle, cosine

    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            update:torch.Tensor = learner.get_update()[0]
            grad:torch.Tensor = learner.get_grad()[0]
            difference = (update - grad).abs()
            if self.mean: learner.log('grad-update mean', (difference).mean())
            if self.std: learner.log('grad-update std', (difference).std())
            if self.var: learner.log('grad-update var', (difference).var())
            if self.min: learner.log('grad-update min', (difference).min())
            if self.max: learner.log('grad-update max', (difference).max())
            if self.argmin: learner.log('grad-update argmin', (difference).argmin())
            if self.argmax: learner.log('grad-update argmax', (difference).argmax())
            if self.median: learner.log('grad-update median', (difference).median())
            if self.angle: learner.log('grad-update angle', angle(update, grad))
            if self.cosine: learner.log('grad-update cosine', torch.nn.functional.cosine_similarity(update, grad, dim=0)) # pylint:disable=E1102

class LogLastGradsAngleCB(CBMethod):
    def __init__(self, step, mean = True, std = False, var = False, min = False, max = False, argmin=False, argmax=False, median = False, angle = False, cosine = False): #pylint:disable=W0621,W0622
        super().__init__()
        self.step = step
        self.mean, self.std, self.var, self.min, self.max, self.median, self.argmin, self.argmax, self.angle, self.cosine = mean, std, var, min, max, median, argmin, argmax, angle, cosine

    def before_train_batch(self, learner:Learner):
        if learner.total_batch < 1: return
        self.prev_grad:torch.Tensor = torch.cat([i.grad.ravel() for i in learner.model.parameters() if i.grad is not None], 0) # type:ignore

    def after_train_batch(self, learner:Learner):
        if learner.total_batch < 1: return
        if learner.total_batch % self.step == 0:
            grad:torch.Tensor = torch.cat([i.grad.ravel() for i in learner.model.parameters() if i.grad is not None], 0) # type:ignore
            difference = (self.prev_grad - grad).abs()
            if self.mean: learner.log('last grads mean', (difference).mean())
            if self.std: learner.log('last grads std', (difference).std())
            if self.var: learner.log('last grads var', (difference).var())
            if self.min: learner.log('last grads min', (difference).min())
            if self.max: learner.log('last grads max', (difference).max())
            if self.argmin: learner.log('last grads argmin', (difference).argmin())
            if self.argmax: learner.log('last grads argmax', (difference).argmax())
            if self.median: learner.log('last grads median', (difference).median())
            if self.angle: learner.log('last grads angle', angle(self.prev_grad, grad))
            if self.cosine: learner.log('last grads cosine', torch.nn.functional.cosine_similarity(self.prev_grad, grad, dim=0)) # pylint:disable=E1102

class LogLastUpdatesAngleCB(CBMethod):
    order = 10
    def __init__(self, step, mean = False, std = True, var = False, min = False, max = False, argmin=False, argmax=False, median = False, angle = False, cosine = False): #pylint:disable=W0621,W0622
        super().__init__()
        self.step = step
        self.mean, self.std, self.var, self.min, self.max, self.median, self.argmin, self.argmax, self.angle, self.cosine = mean, std, var, min, max, median, argmin, argmax, angle, cosine
        self.prev_update = None # type:ignore

    def before_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            try:
                self.prev_update:torch.Tensor = learner.get_update()[0]
            except AttributeError:
                self.prev_update = None # type:ignore

    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0 and self.prev_update is not None:
            new_update:torch.Tensor = learner.get_update()[0]
            difference = (self.prev_update - new_update).abs()
            if self.mean: learner.log('last updates mean', (difference).mean())
            if self.std: learner.log('last updates std', (difference).std())
            if self.var: learner.log('last updates var', (difference).var())
            if self.min: learner.log('last updates min', (difference).min())
            if self.max: learner.log('last updates max', (difference).max())
            if self.argmin: learner.log('last updates argmin', (difference).argmin())
            if self.argmax: learner.log('last updates argmax', (difference).argmax())
            if self.median: learner.log('last updates median', (difference).median())
            if self.angle: learner.log('last updates angle', angle(self.prev_update, new_update))
            if self.cosine: learner.log('last updates cosine', torch.nn.functional.cosine_similarity(self.prev_update, new_update, dim=0)) # pylint:disable=E1102

class LogParamsPathCB(CBMethod):
    #order = 10
    def __init__(self, step, ngroups = 10, mean=True, l1=False, l2=False, median=False, maxparams=100_000, mode='rand', det = True):
        """Mode: `rand` / `step` / `chunk` / `chunkstep`."""
        super().__init__()
        self.step = step
        self.ngroups = ngroups
        self.mean, self.l1, self.l2, self.median = mean, l1, l2, median
        if maxparams is not None: maxparams = int(maxparams // ngroups)
        self.maxparams = maxparams
        self.mode = mode.lower()

        self.group_idxs = None
        self.det = det

    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            params = torch.nn.utils.parameters_to_vector(learner.model.parameters()) # type:ignore
            #params = learner.get_params()[0]

            if self.mode == 'rand':
                if self.group_idxs is None:
                    if self.det: self.group_idxs = randperm(params.shape[0], device=learner.device, seed=0).chunk(self.ngroups)
                    else: self.group_idxs = torch.randperm(params.shape[0], device=learner.device).chunk(self.ngroups)
                    if self.maxparams is not None: self.group_idxs = [i[:self.maxparams] for i in self.group_idxs]
                param_groups = [params[g] for g in self.group_idxs] # type:ignore
            elif self.mode == 'step': param_groups = stepchunk(params, self.ngroups, self.maxparams)
            elif self.mode == 'chunk': param_groups = [i[:self.maxparams] for i in params.chunk(self.ngroups)]
            elif self.mode == 'chunkstep':
                if self.maxparams is None: step=1
                else: step = int(math.ceil(len(params) // self.maxparams))
                param_groups = [i[::step] for i in params.chunk(self.ngroups)]
            else: raise ValueError(f'Unknown mode {self.mode}')

            if self.mean: learner.log('param path mean', [g.mean().detach().cpu() for g in param_groups])
            if self.l1: learner.log('param path L1', [torch.linalg.vector_norm(g, ord=1).detach().cpu() for g in param_groups]) # pylint:disable=E1102
            if self.l2: learner.log('param path L2', [torch.linalg.vector_norm(g, ord=2).detach().cpu() for g in param_groups]) # pylint:disable=E1102
            if self.median: learner.log('param path median', [g.median().detach().cpu() for g in param_groups]) # type:ignore

class LogGradPathCB(CBMethod):
    def __init__(self, step, ngroups = 2, mean=True, l1=False, l2=False, median=False, maxparams=100_000,  det = True):
        super().__init__()
        self.step = step
        self.ngroups = ngroups
        self.mean, self.l1, self.l2, self.median = mean, l1, l2, median
        if maxparams is not None: maxparams = int(maxparams // ngroups)
        self.maxparams = maxparams
        self.group_idxs = None
        self.det = det

    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            grad:torch.Tensor = torch.cat([i.grad.ravel() for i in learner.model.parameters() if i.grad is not None], 0) # type:ignore
            if self.group_idxs is None:
                if self.det: self.group_idxs = randperm(grad.shape[0], device=learner.device, seed=0).chunk(self.ngroups)
                else: self.group_idxs = torch.randperm(grad.shape[0], device=learner.device).chunk(self.ngroups)
                if self.maxparams is not None: self.group_idxs = [i[:self.maxparams] for i in self.group_idxs]
            grad_groups = [grad[g] for g in self.group_idxs]

            if self.mean: learner.log('grad path mean', [g.mean().detach().cpu() for g in grad_groups])
            if self.l1: learner.log('grad path L1', [torch.linalg.vector_norm(g, ord=1).detach().cpu() for g in grad_groups]) # pylint:disable=E1102
            if self.l2: learner.log('grad path L2', [torch.linalg.vector_norm(g, ord=2).detach().cpu() for g in grad_groups]) # pylint:disable=E1102
            if self.median: learner.log('grad path median', [g.median().detach().cpu() for g in grad_groups])

class LogUpdatePathCB(CBMethod):
    order = 10
    def __init__(self, step, ngroups = 2, mean=True, l1=False, l2=False, median=False, maxparams=100_000,  det=True):
        super().__init__()
        self.step = step
        self.ngroups = ngroups
        self.mean, self.l1, self.l2, self.median = mean, l1, l2, median
        if maxparams is not None: maxparams = int(maxparams // ngroups)
        self.maxparams = maxparams
        self.group_idxs = None
        self.det = det

    def after_train_batch(self, learner:Learner):
        if learner.total_batch % self.step == 0:
            update:torch.Tensor = learner.get_update()[0]
            if self.group_idxs is None:
                if self.det: self.group_idxs = randperm(update.shape[0], device=learner.device, seed=0).chunk(self.ngroups)
                else: self.group_idxs = torch.randperm(update.shape[0], device=learner.device).chunk(self.ngroups)
                if self.maxparams is not None: self.group_idxs = [i[:self.maxparams] for i in self.group_idxs]
            update_groups = [update[g] for g in self.group_idxs]

            if self.mean: learner.log('update path mean', [g.mean().detach().cpu() for g in update_groups])
            if self.l1: learner.log('update path L1', [torch.linalg.vector_norm(g, ord=1).detach().cpu() for g in update_groups]) # pylint:disable=E1102
            if self.l2: learner.log('update path L2', [torch.linalg.vector_norm(g, ord=2).detach().cpu() for g in update_groups]) # pylint:disable=E1102
            if self.median: learner.log('update path median', [g.median().detach().cpu() for g in update_groups])

