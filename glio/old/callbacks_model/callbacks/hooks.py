from ..model import Callback
from functools import partial
# from .conditions import condition, Cond_Step, Cond_Times, Cond_Always, Cond_Metric


class _HookStepper:
    """Calls hook every x steps"""
    def __init__(self, hook, step, times, *args, **kwargs):
        if isinstance(hook, type): self.hook = hook(*args, **kwargs)
        else: self.hook = hook
        self.step = step
        self.times = times

        if self.step is None and self.times is None: self.step = 1

    def __call__(self, l, mod_name, mod, inp, outp):
        if self.step is None: self.step = max(int((len(l.dl_train)*l.n_epochs) / self.times) , 1)
        if l.cur_batch % self.step == 0: self.hook(l, mod_name, mod, inp, outp)


def _type_str(obj): return str(type(obj)).replace("<class '", "").replace("'>", "")
def _mod_name(module, i): return f'{i}: {module[0]} - {_type_str(module[1])}'

class ForwardHook(Callback):
    def __init__(self, hook, step,times, mod_filt, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.hook = hook
        self.step = step
        self.times = times
        self.mod_filt = mod_filt

    def enter(self, l):
        self.modules = [i for i in l.model.named_modules()
                if (self.mod_filt(i) if self.mod_filt else True)]
        self.handles = [None]*len(self.modules)

    def _register_hooks(self, l):
        for i, module in enumerate(self.modules):
            self.handles[i] = module[1].register_forward_hook(
                partial(_HookStepper(self.hook, self.step, self.times, *self.args, **self.kwargs),
                                  l, _mod_name(module, i)))
    def _remove_hooks(self, l):
        for i in self.handles: i.remove()

class ForwardHook_Train(ForwardHook):
    def before_epoch(self, l):
        if l.is_training: self._register_hooks(l)
    def after_epoch(self, l):
        if l.is_training: self._remove_hooks(l)

class ForwardHook_TrainAny(ForwardHook):
    def before_epoch(self, l):
        if l.train: self._register_hooks(l)
    def after_epoch(self, l):
        if l.train: self._remove_hooks(l)

class Filt_ModuleBlacklist:
    def __init__(self, blacklist): self.blacklist = blacklist
    def __call__(self, module):
        if any([i.lower() in _type_str(module[1]).lower() for i in self.blacklist]): return False
        if module[1] == '': return False
        return True

default_blacklist = Filt_ModuleBlacklist(['container', 'flatten', '__main__.', 'HistogramLayer', 'ResCallback'])

# _________________________________________________________________________________
# activations mean std
def _Log_ActMeanStd(l, mod_name, mod, inp, outp):
    l.log(f'{mod_name} act mean', outp.mean())
    l.log(f'{mod_name} act std', outp.std())

class Log_ActMeanStd(ForwardHook_Train):
    def __init__(self, times = 128, step = None, mod_filt = default_blacklist):
        super().__init__(_Log_ActMeanStd, step,times, mod_filt)


# activations histogram and dead activations
class _Log_ActHist:
    def __init__(self, range, bins, top_dead, hist):
        self.range = range
        self.bins = bins
        self.top_dead = top_dead
        self.hist = hist
    def __call__(self, l, mod_name, mod, inp, outp):
        hist = outp.float().histc(self.bins, 0 - self.range, 0+self.range)
        if self.hist: l.log(f'{mod_name} act hist', hist)
        zero = int(len(hist)/2)
        if self.top_dead is not None: l.log(f'{mod_name} dead acts', hist[zero-self.top_dead:zero+self.top_dead].sum()/hist.sum())

class Log_ActHist(ForwardHook_Train):
    def __init__(self, times = 128, step = None, mod_filt = default_blacklist, range = 10, bins = 80, top_dead = 1, hist = True):
        super().__init__(_Log_ActHist, step, times, mod_filt)
        self.args = (range, bins, top_dead, hist)

from .. import visualize
import math
import torch

class _PlotPath:
    def __init__(self, grid):
        self.grid = grid
        self.vis = visualize.Visualizer()

    def add(self, l, mod_name: str, mod: torch.nn.Module, inp: torch.Tensor, outp: torch.Tensor):
        # shorten the name
        mod_name1 = mod_name.split(' - ')[0][3:]
        mod_name2 = mod_name.split('.')[-1]
        mod_name = f'{mod_name1}\n{mod_name2}'

        self.is_3D = False
        if len(self.vis) == 0:
            inpx = inp[0] if isinstance(inp, (list,tuple)) else inp
            inp_shape = inpx.shape
            if inpx.ndim == 5:
                self.is_3D = True
                inpx = inpx[:,:,[int(inpx.shape[2]/2)]].squeeze(2)
            self.vis.imshow(inpx[0], mode='chw', label = f'INPUT\n{tuple(inp_shape)}')

        if hasattr(mod, "named_parameters"):
            for param in mod.named_parameters():
                data = param[1].data
                data_shape = data.shape
                if self.is_3D and data.ndim in (4,5): data = data[:,[int(data.shape[1]/2)]].squeeze(1)

                if data.ndim == 2: self.vis.imshow(data, mode='hw', label = f'{mod_name}\n{param[0]}\n{tuple(data_shape)}')
                # weights are not batched; for example, conv2d can have size (out=16, in=3, kerel=5, 5)
                elif data.ndim == 3:
                    self.vis.imshow_grid(data, mode = 'filters', nelems = self.grid, label = f'{mod_name}\n{param[0]}\n{tuple(data_shape)}')
                elif data.ndim == 4:
                    self.vis.imshow_grid(data, mode = 'filters', nelems = self.grid, label = f'{mod_name}\n{param[0]}\n{tuple(data_shape)}')
                else: ...
        outpx = outp[:,:,[int(outp.shape[2]/2)]].squeeze(2) if outp.ndim == 5 else outp
        self.vis.imshow_grid(outpx[0], mode='bhw', nelems = self.grid, label = f'{mod_name}\nOUTPUT\n{tuple(outp.shape)}')

    def plot(self):
        if len(self.vis) > 0:
            self.vis.show(nrows = int(math.ceil(len(self.vis)) ** 0.5), figsize = (30, 30))
            self.vis.clear()

class PlotPath(ForwardHook_Train):
    def __init__(self, times = 10, step = None,grid = 4, mod_filt = default_blacklist, ):
        self.Plotter = _PlotPath(grid)
        super().__init__(self.Plotter.add, step, times, mod_filt)
        self.kwargs = {'grid': grid}
    def after_batch(self, l):
        if l.is_training:
            if self.step is None: self.step = max(int((len(l.dl_train)*l.n_epochs) / self.times) , 1)
            if l.cur_batch % self.step == 0: self.Plotter.plot()