"Stuff that uses `register_forward_hook` and `register_backward_hook`"
from collections.abc import Sequence
import os
import functools
import math

import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

from .Learner import Learner
from ..design.event_model import BasicCallback, MethodCallback
from . hooks_base import LearnerForwardHook, LearnerBackwardHook, LearnerRegisterTensorBackwardHook, LearnerRegisterForwardHook
from ..torch_tools import is_container, is_inplace, ensure_device, is_inplace_recursive, copy_state_dict
from ..transforms.intensity import norm
from ..python_tools import to_valid_fname
from ..loaders.image import imwrite

__all__ = [
    "LogLayerSignalDistributionCB",
    "LogLayerSignalHistorgramCB",
    "LogLayerGradDistributionCB",
    "LogLayerGradHistorgramCB",
    'SaveForwardChannelImagesCB',
    'SaveBackwardChannelImagesCB',
    'SaveUpdateChannelImagesCB',
]


class LogLayerSignalDistributionCB(LearnerForwardHook, BasicCallback):
    def __init__(self, step: int = 1, mean = True, std = True, var = True, min = True, max = True, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.mean, self.std, self.var, self.min, self.max = mean, std, var, min, max
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            if self.mean: learner.log(f'{name} output mean', outputs.mean())
            if self.std: learner.log(f'{name} output std', outputs.std())
            if self.var: learner.log(f'{name} output var', outputs.var())
            if self.min: learner.log(f'{name} output min', outputs.min())
            if self.max: learner.log(f'{name} output max', outputs.max())

class LogLayerSignalHistorgramCB(LearnerForwardHook, BasicCallback):
    def __init__(self, step: int = 1, range = 10, bins = 60, top_dead = 1, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.range, self.bins, self.top_dead = range, bins, top_dead
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            hist = outputs.float().histc(self.bins, 0 - self.range, 0+self.range)
            learner.log(f'{name} output histogram', hist)
            zero = int(len(hist)/2)
            if self.top_dead is not None: learner.log(f'{name} dead activations', hist[zero-self.top_dead:zero+self.top_dead].sum()/hist.sum())

class LogLayerGradDistributionCB(LearnerBackwardHook, BasicCallback):
    def __init__(self, step: int = 1, mean = True, std = True, var = True, min = True, max = True, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.mean, self.std, self.var, self.min, self.max = mean, std, var, min, max
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            if grad_input[0] is not None:
                if self.mean: learner.log(f'{name} input grad mean', grad_input[0].mean())
                if self.std: learner.log(f'{name} input grad std', grad_input[0].std())
                if self.var: learner.log(f'{name} input grad var', grad_input[0].var())
                if self.min: learner.log(f'{name} input grad min', grad_input[0].min())
                if self.max: learner.log(f'{name} input grad max', grad_input[0].max())

            if grad_output[0] is not None:
                if self.mean: learner.log(f'{name} output grad mean', grad_output[0].mean())
                if self.std: learner.log(f'{name} output grad std', grad_output[0].std())
                if self.var: learner.log(f'{name} output grad var', grad_output[0].var())
                if self.min: learner.log(f'{name} output grad min', grad_output[0].min())
                if self.max: learner.log(f'{name} output grad max', grad_output[0].max())

class LogLayerGradHistorgramCB(LearnerBackwardHook, BasicCallback):
    def __init__(self, step: int = 1, range = None, bins = 60, filt = lambda x: not is_container(x)): # pylint: disable=W0622
        self.step = step

        self.range, self.bins = range, bins
        super().__init__(filt=filt)

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]):
        if learner.status == "train" and learner.total_batch % self.step == 0:
            if len(grad_input) > 1: print(f"`grad_input` is a tuple with length of more than 1!!! {name = } gradient grad_input {len(grad_input) = } tensors")
            if len(grad_output) > 1: print(f"`grad_output` is a tuple with length of more than 1!!! {name = } gradient grad_output {len(grad_input) = } tensors")
            if grad_input[0] is not None:
                if self.range is None: r = float(grad_input[0].abs().max().cpu().detach())
                else: r = self.range
                hist = grad_input[0].float().histc(self.bins, 0 - r, 0+r)
                learner.log(f'{name} input grad histogram', hist)

            if grad_output[0] is not None:
                if self.range is None: r = float(grad_output[0].abs().max().cpu().detach())
                else: r = self.range
                hist = grad_output[0].float().histc(self.bins, 0 - r, 0+r)
                learner.log(f'{name} output grad histogram', hist)


def _save_3D_slices_separately(x:torch.Tensor, dir:str, fname:str, mkdir=True, max_numel = 1024 * 1024, max_ch = 32):
    x = x[0]
    if x.ndim not in (2, 3): return
    if x.ndim == 2:
        if x.numel() > max_numel: return
        if mkdir and not os.path.exists(dir): os.mkdir(dir)
        imwrite(x, outfile=os.path.join(dir, f'{fname}2D.jpg'), normalize=True, optimize=True, compression=9)

    elif x.ndim == 3:
        if x[0].numel() > max_numel: return
        if mkdir and not os.path.exists(dir): os.mkdir(dir)
        numpyx: np.ndarray = x.detach().cpu().numpy()
        for i,sl in enumerate(numpyx[:max_ch]):
            imwrite(sl, outfile=os.path.join(dir, f'{fname}{i}.jpg'), normalize=True, optimize=True, compression=9)

def _save_3D_slices_grid(x:torch.Tensor, dir:str, fname:str, mkdir=True, max_numel = 1024 * 1024, max_ch = 32):
    x = x[0]
    if x.ndim not in (2, 3): return
    if x.ndim == 2:
        if x.numel() > max_numel: return
        if mkdir and not os.path.exists(dir): os.mkdir(dir)
        imwrite(x, outfile=os.path.join(dir, f'{fname}.png'), normalize=True, optimize=True, compression=9)

    elif x.ndim == 3:
        if x[0].numel() > max_numel: return
        if mkdir and not os.path.exists(dir): os.mkdir(dir)
        x_bchw = x[:max_ch].unsqueeze(1)
        grid = make_grid(x_bchw, nrow=max(1, int(math.ceil(len(x_bchw)**0.5))), padding=2, normalize=True, scale_each=True, pad_value=0.5)
        #print(grid.shape)
        imwrite(grid[0].detach().cpu().numpy(), os.path.join(dir, f'{fname}.png'), normalize=True, optimize=True, compression=9)
        # numpyx: np.ndarray = x.detach().cpu().numpy()
        # for i,sl in enumerate(numpyx[:max_ch]):
        #     imwrite(sl, outfile=os.path.join(dir, f'{prefix}{i}.jpg'), normalize=True, optimize=True, compression=9)

_SLICE_SAVING_MODES = {
    'separate': _save_3D_slices_separately,
    'grid': _save_3D_slices_grid,
}
class SaveForwardChannelImagesSToseparateFoldersCB(LearnerRegisterForwardHook, MethodCallback):
    def __init__(self, inputs, dir = 'forward channels', mkdir = True, filt = lambda x: not is_container(x),):
        MethodCallback.__init__(self)
        LearnerRegisterForwardHook.__init__(self, filt=filt)

        self.inputs = inputs
        self.dir = dir
        self.mkdir = mkdir
        self.workdir = ''
        self.cur_dir = ''
        self.cur_iter = 1
        self.cur_module = 1

        self.saved_modules = set()

    def after_test_epoch(self, learner:Learner):
        status = learner.status
        self._register(learner)
        learner.inference(self.inputs, to_cpu = False, status="SaveSignalImagesCB")
        self._unregister()
        learner.status = status

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if learner.status == 'SaveSignalImagesCB':
            if name in self.saved_modules:
                self.cur_iter += 1
                self.cur_module = 1
                self.saved_modules = set()

            if self.cur_iter == 1: self.workdir = learner.get_workdir(self.dir, self.mkdir)
            if self.cur_module == 1:
                self.cur_dir = os.path.join(self.workdir, f'{self.cur_iter} - {learner.total_epoch} {learner.total_batch}')
                if not os.path.exists(self.cur_dir): os.mkdir(self.cur_dir)
                # save inputs
                _save_3D_slices_separately(inputs[0], os.path.join(self.cur_dir, '0 - inputs'), fname='')

            _save_3D_slices_separately(outputs, os.path.join(self.cur_dir, f'{to_valid_fname(name)} {tuple(outputs.shape)}'), fname='')
            self.saved_modules.add(name)
            self.cur_module += 1


def is_not_container(x): return not is_container(x)
class SaveForwardChannelImagesCB(LearnerRegisterForwardHook, MethodCallback):
    def __init__(self, inputs:torch.Tensor, dir = 'runs', mkdir = True, max_ch = 42, filt = is_not_container, mode='grid'):
        MethodCallback.__init__(self)
        LearnerRegisterForwardHook.__init__(self, filt=filt)

        self.inputs = inputs
        self.dir = dir
        self.mkdir = mkdir
        self.max_ch = max_ch

        self.workdir = ''
        self.cur_prefix = ''
        self.cur_iter = 1
        self.cur_module = 1

        self.save_3d_slices_fn = _SLICE_SAVING_MODES[mode]

        self.saved_modules = set()

    def after_test_epoch(self, learner:Learner):
        status = learner.status
        self._register(learner)
        learner.inference(self.inputs, to_cpu = False)
        self._unregister()
        learner.status = status

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if name in self.saved_modules:
            self.cur_iter += 1
            self.cur_module = 1
            self.saved_modules = set()

        if self.cur_iter == 1:
            self.workdir = os.path.join(learner.get_workdir(self.dir, self.mkdir), 'forward channels')
            if not os.path.exists(self.workdir): os.mkdir(self.workdir)
        if self.cur_module == 1:
            self.cur_prefix = f'{self.cur_iter} - e{learner.total_epoch} b{learner.total_batch} c'
            # save inputs
            self.save_3d_slices_fn(inputs[0], os.path.join(self.workdir, f'0 - inputs {tuple(inputs[0].shape)}'), fname=f'{self.cur_prefix}', max_ch=self.max_ch)

        self.save_3d_slices_fn(outputs,
                        dir = os.path.join(self.workdir, f'{to_valid_fname(name)} {tuple(outputs.shape)}'),
                        fname=f'{self.cur_prefix}',
                        max_ch=self.max_ch,
                        )
        self.saved_modules.add(name)
        self.cur_module += 1

class SaveBackwardChannelImagesCB(LearnerRegisterTensorBackwardHook, MethodCallback):
    def __init__(self, inputs:torch.Tensor, targets:torch.Tensor, dir = 'runs', mkdir = True, max_ch = 42, filt = is_not_container, unsqueeze=False, mode='grid'):
        MethodCallback.__init__(self)
        LearnerRegisterTensorBackwardHook.__init__(self, filt=filt)

        self.inputs = inputs.unsqueeze(0) if unsqueeze else inputs[0]
        self.targets = targets.unsqueeze(0) if unsqueeze else targets[0]
        self.dir = dir
        self.mkdir = mkdir
        self.max_ch = max_ch

        self.workdir = ''
        self.cur_prefix = ''
        self.cur_iter = 1
        self.cur_module = 1

        self.save_3d_slices_fn = _SLICE_SAVING_MODES[mode]

        self.saved_modules = set()

    def after_test_epoch(self, learner:Learner):
        status = learner.status
        self._register(learner)
        learner.status = 'SaveBackwardChannelImagesCB'
        with torch.enable_grad():
            preds = learner.inference(self.inputs, to_cpu = False, mode = 'train', status="SaveBackwardChannelImagesCB", grad=True)
            loss = learner.get_loss(preds, ensure_device(self.targets, learner.device))
            learner.backward()
            learner.zero_grad()
        self._unregister()
        learner.status = status

    def _forward_hook(self, learner:Learner, name: str, module: torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
        if learner.status == 'SaveBackwardChannelImagesCB': output.register_hook(functools.partial(self, learner, name, module))

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, grad_output: torch.Tensor):
        if learner.status == 'SaveBackwardChannelImagesCB':
            if name in self.saved_modules:
                self.cur_iter += 1
                self.cur_module = 1
                self.saved_modules = set()

            if self.cur_iter == 1:
                self.workdir = os.path.join(learner.get_workdir(self.dir, self.mkdir), 'backward channels')
                if not os.path.exists(self.workdir): os.mkdir(self.workdir)
            if self.cur_module == 1:
                self.cur_prefix = f'{self.cur_iter} - e{learner.total_epoch} b{learner.total_batch} c'

            if grad_output[0] is not None:
                self.save_3d_slices_fn(grad_output,
                                dir = os.path.join(self.workdir, f'{to_valid_fname(name)} {tuple(grad_output[0].shape)}'),
                                fname=f'{self.cur_prefix}',
                                max_ch=self.max_ch,
                                )
                self.saved_modules.add(name)
                self.cur_module += 1


class SaveUpdateChannelImagesCB(LearnerRegisterForwardHook, MethodCallback):
    def __init__(self, inputs:torch.Tensor, targets:torch.Tensor, dir = 'runs', mkdir = True, max_ch = 42, filt = is_not_container, mode='grid', unsqueeze = False,):
        MethodCallback.__init__(self)
        LearnerRegisterForwardHook.__init__(self, filt=filt)

        self.inputs = inputs.unsqueeze(0) if unsqueeze else inputs[0]
        self.targets = targets.unsqueeze(0) if unsqueeze else targets[0]

        self.dir = dir
        self.mkdir = mkdir
        self.max_ch = max_ch

        self.workdir = ''
        self.cur_prefix = ''
        self.cur_iter = 1
        self.cur_module = 1

        self.save_3d_slices_fn = _SLICE_SAVING_MODES[mode]

        self.status = 1
        self.channels_before = {}

    def _init(self):
        self.cur_module = 1
        self.channels_before = {}
        self.status = 1

    def after_test_epoch(self, learner:Learner):
        with learner.freeze():
            self._init()
            status = learner.status
            self._register(learner)
            learner.one_batch(self.inputs, self.targets, train=True)
            self.cur_module = 1
            self.status = 2
            learner.inference(self.inputs, to_cpu = False)
            self._unregister()
            learner.status = status
            self.cur_iter += 1

    def __call__(self, learner: Learner, name: str, module: torch.nn.Module, inputs: tuple[torch.Tensor], outputs: torch.Tensor):
        if self.status == 1:

            if self.cur_iter == 1:
                self.workdir = os.path.join(learner.get_workdir(self.dir, self.mkdir), 'update channels')
                if not os.path.exists(self.workdir): os.mkdir(self.workdir)

            if self.cur_module == 1:
                self.channels_before[f"0 - inputs {tuple(inputs[0].shape)}"] = inputs[0]
                self.cur_module += 1

            self.channels_before[f'{name} {tuple(outputs.shape)}'] = outputs

        elif self.status == 2:
            if self.cur_module == 1:
                self.cur_prefix = f'{self.cur_iter} - e{learner.total_epoch} b{learner.total_batch} c'
                # save inputs
                self.save_3d_slices_fn((inputs[0] - self.channels_before[f"0 - inputs {tuple(inputs[0].shape)}"]).abs(),
                                       dir = os.path.join(self.workdir, f'0 - inputs {tuple(inputs[0].shape)}'), 
                                       fname=f'{self.cur_prefix}', max_ch=self.max_ch)

            modname = f'{name} {tuple(outputs.shape)}'
            self.save_3d_slices_fn((outputs - self.channels_before[modname]).abs(),
                            dir = os.path.join(self.workdir, to_valid_fname(modname)),
                            fname=f'{self.cur_prefix}',
                            max_ch=self.max_ch,
                            )
            self.cur_module += 1
