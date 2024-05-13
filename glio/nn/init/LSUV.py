"""LSUV"""
from typing import Any
import torch
class _LSUV:
    def __init__(self, tolerance:float, max_iter:int, log:bool):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.log = log

        self.allow_forward = True

        self.modules_iterations = {}
        self.modules_list = []
        self.i = 0

        self.means = {}
        self.vars = {}

    def hook(self, module:torch.nn.Module, inp:tuple[torch.Tensor], outp:torch.Tensor):


        if module not in self.modules_list:
            self.modules_list.append(module)
            self.modules_iterations[self.i] = 0
            self.i += 1

        cur = self.modules_list.index(module)
        if cur == 0: self.allow_forward = True

        if hasattr(module, 'weight') and module.weight is not None and module.weight.data.std() != 0:
            if self.allow_forward:
                mean = outp.mean()
                std = outp.std()

                self.means[cur] = round(float(mean.detach().cpu()), 3)
                self.vars[cur] = round(float(std.detach().cpu()), 3)

                #if (outp.var() - 1).abs() < self.tolerance or self.modules_iterations[cur] > self.max_iter:
                if self.modules_iterations[cur] > self.max_iter:
                    self.allow_forward = True
                else:
                    if hasattr(module, 'bias') and module.bias is not None: module.bias.data -= mean
                    module.weight.data /= std
                    self.allow_forward = False
                    self.modules_iterations[cur] += 1

                    #print(self.modules_iterations, self.means, self.vars)
                    if self.log: print(f'{list(self.means.values())}, {list(self.vars.values())}, {list(self.modules_iterations.values())}', end='          \r')


def LSUV(model: torch.nn.Module, dl, tolerance = 0.02, max_iter = 10, max_batches = 10000, log = True, device: Any = torch.device('cuda')):
    if device is None: device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        hooks = []
        LSUV_hook = _LSUV(tolerance, max_iter, log)

        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.data.std() != 0 and module.weight.data.ndim >= 3:
                hooks.append(module.register_forward_hook(LSUV_hook.hook))

        finished = False

        i = 0
        while True:
            for batch in dl:
                model(batch[0].to(device))
                # print(list(LSUVer.modules_iterations.values()), i, min(list(LSUVer.modules_iterations.values())), max_iter)
                i+= 1
                if min(list(LSUV_hook.modules_iterations.values())) > max_iter or i >= max_batches or max(list(LSUV_hook.vars.values())) < tolerance:
                    finished = True
                    break
            if finished: break

        for h in hooks: h.remove()
    if log: print()
    return model
