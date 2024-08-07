from typing import Any
import numpy as np
import torch
from ._base import Transform


def ensure_tensor(x:Any, device = None, dtype = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor): return x.to(device=device, dtype=dtype)
    else: return torch.as_tensor(x, device=device, dtype=dtype)

class EnsureTensor(Transform):
    def __init__(self, device = None, dtype = None):
        self.device = device
        self.dtype = dtype
    def forward(self, x:Any) -> torch.Tensor:
        return ensure_tensor(x, device=self.device, dtype=self.dtype)

def ensure_dtype(x:torch.Tensor, dtype:torch.dtype) -> torch.Tensor:
    return x.to(dtype)

class EnsureDtype(Transform):
    def __init__(self, dtype:torch.dtype):
        self.dtype = dtype
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return ensure_dtype(x, dtype=self.dtype)

def ensure_device(x:torch.Tensor, device:torch.device) -> torch.Tensor:
    return x.to(device)

class EnsureDevice(Transform):
    def __init__(self, device:torch.device):
        self.device = device
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return ensure_device(x, device=self.device)


def ensure_channel_first(x:torch.Tensor) -> torch.Tensor:
    shape = list(x.shape)
    if shape[0] > shape[-1]:
        dims = list(range(len(shape)))
        dims = [dims[-1]] + dims[:-1]
        return x.permute(dims)
    return x