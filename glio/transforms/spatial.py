from typing import Optional, Any
from collections.abc import Sequence
import random
import torch, numpy as np

from ._base import Transform, RandomTransform

__all__ = [
    "randflip",
    "randflipt",
    "RandFlip",
    "RandFlipt",
    "randrot90",
    "randrot90t",
    "RandRot90",
    "RandRot90t",
    "fast_slice_reduce_size",
    "FastSliceReduceSize",
]
def randflip(x:torch.Tensor):
    flip_dims = random.sample(population = range(1, x.ndim), k = random.randint(1, x.ndim-1))
    return x.flip(flip_dims)

def randflipt(x:Sequence[torch.Tensor]):
    flip_dims = random.sample(population = range(1, x[0].ndim), k = random.randint(1, x[0].ndim-1))
    return [i.flip(flip_dims) for i in x]

class RandFlip(RandomTransform):
    def __init__(self, p:float=0.5): self.p = p
    def forward(self, x:torch.Tensor): return randflip(x)

class RandFlipt(RandomTransform):
    def __init__(self, p:float=0.5): self.p = p
    def forward(self, x:Sequence[torch.Tensor]): return randflipt(x)

def randrot90(x:torch.Tensor):
    flip_dims = random.sample(range(1, x.ndim), k=2)
    k = random.randint(-3, 3)
    return x.rot90(k = k, dims = flip_dims)

def randrot90t(x:Sequence[torch.Tensor]):
    flip_dims = random.sample(range(1, x[0].ndim), k=2)
    k = random.randint(-3, 3)
    return [i.rot90(k = k, dims = flip_dims) for i in x]

class RandRot90(RandomTransform):
    def __init__(self, p:float=0.5): self.p = p
    def forward(self, x:torch.Tensor): return randrot90(x)

class RandRot90t(RandomTransform):
    def __init__(self, p:float=0.5): self.p = p
    def forward(self, x:Sequence[torch.Tensor]): return randrot90t(x)


def fast_slice_reduce_size(x:torch.Tensor, min_shape: Sequence[int]):
    times = [i/j for i,j in zip(x.shape[1:], min_shape)]
    min_times = int(min(times))
    if min_times <= 2:
        return x
    else:
        reduction = random.randrange(2, min_times)
        ndim = x.ndim
        if ndim == 2: return x[:, ::reduction]
        elif ndim == 3: return x[:, ::reduction, ::reduction]
        elif ndim == 4: return x[:, ::reduction, ::reduction, ::reduction]
        else: raise ValueError(f'{x.shape = }')

def fast_slice_reduce_sizet(seq:Sequence[torch.Tensor], min_shape: Sequence[int]):
    x = seq[0]
    times = [i/j for i,j in zip(x.shape[1:], min_shape)]
    min_times = int(min(times))
    if min_times <= 2:
        return seq
    else:
        reduction = random.randrange(2, min_times)
        ndim = x.ndim
        if ndim == 2: return [i[:, ::reduction] for i in seq]
        elif ndim == 3: return [i[:, ::reduction, ::reduction] for i in seq]
        elif ndim == 4: return [i[:, ::reduction, ::reduction, ::reduction] for i in seq]
        else: raise ValueError(f'{x.shape = }')

class FastSliceReduceSize(RandomTransform):
    def __init__(self, min_shape: Sequence[int], p:float=0.5):
        self.min_shape = min_shape
        self.p = p
    def forward(self, x:torch.Tensor): return fast_slice_reduce_size(x, self.min_shape)
    
class FastSliceReduceSizet(RandomTransform):
    def __init__(self, min_shape: Sequence[int], p:float=0.5):
        self.min_shape = min_shape
        self.p = p
    def forward(self, x:Sequence[torch.Tensor]): return fast_slice_reduce_sizet(x, self.min_shape)