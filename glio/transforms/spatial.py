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
    "RandRot90t"
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