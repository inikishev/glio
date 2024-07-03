from typing import Optional, Any
import random
import torch, numpy as np
def randflip(x:torch.Tensor):
    flip_dims = random.sample(population = range(1, x.ndim), k = random.randint(1, x.ndim-1))
    return x.flip(flip_dims)

def randflipt(x:tuple[torch.Tensor, ...]):
    flip_dims = random.sample(population = range(1, x[0].ndim), k = random.randint(1, x[0].ndim-1))
    return [i.flip(flip_dims) for i in x]
class RandFlip:
    def __init__(self, p:float=0.5):
        self.p = p
    def __call__(self, x:torch.Tensor):
        if random.random() < self.p: return randflip(x)
        else: return x

class RandFlipt:
    def __init__(self, p:float=0.5):
        self.p = p
    def __call__(self, x:tuple[torch.Tensor, torch.Tensor]):
        if random.random() < self.p: return randflipt(x)
        else: return x

def randrot90(x:torch.Tensor):
    flip_dims = random.sample(range(1, x.ndim), k=2)
    k = random.randint(-3, 3)
    return x.rot90(k = k, dims = flip_dims)

def randrot90t(x:tuple[torch.Tensor, ...]):
    flip_dims = random.sample(range(1, x[0].ndim), k=2)
    k = random.randint(-3, 3)
    return [i.rot90(k = k, dims = flip_dims) for i in x]

class RandRot90:
    def __init__(self, p:float=0.5):
        self.p = p
    def __call__(self, x:torch.Tensor):
        if random.random() < self.p: return randrot90(x)
        else: return x
class RandRot90t:
    def __init__(self, p:float=0.5):
        self.p = p
    def __call__(self, x:tuple[torch.Tensor, ...]):
        if random.random() < self.p: return randrot90t(x)
        else: return x