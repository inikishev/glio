from typing import Optional, Any
import random
import torch, numpy as np
def randflip(x:torch.Tensor):
    flip_dims = random.sample(range(1, x.ndim), random.randint(1, x.ndim-1))
    return x.flip(flip_dims)

def randflipt(x:tuple[torch.Tensor, torch.Tensor]):
    flip_dims = random.sample(range(1, x[0].ndim), random.randint(1, x[0].ndim-1))
    return x[0].flip(flip_dims), x[1].flip(flip_dims)

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