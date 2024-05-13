from contextlib import nullcontext
import random
import numpy as np
import torch
from ..torch_tools import seeded_rng

__all__ = ("randrect", "randrect_like", "randperm", "randperm_like")

def randrect(shape, fill = lambda: random.normalvariate(0, 1), device=None, seed=None):
    """Randomly placed rectangle filled with random value from normal distribution."""
    with seeded_rng(seed):
        # scalar is just random
        if len(shape) == 0:
            return fill()

        # create start-end slices for the rectangle
        slices = []
        for dim_size in shape:
            if dim_size == 1: slices.append(slice(None))
            else:
                start = random.randrange(0, dim_size-1)
                end = random.randrange(start, dim_size)
                slices.append(slice(start, end))
        # determine fill value
        fill_value = fill()
        res = torch.zeros(shape, device=device)
        res[slices] = fill_value
        return res

def randrect_like(x:torch.Tensor, fill = lambda: random.normalvariate(0, 1), device=None, seed=None):
    """Randomly placed rectangle filled with random value from normal distribution."""
    return randrect(x.shape, fill=fill, device=device, seed=seed)

def randperm(n,
    *,
    out = None,
    dtype= None,
    layout = None,
    device= None,
    pin_memory = False,
    requires_grad = False,
    seed=None,
    ):
    with seeded_rng(seed):
        return torch.randperm(n, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)

def randperm_like(obj,
    out = None,
    dtype= None,
    layout = None,
    device= None,
    pin_memory = False,
    requires_grad = False,
    seed=None,
    ):
    return randperm(len(obj), out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad, seed=seed)

