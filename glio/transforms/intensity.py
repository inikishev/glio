from typing import Optional, Any
import random
import torch, numpy as np

from ._base import Transform, RandomTransform

__all__ = [
    "znorm",
    'ZNorm',
    "rand_znorm",
    'RandZNorm',
    "znormch",
    "unznormch",
    'ZNormCh',
    "rand_znormch",
    'RandZNormCh',
    "meanstdnormch",
    "znormbatch",
    "ZNormBatch",
    "rand_znormbatch",
    "RandZNormBatch",
    "norm",
    "NormRange",
    "rand_shift",
    "RandShift",
    "rand_scale",
    "RandScale",
    "normch",
    "NormRangeCh",
    "normbatch",
    "NormRangeBatch",
    "shrink",
    "Shrink",
    "rand_shrink",
    "RandShrink",
    "contrast",
    "Contrast",
    "rand_contrast",
    "RandContrast",
]

def znorm(x:torch.Tensor | np.ndarray, mean=0., std=1.) -> Any:
    """Global z-normalization"""
    if x.std() != 0: return ((x - x.mean()) / (x.std() / std)) + mean
    return x - x.mean()

class ZNorm(Transform):
    def __init__(self, mean=0., std=1.):
        """Global z-normalization"""
        self.mean = mean
        self.std = std
    def forward(self, x): return znorm(x, self.mean, self.std)

def rand_znorm(x, mean = (-1., 1.), std = (0.5, 2)):
    meanv = random.uniform(*mean)
    stdv = random.uniform(*std)
    return znorm(x, meanv, stdv)

class RandZNorm(RandomTransform):
    def __init__(self, mean = (-1., 1.), std = (0.5, 2), p=0.1):
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, x): return rand_znorm(x, self.mean, self.std)


def znormch(x:torch.Tensor, mean=0., std=1.):
    """channel-wise Z-normalization"""
    std = x.std(list(range(1, x.ndim)), keepdim = True) / std
    std[std==0] = 1
    return ((x - x.mean(list(range(1, x.ndim)), keepdim=True)) / std) + mean

def meanstdnormch(x:torch.Tensor, mean, std):
    """Normalize to mean 0 std 1 using given mean and std values"""
    return ((x - x.mean(list(range(1, x.ndim)), keepdim=True)) / std) + mean

def unznormch(x, mean, std):
    """Undoes channel-wise z-norm"""
    inverse_mean = [-mean[i]/std[i] for i in range(len(mean))]
    inverse_std = [1/std[i] for i in range(len(mean))]
    return meanstdnormch(x, inverse_mean, inverse_std)


class ZNormCh(Transform):
    def __init__(self, mean=0., std=1.):
        """channel-wise Z-normalization"""
        self.mean = mean
        self.std = std
    def forward(self, x): return znormch(x, self.mean, self.std)

def rand_znormch(x, mean = (-1., 1.), std = (0.5, 2)):
    meanv = random.uniform(*mean)
    stdv = random.uniform(*std)
    return znormch(x, meanv, stdv)

class RandZNormCh(RandomTransform):
    def __init__(self, mean = (-1., 1.), std = (0.5, 2), p=0.1):
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, x): return rand_znormch(x, self.mean, self.std)


def znormbatch(x:torch.Tensor, mean=0., std=1.):
    """z-normalize a batch channel-wise"""
    std = x.std(list(range(2, x.ndim)), keepdim = True) / std
    std[std==0] = 1
    return ((x - x.mean(list(range(2, x.ndim)), keepdim=True)) / std) + mean

class ZNormBatch(Transform):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    def forward(self, x): return znormbatch(x, self.mean, self.std)

def rand_znormbatch(x, mean = (-1., 1.), std = (0.5, 2)):
    meanv = random.uniform(*mean)
    stdv = random.uniform(*std)
    return znormbatch(x, meanv, stdv)

class RandZNormBatch(RandomTransform):
    def __init__(self, mean = (-1., 1.), std = (0.5, 2), p=0.1):
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, x): return rand_znormbatch(x, self.mean, self.std)


def norm(x:torch.Tensor | np.ndarray, min=0, max=1): #pylint:disable=W0622
    """Normalize to `[min, max]`"""
    x -= x.min()
    xmax = x.max()
    if xmax != 0: x /= xmax
    else: return x
    return x * (max - min) + min

class NormRange(Transform):
    def __init__(self, min=0, max=1):
        """Normalize to `[min, max]`"""
        self.min = min
        self.max = max
    def forward(self, x): return norm(x, self.min, self.max)

def rand_shift(x:torch.Tensor | np.ndarray, val = (-1., 1.)):
    """Shift the input by a random amount. """
    return x + random.uniform(*val)

class RandShift(RandomTransform):
    def __init__(self, val = (-1., 1.), p=0.1):
        self.val = val
        self.p = p
    def forward(self, x): return rand_shift(x, self.val)

def rand_scale(x:torch.Tensor | np.ndarray, val = (0.5, 2)):
    """Scale the input by a random amount. """
    return x * random.uniform(*val)

class RandScale(RandomTransform):
    def __init__(self, val = (0.5, 2), p=0.1):
        self.val = val
        self.p = p
    def forward(self, x): return rand_scale(x, self.val)

def normch(x:torch.Tensor, min=0, max=1): #pylint:disable=W0622
    """Normalize to `[min, max]` channel-wise"""
    x -= x.amin(list(range(1, x.ndim)), keepdim = True)
    xmax = x.amax(list(range(1, x.ndim)), keepdim = True)
    xmax[xmax==0] = 1
    return (x / xmax) * (max - min) + min

class NormRangeCh(Transform):
    def __init__(self, min=0, max=1):
        """Normalize to `[min, max]` channel-wise"""
        self.min = min
        self.max = max
    def forward(self, x): return normch(x, self.min, self.max)

def normbatch(x:torch.Tensor, min=0, max=1): #pylint:disable=W0622
    """Normalize to `[min, max]` channel-wise"""
    x -= x.amin(list(range(2, x.ndim)), keepdim = True)
    xmax = x.amax(list(range(2, x.ndim)), keepdim = True)
    xmax[xmax==0] = 1
    return (x / xmax) * (max - min) + min


class NormRangeBatch(RandomTransform):
    def __init__(self, min=0, max=1):
        """Normalize to `[min, max]` channel-wise"""
        self.min = min
        self.max = max
    def forward(self, x): return normbatch(x, self.min, self.max)


def shrink(x:np.ndarray | torch.Tensor, min=0.2, max=0.8):
    """Shrink the range of the input"""
    xmin = x.min()
    xmax = x.max()
    r = xmax - xmin
    return x.clip(xmin + r * min, xmax - r * (1-max))

class Shrink(Transform):
    def __init__(self, min=0.2, max=0.8):
        """Shrink the range of the input"""
        self.min = min
        self.max = max
    def forward(self, x): return shrink(x, self.min, self.max)

def rand_shrink(x:np.ndarray | torch.Tensor, min=(0., 0.45), max=(0.55, 1.)):
    """Shrink the range of the input"""
    minv = random.uniform(*min)
    maxv = random.uniform(*max)
    return shrink(x, minv, maxv)

class RandShrink(RandomTransform):
    def __init__(self, min=(0., 0.45), max=(0.55, 1.), p=0.1):
        """Shrink the range of the input"""
        self.min = min
        self.max = max
        self.p = p
    def forward(self, x): return rand_shrink(x, self.min, self.max)

def contrast(x, min=0.2, max=0.8):
    """Shrink the range of the input and expand back to original range"""
    xmin = x.min()
    xmax = x.max()
    r = xmax - xmin
    return norm(x.clip(xmin + r * min, xmax - r * (1-max)), xmin, xmax)

class Contrast(Transform):
    def __init__(self, min=0.2, max=0.8):
        """Shrink the range of the input and expand back to original range"""
        self.min = min
        self.max = max
    def forward(self, x): return contrast(x, self.min, self.max)

def rand_contrast(x, min=(0., 0.45), max=(0.55, 1.)):
    """Shrink the range of the input and expand back to original range"""
    minv = random.uniform(*min)
    maxv = random.uniform(*max)
    return contrast(x, minv, maxv)

class RandContrast(RandomTransform):
    def __init__(self, min=(0., 0.45), max=(0.55, 1.), p=0.1):
        """Shrink the range of the input and expand back to original range"""
        self.min = min
        self.max = max
        self.p = p
    def forward(self, x): return rand_contrast(x, self.min, self.max)
