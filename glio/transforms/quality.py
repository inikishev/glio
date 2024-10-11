from typing import Optional, Any
import random
import torch, numpy as np
from ._base import Transform, RandomTransform
__all__ = [
    "add_gaussian_noise",
    "GaussianNoise",
    "add_gaussian_noise_triangular",
    "GaussianNoiseTriangular",
]
def add_gaussian_noise(x):
    return x + torch.randn_like(x)
class GaussianNoise(RandomTransform):
    def __init__(self, p = 0.1):
        self.p = p
    def forward(self, x): return add_gaussian_noise(x)

def add_gaussian_noise_triangular(x):
    return x + torch.randn_like(x) * random.triangular(0, 1, 0)

class GaussianNoiseTriangular(RandomTransform):
    def __init__(self, p = 0.1):
        self.p = p
    def forward(self, x): return add_gaussian_noise_triangular(x)