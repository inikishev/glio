"""Elementwise"""
import torch
from torch import nn

class Elementwise(nn.Module):
    """Applies elementwise multiplication and addition with learnable arrays, i.e. `x * weight + bias`.

    By default initialzied with ones and zeroes and therefore computes identity function.

    `in_size`: `(C, H, W)` for 2d input."""
    def __init__(self, in_size, bias = True, weight_init = torch.ones, bias_init = torch.zeros):
        super().__init__()
        self.weight = nn.Parameter(weight_init(in_size), True)
        if bias:
            self.bias = nn.Parameter(bias_init(in_size), True)
            self.has_bias = True
        else:
            self.has_bias = False

    def forward(self, x):
        if self.has_bias: return x * self.weight + self.bias
        else: return x * self.weight

class ElementwiseAdd(nn.Module):
    """Applies elementwise addition with a learnable array, i.e. `x + bias`.

    `in_size`: `(C, H, W)` for 2d input."""
    def __init__(self, in_size, init = torch.zeros):
        super().__init__()
        self.bias = nn.Parameter(init(in_size), True)
    def forward(self, x):
        return x + self.bias

class ElementwiseChannels(nn.Module):
    """Applies elementwise multiplication and addition with multiple learnable arrays, i.e. `x * weight`, each filter has a bias scalar.

    If `pool` is `None`, all resulting arrays are concatenated by channels, meaning output has `in_channels * n_filters` channels.

    Otherwise `pool` must be a function like torch.mean, that will be taken across the channel dimension of each filter.
    Output has `n_filters` channels.
    Note that `pool` is called like this: `pool(x, 1)` on each element in a batch, vectorized using vmap."""
    def __init__(self, in_size, n_filters, bias = True, pool = None, weight_init = torch.ones, bias_init = torch.zeros):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        self.weight = nn.Parameter(weight_init(n_filters, *in_size), True)
        if bias:
            self.bias = nn.Parameter(bias_init(n_filters, *([1] * len(in_size))), True)
            self.has_bias = True
        else: self.has_bias = False
        if pool is None:
            if self.has_bias:
                self.batched_filt = torch.vmap((lambda x: (x * self.weight + self.bias).flatten(0, 1)))
            else:
                self.batched_filt = torch.vmap((lambda x: (x * self.weight).flatten(0, 1)))
        else:
            if self.has_bias:
                self.batched_filt = torch.vmap((lambda x: pool(x * self.weight + self.bias, 1)))
            else:
                self.batched_filt = torch.vmap((lambda x: pool(x * self.weight, 1)))

    def forward(self, x):
        return self.batched_filt(x)

if __name__ == "__main__":
    filt = ElementwiseChannels((3, 28, 28), 32)
    batch = torch.randn(16, 3, 28, 28)
    print(filt(batch).shape)

    filt_pooled = ElementwiseChannels((3, 28, 28), 32, pool = torch.mean)
    batch = torch.randn(16, 3, 28, 28)
    print(filt_pooled(batch).shape)
