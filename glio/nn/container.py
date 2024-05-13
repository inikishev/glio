from collections.abc import Callable
from torch import nn
import torch

class ChannelConcat(nn.Module):
    """Возвращает конкатенацию выходов слоёв. Слои должны возвращать сигнал одинкового пространственного размера."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.layers], dim=1)


class SignalConcat(nn.Module):
    """Вызывает forward на конкатенации переданных аргументов. Аргументы должны быть одинкового пространственного размера."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, *x):
        return self.layers(torch.cat(x, dim=1))

class EachSignalConcat(nn.Module):
    """Возвращает выход слоёв на каждом из переданных аргументов и возвращает конкатенацию по каналам. Аргументы должны быть одинкового пространственного размера."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, *x):
        return torch.cat([self.layers(i) for i in x], dim=1)


class ChannelMean(nn.Module):
    "Возвращает среднее выходов слоёв. Слои должны возвращать сигнал одинкового размера."
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0: x = layer(x)
            else: x += layer(x)
        return x / len(self.layers)

class SignalMean(nn.Module):
    """Вызывает forward на поканальном среднем переданных аргументов. Аргументы должны быть одинкового размера."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, *x):
        return self.layers(torch.mean(*x, dim=0))

class EachSignalMean(nn.Module):
    """Возвращает выход слоёв на каждом из переданных аргументов и возвращает среднее по каналам. Аргументы должны быть одинкового пространственного размера."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, *x):
        return torch.cat([self.layers(i) for i in x], dim=1)

class FuncModule(nn.Module):
    def __init__(self, func:Callable):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)

def to_module(x) -> nn.Module:
    if isinstance(x, nn.Module): return x
    elif isinstance(x, Callable): return FuncModule(x)
    else: raise TypeError("Can't convert to module")