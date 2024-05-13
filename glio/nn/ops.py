"""Hi"""
import torch
from torch import nn

class LearnableAdd(nn.Module):
    def __init__(self, init = 0., learnable=True):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(init, requires_grad=learnable), learnable)
    def forward(self, x):
        return x + self.bias


class LearnableMul(nn.Module):
    def __init__(self, init = 1., learnable=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init, requires_grad=learnable), learnable)
    def forward(self, x):
        return x * self.weight

class LearnableNorm(nn.Module):
    def __init__(self, init_w = 1., init_b = 0., learnable=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init_w, requires_grad=learnable), learnable)
        self.bias = nn.Parameter(torch.tensor(init_b, requires_grad=learnable), learnable)
    def forward(self, x):
        return (x * self.weight) + self.bias

class LearnableNormChannelwise(nn.Module):
    def __init__(self, in_channels, ndim = 2, init_w = 1., init_b = 0., learnable=True):
        super().__init__()
        self.n_channels = in_channels
        self.weight = nn.Parameter(torch.full(size=(1, in_channels, *[1 for _ in range(ndim)]), fill_value=init_w, requires_grad=learnable), requires_grad=learnable)
        self.bias = nn.Parameter(torch.full(size=(1, in_channels, *[1 for _ in range(ndim)]), fill_value=init_b, requires_grad=learnable), requires_grad=learnable)
        self.learnable = learnable

    def forward(self, x):
        return (x * self.weight) + self.bias

class LearnableChannelMix(nn.Module):
    def __init__(self, in_channels, out_channels, ndim = 2, init_w = 1., init_b = 0., learnable=True):
        super().__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.ParameterList([nn.Parameter(torch.full(size=(1, in_channels, *[1 for _ in range(ndim)]), fill_value=init_w, requires_grad=learnable), requires_grad=learnable) for _ in range(out_channels)])
        self.biases = nn.ParameterList([nn.Parameter(torch.full(size=(1, in_channels, *[1 for _ in range(ndim)]), fill_value=init_b, requires_grad=learnable), requires_grad=learnable) for _ in range(out_channels)])

    def forward(self, x:torch.Tensor):
        return torch.stack([((x * self.weights[i]) + self.biases[i]).mean(dim=1) for i in range(self.out_channels)], dim=1)

if __name__ == "__main__":
    img = torch.randn(16, 3, 28, 28)
    print(LearnableChannelMix(3, 5, 2)(img).shape)