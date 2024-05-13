"""Свёртка операторами"""
from typing import Any, Optional
from collections.abc import Callable
import torch
import torch.nn.common_types
from ..python_tools import identity_if_none

class MSEConv(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Any | None = None,
        dtype: Any | None = None,
        n = 2,
        ndim=2,
        act:Optional[Callable] = None,
):
        super().__init__()
        if ndim == 1: Convnd = torch.nn.Conv1d
        elif ndim == 2: Convnd = torch.nn.Conv2d
        elif ndim == 3: Convnd = torch.nn.Conv3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        ) for _ in range(n)])
        self.act = identity_if_none(act)

    def forward(self, x):
        for i, m in enumerate(self.conv_modules):
            if i == 0: res = self.act(m(x))**2
            else: res += self.act(m(x))**2 # type:ignore
        return res ** 0.5 # type:ignore


class ArctanConv(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Any | None = None,
        dtype: Any | None = None,
        ndim = 2,
        act:Optional[Callable] = None,
):
        super().__init__()
        if ndim == 1: Convnd = torch.nn.Conv1d
        elif ndim == 2: Convnd = torch.nn.Conv2d
        elif ndim == 3: Convnd = torch.nn.Conv3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        ) for _ in range(2)])
        self.act = identity_if_none(act)

    def forward(self, x):
        return torch.arctan2(self.act(self.conv_modules[0](x)), self.act(self.conv_modules[1](x)))


class MSEConvTranspose(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation = 1,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        n = 2,
        ndim=2,
        act:Optional[Callable] = None,
):
        super().__init__()
        if ndim == 1: Convnd = torch.nn.ConvTranspose1d
        elif ndim == 2: Convnd = torch.nn.ConvTranspose2d
        elif ndim == 3: Convnd = torch.nn.ConvTranspose3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        ) for _ in range(n)])
        self.act = identity_if_none(act)

    def forward(self, x):
        for i, m in enumerate(self.conv_modules):
            if i == 0: res = self.act(m(x))**2
            else: res += self.act(m(x))**2 # type:ignore
        return res ** 0.5 # type:ignore


class ArctanConvTranspose(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride = 1,
            padding = 0,
            output_padding = 0,
            groups: int = 1,
            bias: bool = True,
            dilation = 1,
            padding_mode: str = 'zeros',
            device = None,
            dtype = None,
            ndim=2,
            act:Optional[Callable] = None,
        ):
        super().__init__()
        if ndim == 1: Convnd = torch.nn.ConvTranspose1d
        elif ndim == 2: Convnd = torch.nn.ConvTranspose2d
        elif ndim == 3: Convnd = torch.nn.ConvTranspose3d
        else: raise ValueError
        self.conv_modules = torch.nn.ModuleList([Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        ) for _ in range(2)])
        self.act = identity_if_none(act)

    def forward(self, x):
        return torch.arctan2(self.act(self.conv_modules[0](x)), self.act(self.conv_modules[1](x)))