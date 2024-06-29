"""Генераторы блоков"""
from typing import Optional
from collections.abc import Sequence, Iterable, Callable
import torch
                                
__all__ = [
    "seq",
    "ConvBlock",
    "ConvTransposeBlock",
]
def seq(layers, *args):
    """Shortcut for torch.nn.Sequential"""
    modules = []
    if isinstance(layers, Sequence): modules.extend(layers)
    else: modules.append(layers)
    modules.extend(args)
    return torch.nn.Sequential(*modules)

class ConvBlock(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias: bool = True,
        norm: Optional[torch.nn.Module | str | bool] = None,
        dropout: Optional[float | torch.nn.Module] = None,
        act: Optional[torch.nn.Module] = None,
        pool: Optional[int | torch.nn.Module] = None,
        residual = False,
        ndim: int = 2,
        custom_op = None,
        order = "cpand"
    ):
        """Convolution block.

        Default order is: convolution -> pooling -> activation -> normalization -> dropout.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (_type_): Kernel size, integer or tuple.
            stride (int, optional): Stride, integer or tuple. Defaults to 1.
            padding (int, optional): Padding, integer or tuple. Defaults to 0.
            dilation (int, optional): Dilation. Defaults to 1.
            groups (int, optional): Groups, input channels must be divisible by this (?). Defaults to 1.
            bias (bool, optional): Whether to enable convolution bias. Defaults to True.
            norm (Optional[torch.nn.Module  |  str  |  bool], optional): Norm, a module or just `True` for batch norm. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module], optional): Dropout, a module or dropout probability float. Defaults to None.
            act (Optional[torch.nn.Module], optional): Activation function. Defaults to None.
            pool (Optional[int  |  torch.nn.Module], optional): Pooling, a module or an integer of max pooling kernel size and stride. Defaults to None.
            ndim (int, optional): Number of dimensions. Defaults to 2.
            custom_op (_type_, optional): Custom operation to replace convolution. Defaults to None.
            order (str, optional): Order of operations. Defaults to "cpand".
        """
        super().__init__()
        # pick the appropriate modules based on ndim
        if ndim == 1:
            Convnd = torch.nn.Conv1d
            BatchNormnd = torch.nn.BatchNorm1d
            Dropoutnd = torch.nn.Dropout1d
            MaxPoolnd = torch.nn.MaxPool1d
        elif ndim == 2:
            Convnd = torch.nn.Conv2d
            BatchNormnd = torch.nn.BatchNorm2d
            Dropoutnd = torch.nn.Dropout2d
            MaxPoolnd = torch.nn.MaxPool2d
        elif ndim == 3:
            Convnd = torch.nn.Conv3d
            BatchNormnd = torch.nn.BatchNorm3d
            Dropoutnd = torch.nn.Dropout3d
            MaxPoolnd = torch.nn.MaxPool3d
        else: raise NotImplementedError
        if custom_op is not None: Convnd = custom_op

        # convolution
        self.conv = Convnd(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        # pooling
        self.pool = None
        if pool is not None:
            if callable(pool): self.pool = pool
            else: self.pool = MaxPoolnd(pool, pool)

        # activation
        self.act = None
        if act is not None: self.act = act

        # norm
        self.norm = None
        if norm is not None:
            if callable(norm): self.norm = norm
            elif norm is True: self.norm = BatchNormnd(out_channels)
            elif isinstance(norm, str):
                norm = norm.lower()
                if norm in ("batch norm", "batchnorm", "batch", "bn", "b"): self.norm = BatchNormnd(out_channels)
                else: raise ValueError(f"Unknown norm type {norm}")
            else: raise ValueError(f"Unknown norm type {norm}")

        # dropout
        self.dropout = None
        if dropout is not None:
            if callable(dropout): self.dropout = dropout
            elif dropout != 0: self.dropout = Dropoutnd(dropout)
            else: self.dropout = None

        self.order = order.lower()
        self.module_order = []
        for c in self.order:
            if c == "c": self.module_order.append(self.conv)
            elif c == "p": self.module_order.append(self.pool)
            elif c == "a": self.module_order.append(self.act)
            elif c == "n": self.module_order.append(self.norm)
            elif c == "d": self.module_order.append(self.dropout)
            else: raise ValueError(f"Unknown order type `{c}`")

        self.residual = residual

    def forward(self, x:torch.Tensor):
        if self.residual: inputs = x
        for module in self.module_order:
            if module is not None: x = module(x)
        if self.residual: return x + inputs # type:ignore
        return x


class ConvTransposeBlock(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups = 1,
        bias: bool = True,
        dilation = 1,
        norm: Optional[torch.nn.Module | str | bool] = None,
        dropout: Optional[float | torch.nn.Module] = None,
        act: Optional[torch.nn.Module] = None,
        upsample: Optional[float | torch.nn.Module] = None,
        ndim: int = 2,
        custom_op = None,
        order = "ucand"
    ):
        """Transposed convolution block.

        Default order is: upsample -> transposed convolution -> activation -> normalization -> dropout.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (_type_): Kernel size, integer or tuple.
            stride (int, optional): Stride, integer or tuple. Defaults to 1.
            padding (int, optional): Padding, integer or tuple. Defaults to 0.
            output_padding (int, optional): Output padding. Defaults to 0.
            groups (int, optional): Groups, input channels must be divisible by this (?). Defaults to 1.
            bias (bool, optional): Whether to enable convolution bias. Defaults to True.
            dilation (int, optional): Dilation. Defaults to 1.
            norm (Optional[torch.nn.Module  |  str  |  bool], optional): Norm, a module or just `True` for batch norm. Defaults to None.
            dropout (Optional[float  |  torch.nn.Module], optional): Dropout, a module or dropout probability float. Defaults to None.
            act (Optional[torch.nn.Module], optional): Activation function. Defaults to None.
            upsample (Optional[int  |  torch.nn.Module], optional): Upscaling, a module or float of upscaling factor. Defaults to None.
            ndim (int, optional): Number of dimensions. Defaults to 2.
            custom_op (_type_, optional): Custom operation to replace convolution. Defaults to None.
            order (str, optional): Order of operations. Defaults to "cpand".
        """
        super().__init__()
        # pick the appropriate modules based on ndim
        if ndim == 1:
            Convnd = torch.nn.ConvTranspose1d
            BatchNormnd = torch.nn.BatchNorm1d
            Dropoutnd = torch.nn.Dropout1d
        elif ndim == 2:
            Convnd = torch.nn.ConvTranspose2d
            BatchNormnd = torch.nn.BatchNorm2d
            Dropoutnd = torch.nn.Dropout2d
        elif ndim == 3:
            Convnd = torch.nn.ConvTranspose3d
            BatchNormnd = torch.nn.BatchNorm3d
            Dropoutnd = torch.nn.Dropout3d
        else: raise NotImplementedError
        if custom_op is not None: Convnd = custom_op

        # convolution
        self.upconv = Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

        # scale_factor
        self.upsample = None
        if upsample is not None:
            if callable(upsample): self.upscale = upsample
            else: self.upsample = torch.nn.Upsample(scale_factor = upsample)

        # activation
        self.act = None
        if act is not None: self.act = act

        # norm
        self.norm = None
        if norm is not None:
            if callable(norm): self.norm = norm
            elif norm is True: self.norm = BatchNormnd(out_channels)
            elif isinstance(norm, str):
                norm = norm.lower()
                if norm in ("batch norm", "batchnorm", "batch", "bn", "b"): self.norm = BatchNormnd(out_channels)
                else: raise ValueError(f"Unknown norm type {norm}")
            else: raise ValueError(f"Unknown norm type {norm}")

        # dropout
        self.dropout = None
        if dropout is not None:
            if callable(dropout): self.dropout = dropout
            elif dropout != 0: self.dropout = Dropoutnd(dropout)
            else: self.dropout = None

        self.order = order.lower()
        self.module_order = []
        for c in self.order:
            if c == "c": self.module_order.append(self.upconv)
            elif c == "u": self.module_order.append(self.upsample)
            elif c == "a": self.module_order.append(self.act)
            elif c == "n": self.module_order.append(self.norm)
            elif c == "d": self.module_order.append(self.dropout)
            else: raise ValueError(f"Unknown order type `{c}`")

    def forward(self, x:torch.Tensor):
        for module in self.module_order:
            if module is not None: x = module(x)
        return x

class LinearBlock(torch.nn.Module):
    def __init__(self,
        in_features: Optional[int],
        out_features: int,
        bias: bool = True,
        norm: bool = False,
        dropout: Optional[float] = None,
        act: Optional[torch.nn.Module] = None,
        flatten: bool | Sequence[int] = False,
        lazy = False,
        residual = False,
        custom_op = None,
        order = 'fland'
    ):
        """Linear block

        Args:
            in_features (Optional[int]): _description_
            out_features (int): _description_
            bias (bool, optional): _description_. Defaults to True.
            norm (bool, optional): _description_. Defaults to False.
            dropout (Optional[float], optional): _description_. Defaults to None.
            act (Optional[torch.nn.Module], optional): _description_. Defaults to None.
            flatten (bool | Sequence[int], optional): _description_. Defaults to False.
            lazy (bool, optional): _description_. Defaults to False.
            custom_op (_type_, optional): _description_. Defaults to None.
            order (str, optional): _description_. Defaults to 'fland'.
        """
        super().__init__()
        # linear
        if lazy: self.linear = torch.nn.LazyLinear(out_features, bias)
        elif custom_op is None: self.linear = torch.nn.Linear(in_features, out_features, bias) # type:ignore
        else: self.linear = custom_op(in_features, out_features, bias) # type:ignore

        # norm
        self.norm = None
        if norm is not None:
            if callable(norm): self.norm = norm
            elif norm is True: self.norm = torch.nn.BatchNorm1d(out_features)
            elif isinstance(norm, str):
                norm = norm.lower()
                if norm in ("batch norm", "batchnorm", "batch", "bn", "b"): self.norm = torch.nn.BatchNorm1d(out_features)
                else: raise ValueError(f"Unknown norm type {norm}")
            else: raise ValueError(f"Unknown norm type {norm}")

        # dropout
        self.dropout = None
        if dropout is not None:
            if callable(dropout): self.dropout = dropout
            elif dropout != 0: self.dropout = torch.nn.Dropout(dropout)
            else: self.dropout = None

        # activation
        self.act = None
        if act is not None: self.act = act

        self.flatten = None
        if flatten is not None:
            if callable(flatten): self.flatten = flatten
            elif flatten is True: self.flatten = torch.nn.Flatten()
            else: flatten = torch.nn.Flatten(*flatten) # type:ignore

        self.order = order.lower()
        self.module_order = []
        for c in self.order:
            if c == "f": self.module_order.append(self.flatten)
            elif c == "l": self.module_order.append(self.linear)
            elif c == "a": self.module_order.append(self.act)
            elif c == "n": self.module_order.append(self.norm)
            elif c == "d": self.module_order.append(self.dropout)
            else: raise ValueError(f"Unknown order type `{c}`")

        self.residual = residual

    def forward(self, x:torch.Tensor):
        if self.residual: inputs = x
        for module in self.module_order:
            if module is not None: x = module(x)
        if self.residual: return x + inputs # type:ignore
        return x

class Block(torch.nn.Module):
    def __init__(self,
        module: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module],
        out_channels: Optional[int] = None,
        norm: bool = False,
        dropout: Optional[float] = None,
        act: Optional[torch.nn.Module] = None,
        pool: Optional[torch.nn.Module] = None,
        residual = False,
        ndim = 2,
        order = "mpand",
    ):
        super().__init__()
        if callable(module): self.module = module
        else: self.module = seq(module)

        if ndim == 1:
            BatchNormnd = torch.nn.BatchNorm1d
            Dropoutnd = torch.nn.Dropout1d
            MaxPoolnd = torch.nn.MaxPool1d
        elif ndim == 2:
            BatchNormnd = torch.nn.BatchNorm2d
            Dropoutnd = torch.nn.Dropout2d
            MaxPoolnd = torch.nn.MaxPool2d
        elif ndim == 3:
            BatchNormnd = torch.nn.BatchNorm3d
            Dropoutnd = torch.nn.Dropout3d
            MaxPoolnd = torch.nn.MaxPool3d
        else: raise NotImplementedError(ndim)

        # activation
        self.act = None
        if act is not None: self.act = act

        # pooling
        self.pool = None
        if pool is not None:
            if callable(pool): self.pool = pool
            else: self.pool = MaxPoolnd(pool, pool)

        # norm
        self.norm = None
        if norm is not None:
            if callable(norm): self.norm = norm
            elif norm is True: self.norm = BatchNormnd(out_channels) # type:ignore
            elif isinstance(norm, str):
                norm = norm.lower()
                if norm in ("batch norm", "batchnorm", "batch", "bn", "b"): self.norm = BatchNormnd(out_channels) # type:ignore
                else: raise ValueError(f"Unknown norm type {norm}")
            else: raise ValueError(f"Unknown norm type {norm}")

        # dropout
        self.dropout = None
        if dropout is not None:
            if callable(dropout): self.dropout = dropout
            elif dropout != 0: self.dropout = Dropoutnd(dropout)
            else: self.dropout = None


        self.order = order.lower()
        self.module_order = []
        for c in self.order:
            if c == "m": self.module_order.append(self.module)
            elif c == "p": self.module_order.append(self.pool)
            elif c == "a": self.module_order.append(self.act)
            elif c == "n": self.module_order.append(self.norm)
            elif c == "d": self.module_order.append(self.dropout)
            else: raise ValueError(f"Unknown order type `{c}`")

        self.residual = residual

    def forward(self, x:torch.Tensor):
        if self.residual: inputs = x
        for module in self.module_order:
            if module is not None: x = module(x)
        if self.residual: return x + inputs # type:ignore
        return x