
from itertools import product
from collections import OrderedDict

from typing import Optional

def conv_outsize(in_size:tuple,kernel_size, stride = 1, padding = 0,output_padding = 0, dilation=1):
    "conv 2d"
    if isinstance(in_size, int): in_size = (in_size,)
    if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(in_size)
    if isinstance(stride, int): stride = [stride]*len(in_size)
    if isinstance(padding, int): padding = [padding]*len(in_size)
    if isinstance(output_padding, int): output_padding = [output_padding]*len(in_size)
    if isinstance(dilation, int): dilation = [dilation]*len(in_size)
    out_size = [int((in_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1) for i in range(len(in_size))]
    print(out_size)
    return out_size

def convtranpose_outsize(in_size:tuple,kernel_size, stride = 1, padding = 0, output_padding = 0, dilation=(1,1)):
    """conv transpose 2d"""
    if isinstance(in_size, int): in_size = (in_size,)
    if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(in_size)
    if isinstance(stride, int): stride = [stride]*len(in_size)
    if isinstance(padding, int): padding = [padding]*len(in_size)
    if isinstance(output_padding, int): output_padding = [output_padding]*len(in_size)
    if isinstance(dilation, int): dilation = [dilation]*len(in_size)
    out_size = [int((in_size[i]-1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1) for i in range(len(in_size))]
    print(out_size)
    return out_size


def find_samesize_params(
    kernel_size : Optional[int],
    stride: Optional[int],
    padding: Optional[int]=None,
    output_padding: Optional[int]=None,
    dilation: Optional[int]=1,
    order=("stride", "padding", "output_padding", "kernel_size", "dilation"),
    maxvalue = 10,
):
    """Horrible...

    Args:
        kernel_size (Optional[int]): _description_
        stride (Optional[int]): _description_
        padding (Optional[int], optional): _description_. Defaults to None.
        output_padding (Optional[int], optional): _description_. Defaults to None.
        dilation (Optional[int], optional): _description_. Defaults to 1.
        order (tuple, optional): _description_. Defaults to ("stride", "padding", "output_padding", "kernel_size", "dilation").
        maxvalue (int, optional): _description_. Defaults to 10.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    kwargs = OrderedDict(
        in_size=(96, 96),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )
    set_kwargs = OrderedDict({k: v for k, v in kwargs.items() if v is not None})
    unset_kwargs_unordered = OrderedDict({k: v for k, v in kwargs.items() if v is None})
    unset_kwargs = OrderedDict()
    for key in order:
        if key in unset_kwargs_unordered:
            unset_kwargs[key] = unset_kwargs_unordered[key]

    for k, v in unset_kwargs.items():
        if k == "kernel_size":
            unset_kwargs[k] = 1
        elif k == "stride":
            unset_kwargs[k] = 1
        elif k == "padding":
            unset_kwargs[k] = 0
        elif k == "output_padding":
            unset_kwargs[k] = 0
        elif k == "dilation":
            unset_kwargs[k] = 1

    outsize = conv_outsize(**set_kwargs, **unset_kwargs)
    if list(outsize) == [96, 96]:
        all_kwargs = set_kwargs
        all_kwargs.update(unset_kwargs)
        all_kwargs.pop("in_size")
        return all_kwargs
    num_params = len(unset_kwargs)
    for maxv in range(maxvalue):
        combos = product(range(maxv), repeat=num_params)
        for combo in combos:
            combo_kwargs = {}
            for i, k in enumerate(unset_kwargs.keys()):
                if k in ("kernel_size", "stride", "dilation"):
                    v = max(1, combo[i])
                else:
                    v = combo[i]
                combo_kwargs[k] = v
            # print(combo_kwargs)
            if conv_outsize(**set_kwargs, **combo_kwargs) == [96, 96]:
                all_kwargs = set_kwargs
                all_kwargs.update(combo_kwargs)
                all_kwargs.pop("in_size")
                return dict(all_kwargs)
    raise ValueError("No valid combination of parameters found.")
