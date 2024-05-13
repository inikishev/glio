from typing import Any,Optional
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from torchvision.utils import make_grid
import numpy as np, torch
import scipy.ndimage, scipy.signal
from ...torch_tools import to_device


def _match_target(artists:list[Artist], target:Any, types = (Line2D,PathCollection,AxesImage)) -> list[Artist]:
    artists = [a for a in artists if isinstance(a,types)]
    if target is None: return artists
    if isinstance(target, int): return [artists[target]]
    if isinstance(target, str): return [a for a in artists if a.get_label() == target]
    if isinstance(target, (list, tuple)): return [a for a in artists if a.get_label() in target]
    if isinstance(target, slice): return artists[target]
    if callable(target): return [a for a in artists if target(a)]
    raise ValueError(f"invalid target type {type(target)}")

def _get_array(obj:Any) -> np.ndarray:
    if isinstance(obj, Line2D): return np.asanyarray(obj.get_data())
    if isinstance(obj, PathCollection): return np.asanyarray(obj.get_offsets().data).T # type:ignore
    if isinstance(obj, AxesImage): return np.asanyarray(obj.get_array().data) # type:ignore
    raise ValueError(f"invalid object type {type(obj)}")


def linechart(
    ax: Axes,
    y,
    x=None,
    label=None,
    color:Optional[str]=None,
    linewidth=0.5,
    **kwargs,
    ):
    # convert y to numpy
    if y is None: y = np.arange(len(x)) # type:ignore
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    elif not isinstance(y, np.ndarray): y = np.array(y)

    # convert x to numpy
    if x is None: x = np.arange(len(y))
    elif isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray): x = np.array(x)

    # plot
    ax.plot(x, y, label=label, color=color, linewidth=linewidth, **kwargs)

def scatter(
    ax: Axes,
    x,
    y,
    s = None,
    c = None,
    label=None,
    color:Optional[str]=None,
    marker = None,
    cmap = None,
    vmin=None,
    vmax=None,
    alpha = None,
    **kwargs,
):
    # convert y to numpy
    if y is None: y = np.arange(len(x))
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    elif not isinstance(y, np.ndarray): y = np.array(y)

    # convert x to numpy
    if x is None: x = np.arange(len(y))
    elif isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray): x = np.array(x)

    if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
    if isinstance(c, torch.Tensor): c = c.detach().cpu().numpy()

    # plot
    ax.scatter(x, y, s=s, c=c, label=label, color=color, marker=marker,cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, **kwargs)

def imshow(
    ax: Axes,
    x,
    label = None,
    cmap = None,
    vmin=None,
    vmax=None,
    alpha = None,
    mode = 'auto',
    allow_alpha=False,
    **kwargs,):
    # convert to numpy
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray): x = np.array(x)

    # determine mode
    if mode == "auto":
        if x.ndim > 3: mode = "*c*"
        if x.ndim == 3 and x.shape[2] > 4 and x.shape[0] < x.shape[2]: mode = 'c*'

    # if batched, take 1st element
    if mode == "*c*":
        while x.ndim > 3: x = x[0]
        mode = "c*"

    # if channel first, transpose
    if mode == "c*":
        x = x.transpose(1,2,0)

    # fix invalid ch count
    if x.ndim == 3:
        if x.shape[2] == 2:
            x = np.concatenate([x, x[:, :, 0:1]], axis=2)
        elif x.shape[2] > (4 if allow_alpha else 3):
            x = x[:,:,:3]

    ax.imshow(x, label=label, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, **kwargs)

def imshow_batch(
    ax: Axes,
    x,
    label=None,
    maxelems = 16,
    ncol = None,
    nrow = None,
    cmap = None,
    vmin=None,
    vmax=None,
    alpha = None,
    mode = 'auto',
    allow_alpha=False,
    padding=2,
    normalize=True,
    scale_each = False,
    pad_value = 'min',
    **kwargs,
    ):
    # convert to cpu tensor
    if isinstance(x, torch.Tensor): x = x.detach().cpu()
    elif not isinstance(x, (np.ndarray,torch.Tensor)): x = torch.from_numpy(np.array(x))

    # determine mode
    if mode == "auto":
        grid_elem = x[0]
        while grid_elem.ndim > 3: grid_elem = grid_elem[0]
        if grid_elem.ndim == 3:
            if grid_elem.shape[2] > 4 and grid_elem.shape[0] < grid_elem.shape[2]: mode = 'bc*'
            else: mode = "b*c"
        else: mode = "b*"

    # get first maxelems
    if mode.startswith("b"):
        while x.ndim > 4: x = x[0]
        x = x[:maxelems]

    # if channel last, transpose for torcvision make grid
    if mode.endswith("c"):
        x = x.permute(0, 3, 1, 2) # type:ignore

    # fix invalid ch count
    if x.ndim == 4:
        # channel first
        if x.shape[1] == 2:
            x = torch.cat([x, x[:, 0:1]], dim=1) # type:ignore
        elif x.shape[1] > (4 if allow_alpha else 3):
            x = x[:,3]

    # make grid
    # determine nrow
    nelem = x.shape[0]
    if ncol is None:
        if nrow is None:
            # automatically
            grid_rows = max(1, int(math.ceil(nelem**0.5)))
        else: grid_rows = nrow
    # determine from ncol
    else:
        grid_rows = int(math.ceil(nelem/ncol))
    # pad value
    if pad_value == 'min': pad_value = x.min()
    elif pad_value == 'max': pad_value = x.max()
    elif pad_value == 'mean': pad_value = x.mean()
    # grid
    grid = make_grid(x, nrow=grid_rows, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value).permute(1,2,0) # pylint:disable=W0621 # type:ignore

    # we get HWC, now imshow
    ax.imshow(grid, label=label, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, **kwargs)


def show_min(ax:Axes, target = 0, size=6, show_label = True, weight="bold", **kwargs):
    artists = _match_target(ax.get_children(), target)
    for artist in artists:
        data = _get_array(artist)
        # data is x,y tuple - for line and scatter
        if isinstance(data, tuple):
            x, y = data
            minx, miny = x[np.argmin(y)], y.min()
        # image
        else:
            raise NotImplementedError

        #label
        if show_label:
            label = f'{artist.get_label()} min\n'
            if label.startswith("_"): label='min\n'
        else: label = ''
        # put the text
        ax.text(minx, miny, f"{label}x={minx:.3f}\ny={miny:.3f}", size=size, weight=weight, **kwargs)

def show_max(ax:Axes, target = 0, size=6, show_label = True, weight="bold", **kwargs):
    artists = _match_target(ax.get_children(), target)
    for artist in artists:
        data = _get_array(artist)
        # data is x,y tuple - for line and scatter
        if isinstance(data, tuple):
            x, y = data
            maxx, maxy = x[np.argmax(y)], y.max()
        # image
        else:
            raise NotImplementedError

        #label
        if show_label:
            label = f'{artist.get_label()} max\n'
            if label.startswith("_"): label='max\n'
        else: label = ''
        # put the text
        ax.text(maxx, maxy, f"{label}x={maxx:.3f}\ny={maxy:.3f}", size=size, weight=weight, **kwargs)


def fill_below(ax:Axes, target = 0, color = None, alpha = 0.3, **kwargs):
    artists = _match_target(ax.get_children(), target, types = (Line2D, PathCollection))
    for artist in artists:
        x, y = _get_array(artist)
        ax.fill_between(x, ax.get_ylim()[0], y, alpha=alpha, color=color, **kwargs)

def fill_above(ax:Axes, target = 0, color = None, alpha = 0.3, **kwargs):
    artists = _match_target(ax.get_children(), target, types = (Line2D, PathCollection))
    for artist in artists:
        x, y = _get_array(artist)
        ax.fill_between(x, ax.get_ylim()[1], y, alpha=alpha, color=color, **kwargs)

def fill_between(ax:Axes, target1, target2, color = None, alpha = 0.3, **kwargs):
    artists1 = _match_target(ax.get_children(), target1, types = (Line2D, PathCollection))
    artists2 = _match_target(ax.get_children(), target2, types = (Line2D, PathCollection))
    for artist1, artist2 in zip(artists1, artists2):
        x, y1 = _get_array(artist1)
        _, y2 = _get_array(artist2)
        ax.fill_between(x, y1, y2, alpha=alpha, color=color, **kwargs)

def differential(ax:Axes, target = 0, color:Optional[str]=None, linewidth=0.5, order=1, **kwargs,):
    artists = _match_target(ax.get_children(), target, Line2D)
    for artist in artists:
        x, y = _get_array(artist)
        diff = y
        for _ in range(order): diff = np.diff(diff)
        if order == 1: label = f'{artist.get_label()} diff'
        else: label = f'{artist.get_label()} diff {order}'
        ax.plot(x[:-1], diff, label = label, color=color, linewidth=linewidth, **kwargs)

def moving_average(ax:Axes, target = 0, length=0.1, color:Optional[str]=None, linewidth=0.5,**kwargs,):
    artists = _match_target(ax.get_children(), target, Line2D)
    for artist in artists:
        x, y = _get_array(artist)
        if isinstance(length, float): ma_length = int(len(x) * length)
        else: ma_length = length
        ma = scipy.signal.convolve(y, np.ones(ma_length)/ma_length, 'same')
        ax.plot(x, ma, label=f'{artist.get_label()} mean {ma_length}', color=color, linewidth=linewidth, **kwargs)

def moving_median(ax:Axes, target = 0, length=0.1, color:Optional[str]=None, linewidth=0.5,**kwargs,):
    artists = _match_target(ax.get_children(), target, Line2D)
    for artist in artists:
        x, y = _get_array(artist)
        if isinstance(length, float): ma_length = int(len(x) * length)
        else: ma_length = length
        ma = scipy.ndimage.median_filter(y, ma_length, mode='nearest')
        ax.plot(x, ma, label=f'{artist.get_label()} median {ma_length}', color=color, linewidth=linewidth, **kwargs)

def legend(ax:Axes, size=6, edgecolor=None, linewidth=3., frame_alpha = 0.3, **kwargs):
    if 'prop' in kwargs: prop = kwargs["prop"]
    else: prop = {}
    if size is not None: prop['size'] = size

    leg = ax.legend(prop=prop, edgecolor=edgecolor, **kwargs)
    leg.get_frame().set_alpha(frame_alpha)

    if linewidth is not None:
        for line in leg.get_lines():
            line.set_linewidth(linewidth)

def xlim(ax:Axes,left = None, right = None, **kwargs):
    ax.set_xlim(left=left, right=right, **kwargs)

def ylim(ax:Axes,bottom=None,top=None, **kwargs):
    ax.set_ylim(bottom=bottom, top=top, **kwargs)

def majorxticks(ax:Axes, ticks = 35, steps=(1, 2, 2.5, 5, 10), **kwargs):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=ticks, steps=steps, **kwargs))

def majoryticks(ax:Axes, ticks = 20, steps=(1, 2, 2.5, 5, 10), **kwargs):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ticks, steps=steps, **kwargs))

def minorxticks(ax:Axes, ticks = 4, **kwargs):
    ax.xaxis.set_minor_locator(AutoMinorLocator(ticks, **kwargs))

def minoryticks(ax:Axes, ticks = 4, **kwargs):
    ax.yaxis.set_minor_locator(AutoMinorLocator(ticks, **kwargs))

def grid(ax:Axes, major=True, minor=True, major_color = 'black', major_alpha = 0.08, minor_color = 'black', minor_alpha=0.03, **kwargs):
    if major: ax.grid(which="major", color=major_color, alpha=major_alpha, **kwargs)
    if minor: ax.grid(which="minor", color=minor_color, alpha=minor_alpha, **kwargs)

def xtickparams(ax:Axes, size=8, rotation=45, **kwargs):
    ax.tick_params(axis="x", labelsize=size, rotation=rotation, **kwargs)

def ytickparams(ax:Axes, size=8, rotation=0, **kwargs):
    ax.tick_params(axis="y", labelsize=size, rotation=rotation, **kwargs)

def xlabel(ax:Axes, label, **kwargs):
    ax.set_xlabel(label, **kwargs)

def ylabel(ax:Axes, label, **kwargs):
    ax.set_ylabel(label, **kwargs)

def title(ax:Axes, title, **kwargs):#pylint:disable=W0621
    ax.set_title(title, **kwargs)