from typing import Optional
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import torch
from . import transforms as TF

_scalartype = (int,float,np.ScalarType)
def _parse_plotdata(data):
    # to numpy if possible
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)) and isinstance(data[0], torch.Tensor): data = [(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t) for t in data]
    
    # numpy
    # determine data format and convert to sequence of (2, N) or (1, N), so (B, 2, N) / (B, 1, N)
    if isinstance(data, np.ndarray):
        # list of values
        if data.ndim == 1: return [[data]] # returns (1, 1, N)

        # either list of x,y pairs, will have shape=(2,N) or (N,2), or list of linecharts, will have a shape of (N, B)
        if data.ndim == 2:
            if data.shape[0] == 2: return [data] # data already in (2, N) format, e.g. ([1,2,3,4,5], [3,2,5,6,2]), return (1,2,N)
            if data.shape[1] == 2: return [data.T] # data in (N, 2) format, e.g. ([1, 3], [2, 4], [3, 5]), transpose and return (1, 2, N)

            # multiple lines assumed to be in (N, B), e.g. ([3,5,6,3,5,7,9,1], [1,3,6,1,1,5,7,8,5], [1,2,1,1,1,7,6,5,2])
            return [[line] for line in data] # return (B, 1 N)

        # list of lists of x,y pairs, will have shape=(B, 2, N) or (B, N, 2)
        if data.ndim == 3:
            if data.shape[1] == 2: return [line for line in data] # data in (B, 2, N), return it as list
            if data.shape[2] == 2: return [line.T for line in data] # data in (B, N, 2), return transpose into (B, 2, N)

        else: raise ValueError(f"Invalid data shape for plotting: {data.shape}")

    # dicts
    if isinstance(data, dict):
        items = list(data.items())
        # first value is a number, so its a dictionary of (key: scalar)
        if isinstance(items[0][1], _scalartype):
            # first key is a number, so its a dictionary of (scalar: scalar)
            if isinstance(items[0][0], _scalartype): return [list(zip(items))] # returns (1, 2, N)
            # otherwise we don't know how to use keys, we use only values
            else: return [[[i[1] for i in items]]] # returns only values, (1, 1, N)

        # else it is a dictionary of separate linecharts
        # dictionary of lists of scalars, return list of lists of scalars
        elif isinstance(items[0][1][0], _scalartype): return [[i[1]] for i in items] # returns (B, 1, N)
        # dictionary of lists of x-y pairs
        elif len(items[0][1][0]) == 2: return [list(zip(i[1])) for i in items] # returns (B, 2, N)
        # dictionary of lists of lists of x-y pairs
        else: return [i[1] for i in items] # returns (B, 2, N)

    # other sequences
    # first value is a scalars, so its a list of scalars
    if isinstance(data[0], _scalartype): return [data] # returns (1, 1, N)

    # first value is a list of scalars, so either list of linecharts or x-y pairs
    if isinstance(data[0][0], _scalartype):
        # list of linecharts, e.g. [[1,2,3,6,4], [5,3,1,5,4,2,1], [1,2,5]]
        if len(data[0]) == 2: return [[i] for i in data] # returns (B, 1, N)
        # list of x-y pairs, e.g. [[1,5], [1,8], [3,3], [7,4]]
        else: return [[i] for i in zip(data)] # returns (1, 2, N)

    # else its a list of lists of whatevers
    else:
        print(data[0][0])
        print(type(data[0][0]))
        # list of lists of x-y pairs, e.g. [[[1,5], [1,8], [3,3], [7,4]], [[1,2], [3,4], [5,6]]]
        if len(data[0][0]) == 2: return [list(zip(i)) for i in data] # returns (B, 2, N)
        # list of lists of x/y, e.g. [[[1,2,3,4],[1,2,2,3]], [[1,2,3,4],[4,5,6,7]]], which is already (B, 2, N)
        else: return data


def linechart(
    y,
    x=None,
    label=None,
    color:Optional[str]=None,
    linewidth=0.5,
    alpha = 1.,
    **kwargs,
    ):
    return Plot().linechart(
        y=y,
        x=x,
        label=label,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        **kwargs
    )


def scatter(
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
    return Plot().scatter(
        x=x,
        y=y,
        s=s,
        c=c,
        label=label,
        color=color,
        marker=marker,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        **kwargs,
    )

def imshow(img, cmap = None, vmin=None, vmax=None, alpha = None, mode = 'auto', allow_alpha=False, **kwargs):
    return Plot().imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, mode=mode, allow_alpha=allow_alpha, **kwargs)

def imshow_batch(x,
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
    return Plot().imshow_batch(
        x=x,
        label=label,
        maxelems=maxelems,
        ncol=ncol,
        nrow=nrow,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        mode=mode,
        allow_alpha=allow_alpha,
        padding=padding,
        normalize=normalize,
        scale_each=scale_each,
        pad_value=pad_value,
        **kwargs,
    )


class Plot:
    def __init__(self):
        self.tfms = []

    def linechart(
        self,
        y,
        x=None,
        label=None,
        color: Optional[str] = None,
        linewidth=0.5,
        alpha=1.0,
        **kwargs,
    ):  # noqa:F811
        def tfm(ax: Axes):
            TF.linechart.linechart(
                ax,
                y=y,
                x=x,
                label=label,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                **kwargs,
            )

        self.tfms.append(tfm)
        return self

    def scatter(self,
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
        def tfm(ax: Axes):
            TF.linechart.scatter(
                ax,
                x=x,
                y=y,
                s=s,
                c=c,
                label=label,
                color=color,
                marker=marker,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                **kwargs,
            )
        self.tfms.append(tfm)
        return self

    def imshow(self, img, cmap = None, vmin=None, vmax=None, alpha = None, mode = 'auto', allow_alpha=False, **kwargs):
        def tfm(ax:Axes): TF.linechart.imshow(ax, img, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, mode=mode, allow_alpha=allow_alpha, **kwargs)
        self.tfms.append(tfm)
        return self

    def imshow_batch(self,
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
        def tfm(ax: Axes):
            TF.linechart.imshow_batch(
                ax,
                x,
                label=label,
                maxelems=maxelems,
                ncol=ncol,
                nrow=nrow,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                mode=mode,
                allow_alpha=allow_alpha,
                padding=padding,
                normalize=normalize,
                scale_each=scale_each,
                pad_value=pad_value,
                **kwargs,
            )

        self.tfms.append(tfm)
        return self

    def show_min(self, target=0, size=6, show_label = True, weight="bold", **kwargs):
        def tfm(ax:Axes): TF.linechart.show_min(ax, target=target, size=size, show_label=show_label, weight=weight, **kwargs)
        self.tfms.append(tfm)
        return self

    def show_max(self, target=0, size=6, show_label = True, weight="bold", **kwargs):
        def tfm(ax:Axes): TF.linechart.show_max(ax, target=target, size=size, show_label=show_label, weight=weight, **kwargs)
        self.tfms.append(tfm)
        return self

    def fill_below(self, target=0, color = None, alpha = 0.3, **kwargs):
        def tfm(ax:Axes): TF.linechart.fill_below(ax, target=target, color=color, alpha=alpha, **kwargs)
        self.tfms.append(tfm)
        return self

    def fill_above(self, target=0, color = None, alpha = 0.3, **kwargs):
        def tfm(ax:Axes): TF.linechart.fill_above(ax, target=target, color=color, alpha=alpha, **kwargs)
        self.tfms.append(tfm)
        return self

    def fill_between(self, target1, target2, color = None, alpha = 0.3, **kwargs):
        def tfm(ax:Axes): TF.linechart.fill_between(ax, target1=target1, target2=target2, color=color, alpha=alpha, **kwargs)
        self.tfms.append(tfm)
        return self

    def differential(self, target = 0, color:Optional[str]=None, linewidth=0.5, order=1, **kwargs,):
        def tfm(ax:Axes): TF.linechart.differential(ax, target = target, color=color, linewidth=linewidth, order=order, **kwargs)
        self.tfms.append(tfm)
        return self

    def moving_average(self, length=0.1, color:Optional[str]=None, linewidth=0.5,**kwargs,):
        def tfm(ax:Axes): TF.linechart.moving_average(ax, length=length, color=color, linewidth=linewidth, **kwargs)
        self.tfms.append(tfm)
        return self

    def moving_median(self, length=0.1, color:Optional[str]=None, linewidth=0.5, **kwargs,):
        def tfm(ax:Axes): TF.linechart.moving_median(ax, length=length, color=color, linewidth=linewidth, **kwargs)
        self.tfms.append(tfm)
        return self

    def legend(self, size=6, edgecolor=None, linewidth=3., frame_alpha = 0.3, **kwargs):
        def tfm(ax:Axes): TF.linechart.legend(ax, size=size, edgecolor=edgecolor, linewidth=linewidth, frame_alpha=frame_alpha, **kwargs)
        self.tfms.append(tfm)
        return self

    def xlim(self, left=None, right=None, **kwargs):
        def tfm(ax:Axes): TF.linechart.xlim(ax, left=left, right=right, **kwargs)
        self.tfms.append(tfm)
        return self

    def ylim(self, bottom=None, top=None, **kwargs):
        def tfm(ax:Axes): TF.linechart.ylim(ax, bottom=bottom, top=top, **kwargs)
        self.tfms.append(tfm)
        return self

    def lim(self, left=None, right=None, bottom=None, top=None, **kwargs):
        def tfm(ax:Axes):
            TF.linechart.xlim(ax, left=left, right=right, **kwargs)
            TF.linechart.ylim(ax, bottom=bottom, top=top, **kwargs)
        self.tfms.append(tfm)
        return self

    def majorxticks(self, ticks = 35, steps=(1, 2, 2.5, 5, 10), **kwargs):
        def tfm(ax:Axes): TF.linechart.majorxticks(ax, ticks = ticks, steps=steps, **kwargs)
        self.tfms.append(tfm)
        return self

    def majoryticks(self, ticks = 20, steps=(1, 2, 2.5, 5, 10), **kwargs):
        def tfm(ax:Axes): TF.linechart.majoryticks(ax, ticks = ticks, steps=steps, **kwargs)
        self.tfms.append(tfm)
        return self

    def minorxticks(self, ticks = 3, **kwargs):
        def tfm(ax:Axes): TF.linechart.minorxticks(ax, ticks = ticks, **kwargs)
        self.tfms.append(tfm)
        return self

    def minoryticks(self, ticks = 3, **kwargs):
        def tfm(ax:Axes): TF.linechart.minoryticks(ax, ticks = ticks, **kwargs)
        self.tfms.append(tfm)
        return self

    def ticks(self, xmajor = 35, ymajor = 20, xminor = 4, yminor = 4, **kwargs):
        def tfm(ax:Axes):
            TF.linechart.majorxticks(ax, ticks = xmajor, **kwargs)
            TF.linechart.majoryticks(ax, ticks = ymajor, **kwargs)
            TF.linechart.minorxticks(ax, ticks = xminor, **kwargs)
            TF.linechart.minoryticks(ax, ticks = yminor, **kwargs)
        self.tfms.append(tfm)
        return self

    def grid(self,major=True, minor=True, major_color = 'black', major_alpha = 0.08, minor_color = 'black', minor_alpha=0.03, **kwargs):
        def tfm(ax:Axes): TF.linechart.grid(ax, major=major, minor=minor, major_color = major_color, major_alpha = major_alpha, minor_color = minor_color, minor_alpha=minor_alpha, **kwargs)
        self.tfms.append(tfm)
        return self

    def xtickparams(self, size=8, rotation=45, **kwargs):
        def tfm(ax:Axes): TF.linechart.xtickparams(ax, size=size, rotation=rotation, **kwargs)
        self.tfms.append(tfm)
        return self

    def ytickparams(self, size=8, rotation = 0, **kwargs):
        def tfm(ax:Axes): TF.linechart.ytickparams(ax, size=size, rotation=rotation, **kwargs)
        self.tfms.append(tfm)
        return self

    def tickparams(self, size=8, xrotation=45, yrotation=0, **kwargs):
        def tfm(ax:Axes):
            TF.linechart.xtickparams(ax, size=size, rotation=xrotation, **kwargs)
            TF.linechart.ytickparams(ax, size=size, rotation=yrotation, **kwargs)
        self.tfms.append(tfm)
        return self

    def xlabel(self, label, **kwargs):
        def tfm(ax:Axes): TF.linechart.xlabel(ax, label, **kwargs)
        self.tfms.append(tfm)
        return self

    def ylabel(self, label, **kwargs):
        def tfm(ax:Axes): TF.linechart.ylabel(ax, label, **kwargs)
        self.tfms.append(tfm)
        return self

    def label(self, xlabel = None, ylabel = None, **kwargs):
        def tfm(ax:Axes):
            if xlabel is not None: TF.linechart.xlabel(ax, xlabel, **kwargs)
            if ylabel is not None: TF.linechart.ylabel(ax, ylabel, **kwargs)
        self.tfms.append(tfm)
        return self

    def title(self, title, **kwargs):
        def tfm(ax:Axes): TF.linechart.title(ax, title, **kwargs)
        self.tfms.append(tfm)
        return self

    def style_chart(self, title = None, xlabel = 'x', ylabel = 'y', show_min=False, show_max=False, diff=False, avg=False, median=False):
        self.ticks()
        self.tickparams()
        self.grid()
        self.label(xlabel, ylabel)
        if title is not None: self.title(title)
        if show_min: self.show_min()
        if show_max: self.show_max()
        if diff: self.differential()
        if avg: self.moving_average()
        if median: self.moving_median()
        self.legend()

class Figure:
    def __init__(self):
        self.plots:list[Plot] = []

    def add(self, plot:Plot):
        self.plots.append(plot)

    def get(self):
        fig, axes = plt.subplots(len(self.plots))
        for ax, plot in zip(axes, self.plots):
            for tfm in plot.tfms:
                tfm(ax)
        return fig, axes

    def show(self):
        self.get()
        plt.show()

def qlinechart(y,
    x=None,
    label=None,
    color:Optional[str]=None,
    linewidth=0.5,
    alpha = 1.,
    **kwargs,):
    fig = Figure()
    fig.add(linechart(y=y, x=x, label=label, color=color, linewidth=linewidth, alpha=alpha, **kwargs))
    return fig.get()

def qscatter(x,
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
    fig = Figure()
    fig.add(scatter(x=x, y=y, s=s, c=c, label=label, color=color, marker=marker, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, **kwargs))
    return fig.get()

def qimshow(img, cmap = None, vmin=None, vmax=None, alpha = None, mode = 'auto', allow_alpha=False, **kwargs):
    fig = Figure()
    fig.add(imshow(img, cmap = cmap, vmin=vmin, vmax=vmax, alpha=alpha, mode=mode, allow_alpha=allow_alpha, **kwargs))
    return fig.get()

def qimshow_batch(
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
    fig = Figure()
    fig.add(imshow_batch(x, label=label, maxelems=maxelems, ncol=ncol, nrow=nrow, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, mode=mode, allow_alpha=allow_alpha, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value, **kwargs))
    return fig.get()
