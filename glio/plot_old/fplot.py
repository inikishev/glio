from typing import Optional
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

def flinechart(
    y,
    x=None,
    label=None,
    color=None,
    xlim=None,
    ylim=None,
    xlabel='x',
    ylabel='y',
    title = None,
    maxlegnth=4000,
    showmin=False,
    showmax=False,
    linewidth=0.5,
    majorxticks=35,
    majoryticks=20,
    minorxticks=3,
    minoryticks=3,
    grid=True,
    diff=False,
    movingavg=False,
    movingmed=False,
    ax: Optional[Axes] = None,  # type:ignore

    **kwargs,
):
    # create axe
    if ax is None: ax: Axes = plt.subplots()[1]

    # convert to numpy
    if y is None: y = np.arange(len(x)) # type:ignore
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    elif not isinstance(y, np.ndarray): y = np.asanyarray(y)
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    elif x is not None and not isinstance(x, np.ndarray): x = np.asanyarray(list(x))

    # reduce amount of points
    leny = len(y)
    if leny > maxlegnth:
        every = int(math.ceil(leny/maxlegnth))
        y = y[::every]
        if x is not None: x = x[::every]

    # plot
    if x is None: ax.plot(y, label=label, color=color, linewidth=linewidth, **kwargs)
    else: ax.plot(x, y, label=label, color=color, linewidth=linewidth, **kwargs)

    #xlim
    if xlim is not None:
        if isinstance(xlim, tuple): left, right = xlim
        else: left = right = xlim
        ax.set_xlim(left=left, right=right)

    #ylim
    if ylim is not None:
        if isinstance(xlim, tuple): bot, top = ylim
        else: bot = top = xlim
        ax.set_ylim(bottom=bot, top=top)

    # showmin
    if showmin:
        if x is None: x = list(range(y))
        minx = x[np.argmin(y)]
        miny = y.min()
        ax.text(minx, miny, f"{label} min\nx={minx:.3f}\ny={miny:.3f}", size=6, weight="bold")

    # showmax
    if showmax:
        if x is None: x = list(range(y))
        maxx = x[np.argmax(y)]
        maxy = y.max()
        ax.text(maxx, maxy, f"{label} min\nx={maxx:.3f}\ny={maxy:.3f}", size=6, weight="bold")

    #ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=majorxticks, steps=(1, 2, 2.5, 5, 10)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=majoryticks, steps=(1, 2, 2.5, 5, 10)))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minorxticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minoryticks))
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    # grid
    ax.grid(which="major", color='black', alpha=0.08)
    ax.grid(which="minor", color='black', alpha=0.03)

    # axis labels
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_xlabel(xlabel)

    #title
    if title: ax.set_title(title)
    
    # legend
    leg = ax.legend(prop=dict(size=6), edgecolor=None, )
    leg.get_frame().set_alpha(0.3)
    for line in leg.get_lines():
        line.set_linewidth(3.)