import math
from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import torch
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MaxNLocator
from torchvision.utils import make_grid

from ._common import (_ax_get_array, _ax_match_target, _img_norm,
                      _img_norm_tensor, _parse_plotdata)

__all__ = [
    'Figure',
    'qimshow',
    'qimshow_batch',
    'qlinechart',
    'qplot',
    'qpath10d',
    'qpath2d',
    'qimshow_grid',
    'qfuncplot1d',
    'qscatter',
    'qtricontour',
    'qtricontourf',
    'qpcolormesh',
]

class Plot:
    def __init__(self, tfms:list[Callable]):
        self.tfms:list[Callable] = tfms

    def __call__(self, ax:Axes):
        for tfm in self.tfms:
            ax = tfm(ax)
        return ax

    def set_ax(self, ax:Axes):
        self.tfms.append(lambda _: ax)
        return self

    @classmethod
    def from_ax(cls, ax:Axes):
        return cls([lambda _: ax])

    def plot(self, *args, label=None, color=None, alpha=None, linewidth=None, linestyle = None, xlim=None, ylim=None, **kwargs) -> "Plot":
        def plot(ax:Axes) -> Axes:
            ax.plot(*args, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle = linestyle, **kwargs)
            return ax
        self.tfms.append(plot)
        if xlim is not None: self.xlim(*xlim)
        if ylim is not None: self.ylim(*ylim)
        return self

    def autoplot(self, data, labels=None, color=None, alpha=None, linewidth=None, xlim=None, ylim=None, **kwargs) -> "Plot":
        data = _parse_plotdata(data)
        if isinstance(labels, (int,float,str)): labels = (labels,)
        if labels is None: labels = []
        for i, d in enumerate(data):
            if len(labels) > i: label = labels[i]
            else: label = None
            self.plot(*d, label=label, color=color, alpha=alpha, linewidth=linewidth, xlim=xlim, ylim=ylim, **kwargs)
        return self

    def linechart(self, x, y=None, label=None, color=None, alpha=None, linewidth=None, linestyle= None, xlim=None, ylim=None,**kwargs,) -> "Plot":
        # convert y to numpy
        if y is None:
            y = x
            x = None
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
        elif not isinstance(y, np.ndarray): y = np.array(y)

        # convert x to numpy
        if x is None: x = np.arange(len(y))
        elif isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray): x = np.array(x)

        return self.plot(x, y, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, xlim=xlim, ylim=ylim, **kwargs)

    def funcplot1d(
        self,
        func,
        start,
        stop,
        num = None,
        step = None,
        vectorize:Optional[Literal['list', 'numpy', 'torch']] = None,
        ylim=None,
        label=None,
        color=None,
        alpha=None,
        linewidth=None,
        linestyle=None,
        **kwargs,
    ):
        if step is None: step = (stop - start) / num
        if vectorize == 'list': values = func(np.arange(start, stop, step).tolist())
        elif vectorize == 'numpy': values = func(np.arange(start, stop, step))
        elif vectorize == 'torch': values = func(torch.arange(start, stop, step))
        else: values = [func(v) for v in torch.arange(start, stop, step)]
        return self.linechart(np.arange(start, stop, step), values, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, ylim=ylim)

    def scatter(self,
                x,
                y = None,
                s = None,
                c = None,
                label=None,
                marker = None,
                cmap = None,
                vmin=None,
                vmax=None,
                alpha = None,
                xlabel = None,
                ylabel = None,
                norm = None,
                **kwargs) -> "Plot":
        # convert y to numpy
        if y is None: 
            if len(x[0]) == 2: x, y = zip(*x)
            else: y = np.arange(len(x))
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
        elif not isinstance(y, np.ndarray): y = np.array(y)

        # convert x to numpy
        if x is None: x = np.arange(len(y))
        elif isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray): x = np.array(x)

        if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
        if isinstance(c, torch.Tensor): c = c.detach().cpu().numpy()

        def scatter(ax:Axes) -> Axes:
            ax.scatter(x, y, s=s, c=c, label=label, marker=marker,cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm, **kwargs)
            return ax
        self.tfms.append(scatter)

        if xlabel: self.xlabel(xlabel)
        if ylabel: self.ylabel(ylabel)
        return self

    def autoscatter(self, data, label, **kwargs) -> "Plot":
        data = _parse_plotdata(data)
        for d in data:
            self.scatter(*d, label=label, **kwargs)
        return self

    def imshow(self,
        x,
        label = None,
        cmap:Optional[str | Any] = 'gray',
        vmin=None,
        vmax=None,
        alpha = None,
        mode = 'auto',
        allow_alpha=False,
        **kwargs,) -> "Plot":
        # convert to numpy
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray): x = np.array(x)
        x = _img_norm(x,vmin,vmax)

        # determine mode
        if mode == "auto":
            if x.ndim > 3: mode = "*c*"
            if x.ndim == 3 and x.shape[2] > 4 and x.shape[0] < x.shape[2]: mode = 'c*'

        # if batched, take central element
        if mode == "*c*":
            while x.ndim > 3: x = x[int(x.shape[0]//2)]
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

        def imshow(ax:Axes) -> Axes:
            ax.imshow(x, label=label, cmap=cmap, alpha=alpha, **kwargs)
            return ax
        self.tfms.append(imshow)
        return self

    #def mask(x)

    def imshow_batch(self,
        x,
        label=None,
        maxelems:Optional[int] = 256,
        ncol:Optional[int|float] = None,
        nrow:Optional[int|float] = 0.5,
        cmap = 'gray',
        vmin=None,
        vmax=None,
        alpha = None,
        mode = 'auto',
        allow_alpha=False,
        padding=2,
        normalize=True,
        scale_each = False,
        pad_value = 'mean',
        **kwargs,
        ) -> "Plot":
        # convert to cpu tensor
        if isinstance(x, torch.Tensor): x = x.detach().cpu()
        elif not isinstance(x, (np.ndarray,torch.Tensor)): x = torch.from_numpy(np.array(x))
        x = _img_norm_tensor(x,vmin,vmax)

        # determine mode
        if mode == "auto":
            while x.ndim > 4: x = x[int(x.shape[0]//2)]
            if x.ndim == 4:
                if x.shape[1] > 4 and x.shape[3] < x.shape[1]: mode = 'b*c'
                else: mode = "bc*"
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
                x = x[:,:3]
        elif x.ndim == 3:
            # add channel dim
            x = x.unsqueeze(1)

        # make grid
        # determine nrow
        nelem = x.shape[0]
        if ncol is None:
            if nrow is None:
                # automatically
                grid_rows = max(1, int(math.ceil(nelem**0.5)))
            elif isinstance(nrow, float):
                grid_rows = max(1, int(math.ceil(nelem**nrow)))
            else: grid_rows = nrow
        # determine from ncol
        else:
            if isinstance(ncol, float):
                grid_cols = max(1, int(math.ceil(nelem**ncol)))
                grid_rows = max(1, int(math.ceil(nelem/grid_cols)))
            else:
                grid_rows = int(math.ceil(nelem/ncol))
        # pad value
        if pad_value == 'min': pad_value = x.min()
        elif pad_value == 'max': pad_value = x.max()
        elif pad_value == 'mean': pad_value = x.mean()
        # grid
        grid = make_grid(x, nrow=grid_rows, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value).permute(1,2,0) # pylint:disable=W0621 # type:ignore

        # we get HWC, now imshow
        def imshow_batch(ax:Axes) -> Axes:
            ax.imshow(grid, label=label, cmap=cmap, alpha=alpha, **kwargs)
            return ax
        self.tfms.append(imshow_batch)
        return self

    def seg_overlay(
        self,
        x: torch.Tensor | np.ndarray,
        mode="auto",
        ignore_bg=True,
        alpha=0.3,
        colors=('red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white'),
        bg_alpha:float = 0.,
        **kwargs,
    ):
        """Overlay up to 7 classes of segmentation, where each pixel is has only one class."""
        if isinstance(x, torch.Tensor): x = x.detach().cpu()
        elif not isinstance(x, torch.Tensor): x = torch.from_numpy(np.array(x))
        # argmax if not argmaxed
        if x.ndim == 3:
            if mode == 'auto':
                if x.shape[0] < x.shape[2]: mode = 'c*'
                else: mode = '*c'
            if mode == 'c*': x = x.argmax(0)
            else: x = x.argmax(-1)


        # 3 channel image
        segm = torch.zeros((4, *x.shape), dtype=torch.uint8)
        for ch, color in zip(range(int(ignore_bg), len(colors) + int(ignore_bg)), colors):
            if color == 'red': segm[0] = torch.where(x == ch, torch.tensor(255), segm[0])
            elif color == 'green': segm[1] = torch.where(x == ch, torch.tensor(255), segm[1])
            elif color == 'blue': segm[2] = torch.where(x == ch, torch.tensor(255), segm[2])

            # compound colors
            elif color == 'yellow': segm[:2] = torch.where(x == ch, torch.tensor(255), segm[:2])
            elif color == 'cyan': segm[1:3] = torch.where(x == ch, torch.tensor(255), segm[1:3])

            # last compound color
            elif color == 'magenta':
                segm[0] = torch.where(x == ch, torch.tensor(255), segm[0])
                segm[2] = torch.where(x == ch, torch.tensor(255), segm[2])

            # last is white
            elif color == 'white': segm[:3] = torch.where(x == ch, torch.tensor(255), segm[:3])
            #print(segm)
            else: raise ValueError(f'invalid color {color} in {colors = }')

        segm[3] = torch.where(segm[:3].amax(0) > torch.tensor(0), torch.tensor(int(alpha*255)), torch.tensor(bg_alpha))

        def seg_overlay(ax:Axes) -> Axes:
            ax.imshow(segm.permute(1,2,0), **kwargs)
            return ax
        self.tfms.append(seg_overlay)
        return self


    def path(self,
                x,
                y,
                s = None,
                c = None,
                edgecolors = None,
                linewidths = None,
                linecolor:Optional[str] = None,
                linewidth = None,
                label=None,
                marker = None,
                cmap = None,
                line_alpha = None,
                marker_alpha = None,
                **kwargs) -> "Plot":
        # convert y to numpy
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
        elif not isinstance(y, np.ndarray): y = np.array(y)

        # convert x to numpy
        elif isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray): x = np.array(x)

        self.plot(x,y, label=label, color=linecolor, alpha=line_alpha, linewidth=linewidth, **kwargs)
        self.scatter(x, y, s=s, c=c, label=label, marker=marker, cmap=cmap, alpha=marker_alpha,linewidths=linewidths,edgecolors=edgecolors, **kwargs)

        return self

    def path2d(self,
            data,
            label=None,
            c=None,
            s=None,
            marker = None,
            cmap:Optional[str] = 'gnuplot2',
            linecolor:Optional[str] = 'black',
            linewidth = 0.66,
            line_alpha = 0.5,
            marker_alpha = 0.5,
            det=False,
            **kwargs) -> "Plot":
        from ._common import _prepare_scatter2d_data
        d = _prepare_scatter2d_data(data, det=det)
        x, y = d["x"], d["y"]
        if cmap is not None and c is None: c = np.linspace(0, 1, len(x))
        return self.path(x, y, c=c, s=s, label=label, linecolor=linecolor, marker=marker, cmap=cmap, line_alpha=line_alpha, marker_alpha=marker_alpha, linewidth=linewidth, **kwargs)

    def path10d(self, data, label=None,
                linecolor = 'black',
                marker = None,
                cmap = 'gnuplot2',
                line_alpha = 0.5,
                marker_alpha = 0.5,
                linewidth = 0.66,) -> "Plot":
        from ._common import _prepare_scatter10d_data
        d = _prepare_scatter10d_data(data)
        x, y, c, s, linewidths, edgecolors = d["x"], d["y"], d["c"], d["s"], d["linewidths"], d["edgecolors"]
        return self.path(x, y, s=s, c=c, linewidths=linewidths, edgecolors=edgecolors, label=label, linecolor=linecolor, marker=marker, cmap=cmap, line_alpha=line_alpha, marker_alpha=marker_alpha, linewidth=linewidth)

    def vline(self, x, ymin = None, ymax = None, linewidth = None, linestyle = None, color=None, **kwargs) -> "Plot":
        def vline(ax:Axes) -> Axes:
            if ymin is not None and ymax is not None:
                ax.axvline(x, ymin=ymin, ymax=ymax, linewidth=linewidth, linestyle=linestyle, color=color, **kwargs)
            else:
                ax.axline((x, 0), (x, 1), linewidth=linewidth, linestyle=linestyle, color=color, **kwargs)
            return ax
        self.tfms.append(vline)
        return self

    def hline(self, y, xmin = None, xmax = None, linewidth = None, linestyle = None, color=None, **kwargs) -> "Plot":
        def hline(ax:Axes) -> Axes:
            if xmin is not None and xmax is not None:
                ax.axhline(y, xmin=xmin, xmax=xmax, linewidth=linewidth, linestyle=linestyle, color=color, **kwargs)
            else:
                ax.axline((0, y), (1, y), linewidth=linewidth, linestyle=linestyle, color=color, **kwargs)
            return ax
        self.tfms.append(hline)
        return self

    def line(self, xy1: tuple[float, float],xy2: tuple[float, float] | None = None, slope = None, **kwargs):
        def line(ax:Axes) -> Axes:
            ax.axline(xy1, xy2, slope=slope, **kwargs)
            return ax
        self.tfms.append(line)
        return self

    def origin_lines(self, linewidth = 0.5, color='black', coords = (0,0), **kwargs) -> "Plot":
        self.vline(coords[0], linewidth=linewidth, color=color, **kwargs)
        return self.hline(coords[1], linewidth=linewidth, color=color, **kwargs)

    def point(self, x, y, s=None, c=None, alpha=None, marker = '.', **kwargs):
        def point(ax:Axes) -> Axes:
            ax.scatter(x, y, s=s, c=c, alpha=alpha, marker=marker, **kwargs)
            return ax
        self.tfms.append(point)
        return self

    def tricontour(self, x, y, z, levels=10, cmap=None, **kwargs):
        def tricontour(ax:Axes) -> Axes:
            ax.tricontour(x, y, z, levels=levels, cmap=cmap, **kwargs)
            return ax
        self.tfms.append(tricontour)
        return self

    def tricontourf(self, x, y, z, levels=1000, cmap=None, vmin=None, vmax=None, **kwargs):
        def tricontourf(ax:Axes) -> Axes:
            ax.tricontourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            return ax
        self.tfms.append(tricontourf)
        return self

    def _get_grid(self, x, y, z, mode:Literal["linear", "nearest", "clough"], xlim, ylim, zlim, step):
        from scipy.interpolate import (CloughTocher2DInterpolator,
                                       LinearNDInterpolator,
                                       NearestNDInterpolator)
        INTERPOLATORS = {
            'linear': LinearNDInterpolator,
            'nearest': NearestNDInterpolator,
            'clough': CloughTocher2DInterpolator
        }
        if mode not in INTERPOLATORS: raise ValueError(f'Invalid mode {mode}')
        xmin, xmax = np.min(x), np.max(x)
        if xlim is not None:
            if xlim[0] is not None: xmin = xlim[0]
            if xlim[1] is not None: xmax = xlim[1]
        ymin, ymax = np.min(y), np.max(y)
        if ylim is not None:
            if ylim[0] is not None: ymin = ylim[0]
            if ylim[1] is not None: ymax = ylim[1]
        if zlim is not None:
            z = np.clip(z, *zlim)
        x_grid = np.linspace(xmin, xmax, step)
        y_grid = np.linspace(ymin, ymax, step)
        X, Y = np.meshgrid(x_grid,y_grid)
        interpolator = INTERPOLATORS[mode]((x, y), z)
        Z = interpolator(X, Y)
        return X, Y, Z

    def contour(
        self,
        x,
        y,
        z,
        levels=10,
        step=300,
        cmap=None,
        mode: Literal["linear", "nearest", "clough"] = "linear",
        xlim=None,
        ylim=None,
        zlim=None,
        alpha=None,
        norm = None,
    ):
        X, Y, Z = self._get_grid(x=x, y=y, z=z, mode=mode, xlim=xlim, ylim=ylim, zlim=zlim, step=step)
        def contour(ax:Axes) -> Axes:
            ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha = alpha, norm = norm)
            return ax
        self.tfms.append(contour)
        return self

    def contourf(self, x, y, z, levels=10, step = 300, cmap = None, mode:Literal["linear", "nearest", "clough"] = 'linear', xlim = None, ylim = None, zlim = None, alpha = None):
        X, Y, Z = self._get_grid(x=x, y=y, z=z, mode=mode, xlim=xlim, ylim=ylim, zlim=zlim, step=step)
        def pcolormesh(ax:Axes) -> Axes:
            ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha = alpha)
            return ax
        self.tfms.append(pcolormesh)
        return self

    def pcolormesh(
        self,
        x,
        y,
        z,
        step=300,
        cmap=None,
        contour=False,
        contour_cmap="binary",
        contour_levels=10,
        contour_alpha=0.5,
        mode:Literal["linear", "nearest", "clough"]="linear",
        xlim=None,
        ylim=None,
        zlim=None,
        alpha=None,
        shading=None,
        norm = None,
        antialiased: bool = True,
    ):
        X, Y, Z = self._get_grid(x=x, y=y, z=z, mode=mode, xlim=xlim, ylim=ylim, zlim=zlim, step=step)
        def pcolormesh(ax:Axes) -> Axes:
            ax.pcolormesh(X, Y, Z, cmap=cmap, alpha = alpha, shading = shading, antialiased = antialiased, zorder=0, norm=norm)
            return ax
        self.tfms.append(pcolormesh)
        if contour: self.contour(x,y,z, levels=contour_levels, step=step, cmap=contour_cmap, mode=mode, alpha=contour_alpha, xlim=xlim, ylim=ylim, zlim=zlim, norm=norm)
        return self


    def colorbar(self, location = None, orientation = None, fraction=0.15, shrink = 1., aspect = 20.):
        def colorbar(ax:Axes) -> Axes:
            plt.colorbar(ax.collections[0], location=location,orientation=orientation, fraction=fraction, shrink=shrink, aspect=aspect)
            #ax.get_figure().colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax, location=location, orientation=orientation, fraction=fraction, shrink=shrink, aspect=aspect) # type:ignore
            return ax
        self.tfms.append(colorbar)
        return self

    def legend(self, size=6, edgecolor=None, linewidth=3., frame_alpha = 0.3, **kwargs) -> "Plot":
        def legend(ax:Axes) -> Axes:
            if 'prop' in kwargs: prop = kwargs["prop"]
            else: prop = {}
            if size is not None: prop['size'] = size

            leg = ax.legend(prop=prop, edgecolor=edgecolor, **kwargs)
            leg.get_frame().set_alpha(frame_alpha)

            if linewidth is not None:
                for line in leg.get_lines():
                    line.set_linewidth(linewidth)

            return ax
        self.tfms.append(legend)
        return self

    def xlim(self,left = None, right = None, **kwargs) -> "Plot":
        def xlim(ax:Axes) -> Axes:
            ax.set_xlim(left=left, right=right, **kwargs)
            return ax
        self.tfms.append(xlim)
        return self

    def ylim(self,bottom=None,top=None, **kwargs) -> "Plot":
        def ylim(ax:Axes) -> Axes:
            ax.set_ylim(bottom=bottom, top=top, **kwargs)
            return ax
        self.tfms.append(ylim)
        return self

    def lim(self, xlim = None, ylim = None, **kwargs) -> "Plot":
        if xlim is not None: self.xlim(*xlim, **kwargs)
        if ylim is not None: self.ylim(*ylim, **kwargs)
        return self

    def majorxticks(self, ticks = "auto") -> "Plot":
        def majorxticks(ax:Axes) -> Axes:
            if ticks == "auto": ax.xaxis.set_major_locator(AutoLocator())
            else: raise NotImplementedError
            return ax
        self.tfms.append(majorxticks)
        return self
    def majoryticks(self, ticks = "auto") -> "Plot":
        def majoryticks(ax:Axes) -> Axes:
            if ticks == "auto": ax.xaxis.set_major_locator(AutoLocator())
            else: raise NotImplementedError
            return ax
        self.tfms.append(majoryticks)
        return self
    def minorxticks(self, ticks = 'auto', **kwargs) -> "Plot":
        def minorxticks(ax:Axes) -> Axes:
            ax.xaxis.set_minor_locator(AutoMinorLocator(ticks, **kwargs)) # type:ignore
            return ax
        self.tfms.append(minorxticks)
        return self
    def minoryticks(self, ticks = 'auto', **kwargs) -> "Plot":
        def minoryticks(ax:Axes) -> Axes:
            ax.yaxis.set_minor_locator(AutoMinorLocator(ticks, **kwargs)) # type:ignore
            return ax
        self.tfms.append(minoryticks)
        return self

    def ticks(self, xmajor = 'auto', ymajor = 'auto', xminor = 'auto', yminor = 'auto', **kwargs) -> "Plot":
        if xmajor == 'auto': self.majorxticks()
        if ymajor == 'auto': self.majoryticks()
        if xminor is not None: self.minorxticks(ticks = xminor, **kwargs)
        if yminor is not None: self.minoryticks(ticks = yminor, **kwargs)
        return self

    def grid(self, major=True, minor=True, major_color = 'black', major_alpha = 0.08, minor_color = 'black', minor_alpha=0.03, **kwargs) -> "Plot":
        def grid(ax:Axes) -> Axes:
            if major: ax.grid(which="major", color=major_color, alpha=major_alpha, **kwargs)
            if minor: ax.grid(which="minor", color=minor_color, alpha=minor_alpha, **kwargs)
            return ax
        self.tfms.append(grid)
        return self


    def xtickparams(self, size=8, rotation=45, **kwargs) -> "Plot":
        def xtickparams(ax:Axes) -> Axes:
            ax.tick_params(axis="x", labelsize=size, rotation=rotation, **kwargs)
            return ax
        self.tfms.append(xtickparams)
        return self


    def ytickparams(self, size=8, rotation=0, **kwargs) -> "Plot":
        def ytickparams(ax:Axes) -> Axes:
            ax.tick_params(axis="y", labelsize=size, rotation=rotation, **kwargs)
            return ax
        self.tfms.append(ytickparams)
        return self

    def tickparams(self, size=8, xrotation=45, yrotation=0, **kwargs):
        self.xtickparams(size, xrotation, **kwargs)
        self.ytickparams(size, yrotation, **kwargs)
        return self


    def xlabel(self, label, **kwargs) -> "Plot":
        def xlabel(ax:Axes) -> Axes:
            ax.set_xlabel(label, **kwargs)
            return ax
        self.tfms.append(xlabel)
        return self


    def ylabel(self, label, **kwargs) -> "Plot":
        def ylabel(ax:Axes) -> Axes:
            ax.set_ylabel(label, **kwargs)
            return ax
        self.tfms.append(ylabel)
        return self

    def axoff(self):
        def axoff(ax:Axes) -> Axes:
            ax.set_axis_off()
            return ax
        self.tfms.append(axoff)
        return self

    def axlabels(self, xlabel = None, ylabel=None, fontsize = None) -> "Plot":
        if xlabel is not None: self.xlabel(xlabel, fontsize=fontsize)
        if ylabel is not None: self.ylabel(ylabel, fontsize=fontsize)
        return self

    def title(self, title, **kwargs) -> "Plot":
        def title_(ax:Axes) -> Axes:
            ax.set_title(title, **kwargs)
            return ax
        self.tfms.append(title_)
        return self

    def xscale(self, scale, **kwargs):
        def xscale(ax:Axes) -> Axes:
            ax.set_xscale(scale, **kwargs)
            return ax
        self.tfms.append(xscale)
        return self
    def yscale(self, scale, **kwargs):
        def yscale(ax:Axes) -> Axes:
            ax.set_yscale(scale, **kwargs)
            return ax
        self.tfms.append(yscale)
        return self
    def scale(self, xscale, yscale, **kwargs):
        self.xscale(xscale, **kwargs)
        self.yscale(yscale, **kwargs)
        return self

    def show_min(self, target = 0, size=6, show_label = True, weight="bold", **kwargs) -> "Plot":
        def show_min(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target)
            for artist in artists:
                data = _ax_get_array(artist)
                # data is x,y tuple - for line and scatter
                if isinstance(data, (list,tuple,np.ndarray)):
                    x, y = data
                    minx, miny = x[np.argmin(y)], y.min()
                # image
                else:
                    raise NotImplementedError(type(data), data)

                #label
                if show_label:
                    label = f'{artist.get_label()} min\n'
                    if label.startswith("_"): label='min\n'
                else: label = ''
                # put the text
                ax.text(minx, miny, f"{label}x={minx:.3f}\ny={miny:.3f}", size=size, weight=weight, **kwargs)
            return ax
        self.tfms.append(show_min)
        return self

    def show_max(self, target = 0, size=6, show_label = True, weight="bold", **kwargs) -> "Plot":
        def show_max(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target)
            for artist in artists:
                data = _ax_get_array(artist)
                # data is x,y tuple - for line and scatter
                if isinstance(data, (list,tuple,np.ndarray)):
                    x, y = data
                    maxx, maxy = x[np.argmax(y)], y.max()
                # image
                else:
                    raise NotImplementedError(type(data), data)

                #label
                if show_label:
                    label = f'{artist.get_label()} max\n'
                    if label.startswith("_"): label='max\n'
                else: label = ''
                # put the text
                ax.text(maxx, maxy, f"{label}x={maxx:.3f}\ny={maxy:.3f}", size=size, weight=weight, **kwargs)
            return ax
        self.tfms.append(show_max)
        return self


    def differential(self, target = 0, color:Optional[str]=None, linewidth=0.5, order=1, **kwargs,) -> "Plot":
        def differential(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target, Line2D)
            for artist in artists:
                x, y = _ax_get_array(artist)
                diff = y
                for _ in range(order): diff = np.diff(diff)
                if order == 1: label = f'{artist.get_label()} diff'
                else: label = f'{artist.get_label()} diff {order}'
                ax.plot(x[:-1], diff, label = label, color=color, linewidth=linewidth, **kwargs)
            return ax
        self.tfms.append(differential)
        return self


    def moving_average(self, target = 0, length=0.1, color:Optional[str]=None, linewidth=0.5,**kwargs,) -> "Plot":
        def moving_average(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target, Line2D)
            for artist in artists:
                x, y = _ax_get_array(artist)
                if isinstance(length, float): ma_length = int(len(x) * length)
                else: ma_length = length
                ma = scipy.signal.convolve(y, np.ones(ma_length)/ma_length, 'same')
                ax.plot(x, ma, label=f'{artist.get_label()} mean {ma_length}', color=color, linewidth=linewidth, **kwargs)
            return ax
        self.tfms.append(moving_average)
        return self


    def moving_median(self, target = 0, length=0.1, color:Optional[str]=None, linewidth=0.5,**kwargs,) -> "Plot":
        def moving_median(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target, Line2D)
            for artist in artists:
                x, y = _ax_get_array(artist)
                if isinstance(length, float): ma_length = int(len(x) * length)
                else: ma_length = length
                ma = scipy.ndimage.median_filter(y, ma_length, mode='nearest')
                ax.plot(x, ma, label=f'{artist.get_label()} median {ma_length}', color=color, linewidth=linewidth, **kwargs)
            return ax
        self.tfms.append(moving_median)
        return self


    def fill_below(self, target = 0, color = None, alpha = 0.3, **kwargs) -> "Plot":
        def fill_below(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target, types = (Line2D, PathCollection))
            for artist in artists:
                x, y = _ax_get_array(artist)
                ax.fill_between(x, ax.get_ylim()[0], y, alpha=alpha, color=color, **kwargs)
            return ax
        self.tfms.append(fill_below)
        return self


    def fill_above(self, target = 0, color = None, alpha = 0.3, **kwargs) -> "Plot":
        def fill_above(ax:Axes) -> Axes:
            artists = _ax_match_target(ax.get_children(), target, types = (Line2D, PathCollection))
            for artist in artists:
                x, y = _ax_get_array(artist)
                ax.fill_between(x, ax.get_ylim()[1], y, alpha=alpha, color=color, **kwargs)
            return ax
        self.tfms.append(fill_above)
        return self


    def fill_between(self, target1, target2, color = None, alpha = 0.3, **kwargs) -> "Plot":
        def fill_between(ax:Axes) -> Axes:
            artists1 = _ax_match_target(ax.get_children(), target1, types = (Line2D, PathCollection))
            artists2 = _ax_match_target(ax.get_children(), target2, types = (Line2D, PathCollection))
            for artist1, artist2 in zip(artists1, artists2):
                x, y1 = _ax_get_array(artist1)
                _, y2 = _ax_get_array(artist2)
                ax.fill_between(x, y1, y2, alpha=alpha, color=color, **kwargs)
            return ax
        self.tfms.append(fill_between)
        return self

    def style_chart(
        self,
        title=None,
        xlabel: Optional[Any] = "x",
        ylabel: Optional[Any] = "y",
        show_min=False,
        show_max=False,
        diff=False,
        avg=False,
        median=False,
        legend_size=6,
        xscale = None,
        yscale = None,
        xlim = None,
        ylim = None,
    ):
        if xlim is not None: self.xlim(xlim)
        if ylim is not None: self.ylim(ylim)
        if xscale is not None: self.xscale(xscale)
        if yscale is not None: self.yscale(yscale)
        if xscale != 'log': self.ticks()
        if xscale != 'log': self.tickparams()
        self.grid()
        self.axlabels(xlabel, ylabel)
        if title is not None: self.title(title)
        if show_min: self.show_min()
        if show_max: self.show_max()
        if diff: self.differential()
        if avg: self.moving_average()
        if median: self.moving_median()
        self.legend(size=legend_size)
        return self

    def style_img(self, title = None, xlabel:Optional[Any] = None, ylabel:Optional[Any] = None, axes=False):
        if (xlabel is not None) or (ylabel is not None): self.axlabels(xlabel, ylabel)
        if title is not None: self.title(title)
        if not axes: self.axoff()
        return self

class Figure:
    def __init__(self):
        self.plots: list[Plot] = []
        self.cur = 0

    def add(self):
        self.plots.append(Plot([]))
        return self.plots[-1]

    def get(self, loc:int = -1):
        return self.plots[loc]

    def create(self, nrow:Optional[int|float] = None, ncol:Optional[int|float] = None, figsize = None, layout="compressed", title=None, **kwargs):

        # determine nrow
        nelem = len(self.plots)
        if nelem == 0: return
        if nrow is None:
            if ncol is None:
                if isinstance(figsize, int): figsize = (figsize, figsize)
                fsize = (1, 1) if figsize is None else figsize
                nrow = int(math.ceil(nelem ** (fsize[1] / sum(fsize))))
                ncol = nelem / nrow # type:ignore
            else:
                if isinstance(ncol, float): ncol = int(math.ceil(nelem**ncol))
                nrow = nelem / ncol # type:ignore
        elif ncol is None:
            if isinstance(nrow, float): nrow = int(math.ceil(nelem**nrow))
            ncol = nelem / nrow # type:ignore

        # make sure nrow*ncol is equal or higher than nelem
        if nrow*ncol < nelem: nrow = nelem / ncol # type:ignore

        # create figure
        if isinstance(figsize, int): figsize = (int(math.ceil(ncol*figsize*1.66)), int(math.ceil(nrow*figsize*0.66))) # type:ignore
        nrow = max(1, nrow)
        ncol = max(1, ncol)
        self.fig, self.axes = plt.subplots(int(math.ceil(nrow)), int(math.ceil(ncol)), figsize=figsize, layout=layout, **kwargs) # type:ignore
        if isinstance(self.axes, Axes): self.axes = np.array([self.axes])

        # plot
        for i,ax in enumerate(self.axes.ravel()):
            if i<len(self.plots): self.plots[i](ax)
            else:
                ax.set_axis_off()
                ax.set_frame_on(False)

        if title is not None: self.fig.suptitle(title)
        return self.fig, self.axes

    def show(self, nrow = None, ncol = None, figsize = None, layout="compressed", title=None, **kwargs):
        self.create(nrow, ncol, figsize=figsize, layout=layout, title=title)
        plt.show(**kwargs)
        if hasattr(self, 'fig'): self.fig.canvas.draw()

    def clear_data(self):
        self.plots:list[Plot] = []

    def savefig(self, path):
        if hasattr(self, "fig"): self.fig.savefig(path, bbox_inches='tight', pad_inches=0)

    def close(self):
        if hasattr(self, 'fig'): plt.close(self.fig)

def _create_fig(ax) -> tuple[Figure, Plot]:
    """Creates a figure and a plot on it, optionally from ax. Returns (Figure, Plot)"""
    fig = Figure()
    if ax is None: return fig, fig.add()
    return fig, fig.add().set_ax(ax)

def qplot(
    data,
    labels=None,
    color=None,
    alpha=None,
    linewidth=None,
    xlim=None,
    ylim=None,
    xlabel='x',
    ylabel='y',
    xscale=None,
    yscale=None,
    title=None,
    ax=None,
    show=False,
    **kwargs,
):
    fig, plot = _create_fig(ax)
    (
    plot
    .autoplot(data=data, labels=labels, color=color, alpha=alpha, linewidth=linewidth, xlim=xlim, ylim=ylim, **kwargs)
    .style_chart(xlabel=xlabel,ylabel=ylabel,title=title,xscale=xscale,yscale=yscale)
    )
    if show: fig.show()
    else: fig.create()
    return fig


def qlinechart(
    x,
    y=None,
    label=None,
    color=None,
    alpha=None,
    linewidth=None,
    linestyle=None,
    xlim=None,
    ylim=None,
    xlabel='x',
    ylabel='y',
    xscale=None,
    yscale=None,
    title=None,
    ax=None,
    show=False,
    **kwargs,
):
    fig, plot = _create_fig(ax)
    (
    plot
    .linechart(x=x, y=y, label=label, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, xlim=xlim, ylim=ylim, **kwargs)
    .style_chart(xlabel=xlabel,ylabel=ylabel,title=title,xscale=xscale,yscale=yscale)
    )
    if show: fig.show()
    else: fig.create()
    return fig


def qpath2d(
    data,
    label=None,
    c:Optional[str]=None,
    s = None,
    marker=None,
    cmap:Optional[str]="gnuplot2",
    linecolor:Optional[str]='black',
    line_alpha=0.5,
    marker_alpha=0.5,
    linewidth=0.66,
    det=False,
    xlim=None,
    ylim=None,
    xlabel:Optional[str]='x',
    ylabel:Optional[str]='y',
    title=None,
    figsize=None,
    ax=None,
    show=False,
    **kwargs,
    ):
    fig, plot = _create_fig(ax)
    (
    plot
    .path2d(data=data, label=label, c=c, s=s, marker=marker, cmap=cmap, linecolor=linecolor, line_alpha=line_alpha, marker_alpha=marker_alpha, linewidth=linewidth, det=det, **kwargs)
    .lim(xlim, ylim)
    .style_chart(xlabel=xlabel, ylabel=ylabel, title=title)
    )
    if show: fig.show(figsize=figsize)
    else: fig.create(figsize=figsize)
    return fig

def qpath10d(
    data,
    label=None,
    linecolor="black",
    marker=None,
    cmap="gnuplot2",
    line_alpha=0.5,
    marker_alpha=0.5,
    linewidth=0.66,
    xlim=None,
    ylim=None,
    xlabel='x',
    ylabel='y',
    title=None,
    figsize=None,
    ax=None,
    show=False,
    ):
    fig, plot = _create_fig(ax)
    (
    plot
    .path10d(data=data, label=label, linecolor=linecolor, marker=marker, cmap=cmap, line_alpha=line_alpha, marker_alpha=marker_alpha, linewidth=linewidth)
    .lim(xlim, ylim)
    .style_chart(xlabel=xlabel, ylabel=ylabel, title=title)
    )
    if show: fig.show(figsize=figsize)
    else: fig.create(figsize=figsize)
    return fig

def qimshow(x,
        title = None,
        cmap = 'gray',
        vmin=None,
        vmax=None,
        alpha = None,
        mode = 'auto',
        allow_alpha=False,
        xlabel='x',
        ylabel='y',
        figsize=None,
        ax=None,
        show=False,
        **kwargs,):
    fig, plot = _create_fig(ax)
    (
    plot
    .imshow(x=x, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, mode=mode, allow_alpha=allow_alpha, **kwargs)
    .style_img(title=title, xlabel=xlabel, ylabel=ylabel)
    )
    if show: fig.show(figsize=figsize)
    else: fig.create(figsize=figsize)
    return fig


def qimshow_batch(x,
        title = None,
        maxelems = 256,
        ncol = None,
        nrow = None,
        cmap = 'gray',
        vmin=None,
        vmax=None,
        alpha = None,
        mode = 'auto',
        allow_alpha=False,
        padding=2,
        normalize=True,
        scale_each = False,
        pad_value = 'min',
        xlabel='x',
        ylabel='y',
        figsize=None,
        ax=None,
        show=False,
        **kwargs,):
    fig, plot = _create_fig(ax)
    (
    plot
    .imshow_batch(x=x, maxelems=maxelems, ncol=ncol, nrow=nrow, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, mode=mode, allow_alpha=allow_alpha, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value, **kwargs)
    .style_img(title=title, xlabel=xlabel, ylabel=ylabel)
    )
    if show: fig.show(figsize=figsize)
    else: fig.create(figsize=figsize)
    return fig


def qimshow_grid(images,
                 labels = None,
                 maxelems = 256,
                 ncol = None,
                 nrow = None,
                 cmap = 'gray',
                 vmin = None,
                 vmax = None,
                 allow_alpha=False,
                 #normalize=True,
                 title=None,
                 figsize=None,
                show=False,
                layout='compressed',
                **kwargs,):
    if not isinstance(labels, Sequence): labels = [labels] * len(images)
    fig = Figure()
    for i,img in enumerate(images[:maxelems]):
        fig.add().imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, allow_alpha=allow_alpha, **kwargs).style_img(title=labels[i])
    if show: fig.show(nrow=nrow, ncol=ncol, title=title, figsize=figsize, layout=layout)
    else: fig.create(nrow=nrow, ncol=ncol, title=title, figsize=figsize, layout=layout)
    return fig


FIG_KWARGS = {"title", "figsize", "layout"}

def _get_plot_kwargs(locals_copy, blacklist:Sequence | Collection = frozenset()):
    return {k:v for k,v in locals_copy.items() if k not in FIG_KWARGS and k not in ('show', 'fig') and k not in blacklist}
def _get_fig_kwargs(locals_copy, blacklist:Sequence | Collection = frozenset()):
    return {k:v for k,v in locals_copy.items() if k in FIG_KWARGS and k not in ('show', 'fig') and k not in blacklist}

def qfuncplot1d(
        func,
        start,
        stop,
        num = None,
        step = None,
        vectorize:Optional[Literal['list', 'numpy', 'torch']] = None,
        ylim=None,
        label=None,
        color=None,
        alpha=None,
        linewidth=None,
        linestyle=None,
        xscale=None,
        yscale=None,
        title=None,
        figsize=None,
        show=False,
        layout='compressed',
        ):
    fig = Figure()
    kwargs = locals().copy()
    fig.add().funcplot1d(**_get_plot_kwargs(kwargs, ["xscale", "yscale"])).style_chart(xscale=xscale, yscale=yscale)
    if show: fig.show(**_get_fig_kwargs(kwargs))
    else: fig.create(**_get_fig_kwargs(kwargs))
    return fig


def qscatter(
    x,
    y,
    s = None,
    c = None,
    label=None,
    marker = None,
    cmap = None,
    xlabel = 'x',
    ylabel = 'y',
    vmin=None,
    vmax=None,
    alpha = None,
    xscale=None,
    yscale=None,
    norm = None,
    colorbar = False,
    title=None,
    figsize=None,
    show=False,
    layout='compressed',
    ):
    fig = Figure()
    kwargs = locals().copy()
    (
    fig
    .add()
    .scatter(**_get_plot_kwargs(kwargs, ['xlabel', 'ylabel', "xscale", "yscale", 'colorbar']))
    .style_chart(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
    )
    if colorbar: fig.get().colorbar()
    if show: fig.show(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel']))
    else: fig.create(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel']))
    return fig

def qtricontour(
    x,
    y,
    z,
    levels = 10,
    cmap = None,
    colorbar = True,
    xlabel = 'x',
    ylabel = 'y',
    vmin = None,
    vmax = None,
    zclip = None,
    xscale=None,
    yscale=None,
    title=None,
    figsize=None,
    show=False,
    layout='compressed',
    ):
    fig = Figure()
    kwargs = locals().copy()
    if zclip is not None: kwargs['z'] = np.clip(z, *zclip)
    (
    fig
    .add().tricontour(**_get_plot_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar', 'clipz', "xscale", "yscale"]))
    .style_chart(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
    )
    if colorbar: fig.get().colorbar()
    if show: fig.show(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar']))
    else: fig.create(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar']))
    return fig

def qtricontourf(
    x,
    y,
    z,
    levels = 1000,
    cmap = None,
    colorbar = True,
    xlabel = 'x',
    ylabel = 'y',
    vmin = None,
    vmax = None,
    zclip = None,
    xscale=None,
    yscale=None,
    title=None,
    figsize=None,
    show=False,
    layout='compressed',
    ):
    fig = Figure()
    kwargs = locals().copy()
    if zclip is not None: kwargs['z'] = np.clip(z, *zclip)
    (
    fig
    .add()
    .tricontourf(**_get_plot_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar', 'clipz', "xscale", "yscale"]))
    .style_chart(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
    )
    if colorbar: fig.get().colorbar()
    if show: fig.show(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar']))
    else: fig.create(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar']))
    return fig

def qpcolormesh(
    x,
    y,
    z,
    step=300,
    cmap=None,
    contour=False,
    contour_cmap="binary",
    contour_levels=10,
    contour_alpha=0.5,
    mode:Literal["linear", "nearest", "clough"]="linear",
    xlim=None,
    ylim=None,
    zlim=None,
    alpha=None,
    shading=None,
    antialiased: bool = True,
    colorbar = True,
    xlabel = 'x',
    ylabel = 'y',
    xscale=None,
    yscale=None,
    norm = None,
    title=None,
    figsize=None,
    show=False,
    layout='compressed',
    ):
    fig = Figure()
    kwargs = locals().copy()
    (
    fig
    .add()
    .pcolormesh(**_get_plot_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar', "xscale", "yscale"]))
    .style_chart(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
    )
    if colorbar: fig.get().colorbar()
    if show: fig.show(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar']))
    else: fig.create(**_get_fig_kwargs(kwargs, ['xlabel', 'ylabel', 'colorbar']))
    return fig