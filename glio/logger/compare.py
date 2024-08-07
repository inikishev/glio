from typing import Optional, Literal
from collections.abc import Sequence
import os, logging

import yaml

from . import Logger
from ..python_tools import listdir_fullpaths, reduce_dim, flexible_filter, int_at_beginning
from ..plot import *

__all__ = [
    "Comparison"
]
def _loggers_plot(loggers, key, figsize=None, show=False, legend_size = 6, **kwargs):
    fig=Figure()
    fig.add()
    loggers_with_key = sorted([(name, logger) for name, logger in loggers.items() if key in logger], key=lambda x: x[1].last(key))
    for name, logger in loggers_with_key: fig.get().linechart(*logger(key), label=name, **kwargs)
    fig.get().style_chart(xlabel='batch', ylabel=key, legend_size=legend_size)
    if show: fig.show(figsize=figsize)
    else: fig.create(figsize=figsize)
    return fig

class Comparison:
    def __init__(self, loggers:dict[str, Logger]):
        self.loggers = loggers

    @classmethod
    def from_checkpoint_folder(cls, path):
        loggers = {}
        for model in os.listdir(path):
            checkpoints = os.path.join(path, model, 'checkpoints')
            if os.path.isdir(checkpoints) and len(os.listdir(checkpoints)) > 0:
                for i in reversed(sorted(os.listdir(checkpoints), key=lambda x: int(''.join([str(i) for i in x if i.isdigit()])))):
                    fullpath = os.path.join(checkpoints, i)
                    if os.path.isfile(os.path.join(fullpath, 'logger.npz')):
                        loggers[model] = Logger.from_file(os.path.join(fullpath, 'logger.npz'))
                        break
                    #else: logging.warning(fullpath)
        return cls(loggers)

    @classmethod
    def from_benchmarks_folder(cls, path):
        loggers = {}
        for model_dir in os.listdir(path):
            fullpath = os.path.join(path, model_dir)

            # check if folder has logger file
            if os.path.isfile(os.path.join(fullpath, 'logger.npz')):
                # check if there is hyperparams file
                if os.path.isfile(os.path.join(fullpath, 'hyperparameters.yaml')):
                    # if there is, open it to notes
                    with open(os.path.join(fullpath, 'hyperparameters.yaml'), 'r', encoding='utf8') as f:
                        note = yaml.safe_load(f)
                # no hyperparameters.yaml
                else: note = None
                # load logger
                loggers[model_dir] = Logger.from_file(os.path.join(fullpath, 'logger.npz'), note = note)
            #else: logging.warning(fullpath)
        return cls(loggers)

    def _filt(self, filt):
        if filt is not None:
            if not isinstance(filt, Sequence): filt = (filt, )
            return {name: logger for name, logger in self.loggers.items() if name in flexible_filter(self.loggers.keys(), filt)}
        else: return self.loggers

    def plot(self, key, figsize=None, show=False, filt=None, legend_size=6, **kwargs):
        return _loggers_plot(self._filt(filt), key, figsize=figsize, show=show, legend_size=legend_size, **kwargs)

    def _get_best_loggers(self, key, n=10, criterion='min', filt=None):
        loggers = self._filt(filt)
        if criterion == 'min': best = sorted([(name, logger.min(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1])[:n]
        elif criterion == 'max': best = sorted([(name, logger.max(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1], reverse=True)[:n]
        elif criterion == 'lastmin': best = sorted([(name, logger.last(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1])[:n]
        elif criterion == 'lastmax': best = sorted([(name, logger.last(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1], reverse=True)[:n]
        else: raise ValueError(f'Unknown criterion {criterion}')
        best_loggers = {name: loggers[name] for name, _ in best}
        if filt is not None: best_loggers = {name: logger for name, logger in best_loggers.items() if name in flexible_filter(best_loggers.keys(), filt)}
        return best_loggers

    def plot_best(self, key, n=10, criterion='min', figsize=None, show=False, filt=None, **kwargs):
        best_loggers = self._get_best_loggers(key, n, criterion, filt)
        return _loggers_plot(best_loggers, key, figsize=figsize, show=show, **kwargs)

    def _get_contour_values(self, filt:Optional[str], x:str, y:str, z:str, mode = 'last', ):
        xvals = []
        yvals = []
        zvals = []
        loggers = self._filt(filt)
        for name, logger in loggers.items():
            xvals.append(logger.note[x]) # type:ignore
            yvals.append(logger.note[y]) # type:ignore
            if mode == 'last': zvals.append(logger.last(z))
            elif mode == 'min': zvals.append(logger.min(z))
            elif mode == 'max': zvals.append(logger.max(z))
            else: raise ValueError(f'Unknown mode {mode}')
        return xvals, yvals, zvals

    def tricontourf(
        self,
        filt: Optional[str],
        x: str,
        y: str,
        z: str,
        mode="last",
        levels=1000,
        cmap=None,
        vmin=None,
        vmax=None,
        zclip=None,
        xscale=None,
        yscale=None,
        **kwargs,
    ):
        xvals, yvals, zvals = self._get_contour_values(filt = filt, x=x, y=y, z=z, mode=mode)
        qtricontourf(
            xvals,
            yvals,
            zvals,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zclip=zclip,
            xlabel=x,
            ylabel=y,
            title=z,
            xscale=xscale,
            yscale=yscale,
            **kwargs,
        )

    def pcolormesh(
        self,
        filt: Optional[str],
        x: str,
        y: str,
        z: str,
        mode="last",
        step=500,
        cmap=None,
        xlim = None,
        ylim = None,
        zlim = None,
        contour=True,
        contour_cmap="binary",
        contour_alpha=0.5,
        contour_levels=10,
        xscale=None,
        yscale=None,
        figsize = None,
        interp_mode: Literal["linear", "nearest", "clough"] = 'linear',
        **kwargs,
    ):
        xvals, yvals, zvals = self._get_contour_values(filt = filt, x=x, y=y, z=z, mode=mode)
        qpcolormesh(
            xvals,
            yvals,
            zvals,
            step=step,
            cmap=cmap,
            xlim = xlim,
            ylim = ylim,
            zlim = zlim,
            contour=contour,
            contour_cmap=contour_cmap,
            contour_alpha=contour_alpha,
            contour_levels=contour_levels,
            xlabel=x,
            ylabel=y,
            xscale=xscale,
            yscale=yscale,
            title=z,
            mode = interp_mode,
            figsize = figsize,
            **kwargs,
        )

    def scatter(
        self,
        filt: Optional[str],
        x: str,
        y: str,
        c: str,
        mode="last",
        cmap=None,
        vmin=None,
        vmax=None,
        xscale=None,
        yscale=None,
        norm = None,
        **kwargs,
    ):
        xvals, yvals, zvals = self._get_contour_values(filt = filt,x=x, y=y, z=c, mode=mode)
        qscatter(
            x = xvals,
            y = yvals,
            c = zvals,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xlabel=x,
            ylabel=y,
            title=c,
            xscale=xscale,
            yscale=yscale,
            norm = norm,
            **kwargs,
        )

    def plot_compare_with_best(self, key, compare, n = 1, criterion = 'min', figsize=None, show=False, filt=None, **kwargs):
        best_loggers = self._get_best_loggers(key, n, criterion, filt)
        if compare in self.loggers: best_loggers[compare] = self.loggers[compare]
        else:
            keys = flexible_filter(self.loggers.keys(), [compare])
            if len(keys) == 0: raise ValueError(f'Could not find key {compare}')
            best_loggers[keys[0]] = self.loggers[keys[0]]
        return _loggers_plot(best_loggers, key, figsize=figsize, show=show, **kwargs)

    def keys(self):
        return list(set(reduce_dim([list(i.keys()) for i in self.loggers.values()])))

    def __getitem__(self, key):
        return self.loggers[key]

    def __setitem__(self, key, value):
        self.loggers[key] = value