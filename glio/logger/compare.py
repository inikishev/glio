from collections.abc import Sequence
import os
from . import Logger
from ..python_tools import listdir_fullpaths, reduce_dim, flexible_filter
from ..plot import *

def _loggers_plot(loggers, key, figsize=None, show=False, **kwargs):
    fig=Figure()
    fig.add()
    for name, logger in loggers.items():
        if key in logger:
            fig.get().linechart(*logger(key), label=name, **kwargs)
    fig.get().style_chart(xlabel='batch', ylabel=key)
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
            fullpath = os.path.join(path, model)
            loggers[model] = Logger.from_file(os.path.join(fullpath, 'logger.npz'))
        return cls(loggers)

    def _filt(self, filt):
        if filt is not None: return {name: logger for name, logger in self.loggers.items() if name in flexible_filter(self.loggers.keys(), filt)}
        else: return self.loggers

    def plot(self, key, figsize=None, show=False, filt=None, **kwargs):
        return _loggers_plot(self._filt(filt), key, figsize=figsize, show=show, **kwargs)

    def plot_best(self, key, n=10, criterion='min', figsize=None, show=False, filt=None, **kwargs):
        loggers = self._filt(filt)
        if criterion == 'min': best = sorted([(name, logger.min(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1])[:n]
        elif criterion == 'max': best = sorted([(name, logger.max(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1], reverse=True)[:n]
        else: raise ValueError(f'Unknown criterion {criterion}')
        best_loggers = {name: loggers[name] for name, _ in best}
        if filt is not None: best_loggers = {name: logger for name, logger in best_loggers.items() if name in flexible_filter(best_loggers.keys(), filt)}
        return _loggers_plot(best_loggers, key, figsize=figsize, show=show, **kwargs)

    def keys(self):
        return list(set(reduce_dim([list(i.keys()) for i in self.loggers.values()])))