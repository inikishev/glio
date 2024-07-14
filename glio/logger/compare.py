from collections.abc import Sequence
import os, logging
from . import Logger
from ..python_tools import listdir_fullpaths, reduce_dim, flexible_filter, int_at_beginning
from ..plot import *

__all__ = [
    "Comparison"
]
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
            checkpoints = os.path.join(path, model, 'checkpoints')
            if os.path.isdir(checkpoints) and len(os.listdir(checkpoints)) > 0:
                for i in reversed(sorted(os.listdir(checkpoints), key=lambda x: int(''.join([str(i) for i in x if i.isdigit()])))):
                    fullpath = os.path.join(checkpoints, i)
                    if os.path.isfile(os.path.join(fullpath, 'logger.npz')):
                        loggers[model] = Logger.from_file(os.path.join(fullpath, 'logger.npz'))
                        break
                    #else: logging.warning(fullpath)
        return cls(loggers)

    def _filt(self, filt):
        if filt is not None: return {name: logger for name, logger in self.loggers.items() if name in flexible_filter(self.loggers.keys(), filt)}
        else: return self.loggers

    def plot(self, key, figsize=None, show=False, filt=None, **kwargs):
        return _loggers_plot(self._filt(filt), key, figsize=figsize, show=show, **kwargs)

    def _get_best_loggers(self, key, n=10, criterion='min', filt=None):
        loggers = self._filt(filt)
        if criterion == 'min': best = sorted([(name, logger.min(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1])[:n]
        elif criterion == 'max': best = sorted([(name, logger.max(key)) for name, logger in loggers.items() if key in logger], key=lambda x: x[1], reverse=True)[:n]
        else: raise ValueError(f'Unknown criterion {criterion}')
        best_loggers = {name: loggers[name] for name, _ in best}
        if filt is not None: best_loggers = {name: logger for name, logger in best_loggers.items() if name in flexible_filter(best_loggers.keys(), filt)}
        return best_loggers

    def plot_best(self, key, n=10, criterion='min', figsize=None, show=False, filt=None, **kwargs):
        best_loggers = self._get_best_loggers(key, n, criterion, filt)
        return _loggers_plot(best_loggers, key, figsize=figsize, show=show, **kwargs)

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