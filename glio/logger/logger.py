"""Логгирование"""
# Автор - Никишев Иван Олегович группа 224-31

from typing import Any, Optional
import logging
import matplotlib.pyplot as plt, numpy as np, torch
from ..visualize import datashow
from ..python_tools import flexible_filter
from ..torch_tools import smart_detach_cpu
from ..plot import *

class Logger:
    def __init__(self):
        self.logs: dict[str, dict[int, Any]] = {}

    def __getitem__(self, key):
        return self.logs[key]

    def __setitem__(self, key, value):
        self.logs[key] = value

    # def __getattr__(self, key):
    #     if key in self.logs: return self.logs[key]
    #     if key.replace('_', ' ') in self.logs: return self.logs[key.replace('_', ' ')]
    #     raise AttributeError(key)

    def __contains__(self, key): return key in self.logs

    def has_substring(self, key):
        """Возвращает True если в ключах есть подстрока key"""
        return any(key in k for k in self.logs)

    def __call__(self, key): return list(self[key].keys()), list(self[key].values())

    def add(self, metric, value, cur_batch):
        """Добавляет значение значение `value` метрики `metric` под текущим пакетом `cur_batch`"""
        if isinstance(value, torch.Tensor): value = value.detach().cpu()
        if metric not in self.logs: self.logs[metric] = {cur_batch: value}
        else: self.logs[metric][cur_batch] = value

    def keys(self): return self.logs.keys()
    def values(self): return self.logs.values()
    def items(self): return self.logs.items()

    def clear(self): self.logs = {}

    def toarray(self, key): return np.array(list(self.logs[key].values()), copy = False)
    def tolist(self, key): return list(self.logs[key].values())
    def totensor(self, key):
        if isinstance(self.last(key), (int,float)): return torch.as_tensor(list(self.logs[key].values()))
        elif isinstance(self.last(key), (list,tuple)): return torch.as_tensor(self.toarray(key))
        return torch.stack(list(self.logs[key].values()))

    def get_keys_num(self):
        """Возвращает список ключей c числовыми значениями"""
        keys = []
        for k,v in self.logs.items():
            if len(v) > 0:
                last = self.last(k)
                if (isinstance(last, (np.ndarray, torch.Tensor)) and last.ndim == 0) or isinstance(last, (int, float, np.ScalarType)): keys.append(k)
        return keys

    def get_keys_vec(self):
        """Возвращает список ключей c векторными значениями"""
        keys = []
        for k,v in self.logs.items():
            if len(v) > 0:
                last = self.last(k)
                if (isinstance(last, (np.ndarray, torch.Tensor)) and last.ndim == 1): keys.append(k)
        return keys

    def get_keys_img(self):
        """Возвращает список ключей со значениями, соответствующими массивам второго или третьего ранга"""
        keys = []
        for k, v in self.logs.items():
            if len(v) > 0:
                last = self.last(k)
                if (isinstance(last, (np.ndarray, torch.Tensor)) and last.ndim in (2, 3)): keys.append(k)
        return keys

    def max(self, key):
        """Возвращает максимальное значение метрики под ключём `key`"""
        return max(self.tolist(key)) # pyright:ignore
    def min(self, key):
        """Возвращает минимальное значение метрики под ключём `key`"""
        return min(self.tolist(key)) # pyright:ignore

    def last(self, key):
        """Возвращает последнее значение метрики под ключём `key`"""
        return self.tolist(key)[-1]
    
    def stats_str(self, key):
        return f"{key}: last={self.last(key):.3f}, min={self.min(key):.3f}, max={self.max(key):.3f}"

    def plot(self, *args, show=False, **kwargs):
        """Строит график метрик по списку их ключей."""
        for m in args: plt.plot(*self(m), label = m, **kwargs) # pyright:ignore
        plt.legend()

    def plot_all(self, *filters, show=False):
        """Строит график метрик по списку их подстрок ключей"""
        keys = flexible_filter(self.get_keys_num(), filters)
        keys.sort(key = lambda x: len(self[x]), reverse=True)
        for key in keys: plt.plot(*self(key), label=key) # pyright:ignore
        plt.legend(prop={'size': 6})
        if show: plt.show()

    def imshow(self, key, fit = True, to_square = True, show=False):
        datashow(self.toarray(key), title = key, fit = fit)
        if show: plt.show()

    def imshow_all(self, *filters, fit = True, to_square = True, show=False):
        keys = flexible_filter(self.get_keys_img(), filters)
        datashow([self.toarray(key) for key in keys], title = ', '.join([str(i) for i in filters]), fit = fit)
        if show: plt.show()

    def hist(self, key, show=False):
        hist = torch.stack(self.totensor(key)).log1p().T.flip(0) # pyright:ignore
        datashow(hist, title=key, fit = True, resize=(256,512), max_size=2048) # pyright:ignore
        if show: plt.show()

    def hist_all(self, *filters, figsize = (12, 5), show=False):
        keys = flexible_filter(self.get_keys_vec(), filters)
        hists = [torch.stack(self.tolist(key)).log1p().T.flip(0) for key in keys] # pyright:ignore
        if len(hists) == 0: raise ValueError(f'No data found for {", ".join(filters)}')
        datashow(hists,  labels = keys, title = ', '.join([str(i) for i in filters]), fit = True, resize=(256,512), figsize=figsize, interpolation='nearest') # pyright:ignore
        if show: plt.show()

    def path2d(self, key, s=None, c=None, cmap: Optional[str]|Any = "gnuplot2", linecolor:Optional[str] = 'black', figsize=None, det=False, ax=None, show=False, **kwargs):
        path = self.toarray(key)
        return qpath2d(path, title=key, c=c, s=s, cmap = cmap, linecolor=linecolor, figsize=figsize, det=det, ax=ax, show=show, **kwargs)

    def path10d(self, key, show=False):
        path = self.toarray(key)
        fig = Figure()
        fig.add().path10d(path, label=key).style_chart()
        fig.create()
        if show: plt.show()

    def plot_multiple(self, filters, show=False):
        ...

    def pickle(self, path:str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def state_dict(self):
        """Возвращает словарь массивов torch.Tensor для сериализации. Этот словарь можно загрузить в логгер методами `load_dict` или `logger = Logger.from_dict(d)`."""
        #state_dict = {f"KEYS {k}": torch.as_tensor(self(k)[0]) for k in self.keys()}
        #state_dict.update({f"VALS {k}": self.totensor(k) for k in self.keys()})
        state_dict = {}
        for k in self.keys():
            try:
                state_dict[f"VALS {k}"] = self.totensor(k)
                state_dict[f"KEYS {k}"] = torch.as_tensor(self(k)[0])
            except Exception as e: # pylint:disable=W0718
                logging.warning(msg = "Failed to save `%s`: %s" % (k, e)) # pylint:disable=C0209
        return state_dict

    def save(self, path:str):
        """Сохраняет логгер, используя сжатый формат .npz. Его потом можно загрузить через `logger = Logger.from_file(path)`"""
        if not path.endswith(".npz"): logging.warning("%s doesn't end with .npz", path)
        arrays = self.state_dict()
        np.savez_compressed(path, **arrays)

    def load_state_dict(self, state_dict:dict):
        for n, keys in state_dict.items():
            if n.startswith('KEYS '):
                name = n.replace("KEYS ", "")
                values = state_dict[f"VALS {name}"]
                self.logs[name] = dict(zip(keys, values))

    def load(self, path:str):
        """Загружает логгер из сжатого формат .npz, все ключи файла перезапишут соответствующие ключи логгера, если те существуют."""
        arrays = np.load(path)
        self.load_state_dict(arrays)

    @classmethod
    def from_file(cls, path:str):
        """Загружает логгер из сжатого формат .npz"""
        logger:Logger = cls()
        logger.load(path)
        return logger

    @classmethod
    def from_dict(cls, d:dict):
        logger:Logger = cls()
        logger.load_state_dict(d)
        return logger

    def rollback(self, n_batch:int):
        for key in self.keys():
            self.logs[key] = {k:v for k,v in self.logs[key].items() if k <= n_batch}

    def range(self, start, stop, metrics=None):
        l = Logger()
        if metrics is None:
            for k,v in self.items():
                l[k] = {batch: metric for batch, metric in v.items() if start <= batch <= stop}
            return l
        elif isinstance(metrics, str): metrics = [metrics]
        for m in metrics:
            l[m] = {batch: metric for batch, metric in self[m].items() if start <= batch <= stop}
        return l