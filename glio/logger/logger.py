"""Logging"""
from collections.abc import Callable, Sequence
from typing import Any, Optional
import logging
from bisect import insort
import matplotlib.pyplot as plt, numpy as np, torch
from ..plot import qimshow, qimshow_grid, Figure, qpath2d
from ..python_tools import flexible_filter, sequence_to_md_table

__all__ = [
    'Logger',
]
class Logger:
    def __init__(self, note:Any = None):
        self.logs: dict[str, dict[int, Any]] = {}
        self.batch = 0
        self.note = note

    def __getitem__(self, key):
        return self.logs[key]

    def __setitem__(self, key, value):
        self.logs[key] = value

    # def __getattr__(self, key):
    #     if key in self.logs: return self.logs[key]
    #     if key.replace('_', ' ') in self.logs: return self.logs[key.replace('_', ' ')]
    #     raise AttributeError(key)

    def __contains__(self, key): return key in self.logs

    def has_substring(self, substring):
        """Returns `True` if at least one key has `substring` substring.

        Args:
            substring (_type_): _description_

        Returns:
            _type_: _description_
        """
        return any(substring in key for key in self.logs)

    def __call__(self, key):
        """Returns a list of keys and values for the given key.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return list(self[key].keys()), list(self[key].values())

    def add(self, metric, value, cur_batch):
        """Log a metric.

        Args:
            metric (_type_): _description_
            value (_type_): _description_
            cur_batch (_type_): _description_
        """
        if isinstance(value, torch.Tensor): value = value.detach().cpu()
        if metric not in self.logs: self.logs[metric] = {cur_batch: value}
        else: self.logs[metric][cur_batch] = value
        if self.batch < cur_batch: self.batch = cur_batch

    def set(self, metric, value, cur_batch=0):
        """Sets a metric under a given batch. Also useful for logging stuff that you only need the last value of.

        Args:
            metric (_type_): _description_
            value (_type_): _description_
            cur_batch (int, optional): _description_. Defaults to 0.
        """
        self.logs[metric][cur_batch] = value
        if self.batch < cur_batch: self.batch = cur_batch

    def keys(self): return self.logs.keys()
    def values(self): return self.logs.values()
    def items(self): return self.logs.items()

    def clear(self):
        """Removes all data from the logger.
        """
        self.logs = {}

    def toarray(self, key):
        """Returns an array of values.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.array(list(self.logs[key].values()), copy = False)
    def tolist(self, key):
        """Returns a list of values.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return list(self.logs[key].values())
    def totensor(self, key):
        """Returns a torch.Tensor of values.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(self.last(key), (int,float,np.ScalarType)): return torch.as_tensor(list(self.logs[key].values()))
        elif isinstance(self.last(key), (list,tuple)): return torch.as_tensor(self.toarray(key))
        return torch.from_numpy(np.asanyarray(list(self.logs[key].values())))

    def get_keys_num(self):
        """Returns all keys in the logger with scalar last value.

        Returns:
            _type_: _description_
        """
        keys = []
        for k,v in self.logs.items():
            if len(v) > 0:
                last = self.last(k)
                if (isinstance(last, (np.ndarray, torch.Tensor)) and last.ndim == 0) or isinstance(last, (int, float, np.ScalarType)): keys.append(k)
        return keys

    def get_keys_vec(self):
        """Returns all keys in the logger with vector (1D) last value.

        Returns:
            _type_: _description_
        """
        keys = []
        for k,v in self.logs.items():
            if len(v) > 0:
                last = self.last(k)
                if (isinstance(last, (np.ndarray, torch.Tensor)) and last.ndim == 1): keys.append(k)
        return keys

    def get_keys_img(self):
        """Returns all keys in the logger with 2D or 3D array as the last value.

        Returns:
            _type_: _description_
        """
        keys = []
        for k, v in self.logs.items():
            if len(v) > 0:
                last = self.last(k)
                if (isinstance(last, (np.ndarray, torch.Tensor)) and last.ndim in (2, 3)): keys.append(k)
        return keys

    def max(self, key):
        """Returns max value of the `key` metric.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return max(self.tolist(key)) # pyright:ignore
    def min(self, key):
        """Returns min value of the `key` metric.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return min(self.tolist(key)) # pyright:ignore


    def first(self, key):
        """Returns first recorded value of the `key` metric.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.tolist(key)[0]

    def num(self, key):
        """Returns number of values recorded in the `key` metric.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return len(self[key])


    def last(self, key):
        """Returns last recorded value of the `key` metric.

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.tolist(key)[-1]

    def stats_str(self, key):
        """Makes a string like:
        ```txt
        train loss: last=0.023, min=0.023, max=1.579"
        ```

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return f"{key}: last={self.last(key):.3f}, min={self.min(key):.3f}, max={self.max(key):.3f}"

    def plot(self, *args, show=False, figsize = None, **kwargs):
        """Plot all metrics passed as args.

        Args:
            show (bool, optional): _description_. Defaults to False.
        """
        f = Figure()
        f.add()
        for m in args: f.get().linechart(*self(m), label = m, **kwargs) # pyright:ignore
        f.get().style_chart()
        if show: f.show(figsize=figsize)
        else: f.create(figsize=figsize)

    def plot_all(self, *filters, figsize = None, show=False):
        """Plot all metrics using filters passed as args.

        Args:
            show (bool, optional): _description_. Defaults to False.
        """
        keys = flexible_filter(self.get_keys_num(), filters)
        keys.sort(key = lambda x: len(self[x]), reverse=True)
        f = Figure()
        f.add()
        for key in keys: f.get().plot(*self(key), label=key) # pyright:ignore
        f.get().style_chart()
        if show: f.show(figsize=figsize)
        else: f.create(figsize=figsize)

    def imshow(self, key, figsize = None, show=False):
        qimshow(self.toarray(key), title = key, figsize=figsize, show=show)

    def imshow_all(self, *filters, figsize = None, show=False):
        keys = flexible_filter(self.get_keys_img(), filters)
        qimshow_grid([self.toarray(key) for key in keys], title = ', '.join([str(i) for i in filters]), figsize=figsize, show=show)

    def hist(self, key, figsize = None, show=False):
        hist = torch.stack(self.totensor(key)).log1p().T.flip(0) # pyright:ignore
        qimshow(hist, title=key, figsize=figsize, show=show) # pyright:ignore

    def hist_all(self, *filters, figsize = (12, 5), show=False):
        keys = flexible_filter(self.get_keys_vec(), filters)
        hists = [torch.stack(self.tolist(key)).log1p().T.flip(0) for key in keys] # pyright:ignore
        if len(hists) == 0: raise ValueError(f'No data found for {", ".join(filters)}')
        qimshow_grid(hists,  labels = keys, title = ', '.join([str(i) for i in filters]), figsize=figsize, show=show) # pyright:ignore
        if show: plt.show()

    def path2d(self, key, s=None, c=None, cmap: Optional[str]|Any = "gnuplot2", linecolor:Optional[str] = 'black', figsize=None, det=False, ax=None, show=False, **kwargs):
        path = self.toarray(key)
        return qpath2d(path, title=key, c=c, s=s, cmap = cmap, linecolor=linecolor, figsize=figsize, det=det, ax=ax, show=show, **kwargs)

    def path10d(self, key, show=False):
        path = self.toarray(key)
        fig = Figure()
        fig.add().path10d(path, label=key).style_chart()
        if show: fig.show()
        else: fig.create()

    def plot_multiple(self, filters, show=False):
        ...

    def pickle(self, path:str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def state_dict(self):
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

    def save(self, filepath:str):
        """Saves logger into a compressed numpy array (npz) file.

        Args:
            path (str): _description_
        """
        if not filepath.endswith(".npz"): logging.warning("%s doesn't end with .npz", filepath)
        arrays = self.state_dict()
        np.savez_compressed(filepath, **arrays)

    def load_state_dict(self, state_dict:dict):
        for n, keys in state_dict.items():
            if n.startswith('KEYS '):
                name = n.replace("KEYS ", "")
                values = state_dict[f"VALS {name}"]
                self.logs[name] = dict(zip(keys, values))

    def load(self, filepath:str):
        """Loads logger from a compressed numpy array (npz) file.

        Args:
            path (str): _description_
        """
        arrays = np.load(filepath)
        self.load_state_dict(arrays)

    @classmethod
    def from_file(cls, filepath:str, note:Any = None):
        """Loads logger from a compressed numpy array (npz) file.

        Args:
            path (str): _description_

        Returns:
            _type_: _description_
        """
        logger:Logger = cls(note = note)
        logger.load(filepath)
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

    def info_str(self):
        text = ""
        for key in sorted(self.keys()):
            last = self.last(key)
            text += f"{key}:\n"
            text += f"    count: {len(self[key])}\n"
            text += f"    type: {type(last)}\n"
            if isinstance(last, (torch.Tensor, np.ndarray)) and last.ndim > 0:
                text += f"    last dtype: {last.dtype}\n"
                text += f"    last ndim: {last.ndim}\n"
                text += f"    last shape: {last.shape}\n"
                text += f"    last min: {last.min()}\n"
                text += f"    last max: {last.max()}\n"
                text += f"    last mean: {last.mean()}\n"
                text += f"    last var: {last.var()}\n"
                text += f"    last std: {last.std()}\n"
                text += f"    elements: {last.numel() if isinstance(last, torch.Tensor) else last.size}\n"
            elif isinstance(last, (int, float)) or (isinstance(last, (torch.Tensor, np.ndarray)) and last.ndim == 0):
                values = self.toarray(key)
                text += f"    last value: {float(last)}\n"
                text += f"    lowest: {values.min()}\n"
                text += f"    highest: {values.max()}\n"
                text += f"    mean: {values.mean()}\n"
            text += "\n"

        return text

    def to_md_table_str(self):
        keys = sorted(self.get_keys_num())
        keys_sorted = [i for i in keys if 'train' not in i]
        for key in keys:
            if 'train' in key:
                if key.replace('train', 'test') in keys_sorted:
                    keys_sorted.insert(keys_sorted.index(key.replace('train', 'test')), key)
                else:
                    insort(keys_sorted, key)

        table:list[list[Any]] = [['metric', 'n', 'min', 'max', 'first', 'last',]]
        for key in keys_sorted:
            l = self.tolist(key)
            table.append([key, len(l), f'{np.nanmin(l):.4f}',  f'{np.nanmax(l):.4f}', f'{float(l[0]):.4f}', f'{float(l[-1]):.4f}',])
        return table

    def display_table(self):
        from IPython.display import display, Markdown
        table = self.to_md_table_str()
        md = sequence_to_md_table(table, first_row_keys=True)
        display(Markdown(md))