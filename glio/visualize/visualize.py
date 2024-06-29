"""Vis"""
# Автор - Никишев Иван Олегович группа 224-31
from typing import Optional
import math
from PIL import Image
import torch
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from ..python_tools import key_value, reduce_dim
from ..transforms import fToRange,fToChannels

def info(X, tab = ''):
    v_info = f'{tab}type = {type(X)}'
    _newline = '\n'
    _whitespace = "    "
    if isinstance(X, (list, tuple, dict)):
        v_info+= '; children:\n'
        v_info += f'{_newline.join([f"{k}: {info(v, tab=tab+_whitespace)}" for k,v in key_value(X)])}'
    if isinstance(X, torch.Tensor): v_info += f'; shape = {X.shape}; pixels = {np.prod(X.shape)}; dtype = {X.dtype}, min = {X.float().min()}; max = {X.float().max()}, avg = {X.float().mean()}'
    elif isinstance(X, np.ndarray): v_info += f'; shape = {X.shape}; pixels = {np.prod(X.shape)}; dtype = {X.dtype}, min = {X.min()}; max = {X.max()}, avg = {X.mean()}'
    elif isinstance(X, str): v_info += f'; length = {len(X)}, first 100 chars = {X[:min(100, len(X))]}'
    elif isinstance(X, (int, float)): v_info += f'; value = {X}'
    return v_info

def plot(x, y = None, figsize = (6,6), **kwargs): # pyright:ignore[reportRedeclaration]
    # list of plots
    if isinstance(x[0], (int, float)): x = [x]
    if y is None:
        y: list = [None]*len(x)
        for i in range(len(y)): y[i] = list(range(len(x[i])))

    if len(y) != len(x):
        if len(y) != len(x[0]): raise ValueError(f'y must have the same length as x, its current length = {len(y)}, len(x) = {len(x[0])}')
        if isinstance(y, (np.ndarray, torch.Tensor)): y = y.tolist()
        y = y*len(x)

    xmax = ymax = math.inf
    xmin = ymin = -math.inf

    for i in x:
        xmax = max(xmax, i)
        xmin = min(xmin, i)
    for i in y:
        ymax = max(ymax, i)
        ymin = min(ymin, i)


def ax_imshow(ax, image, **kwargs):
    ax.imshow(image, **kwargs)
    ax.set_axis_off()
    return ax

def ax_plot(ax, x, y = None, **kwargs):
    if y: ax.plot(x, y, **kwargs)
    else: ax.plot(x, **kwargs)
    return ax

def ax_plot_multiple(ax, x,y, **kwargs):
    if y:
        ax.plot(x, y, **kwargs)
        for xx,yy in zip(x,y): ax_plot(ax,xx,yy, **kwargs)
    else:
        for i in x: ax_plot(ax, i, **kwargs)
    return ax

def ax_print(ax, data, **kwargs):
    ax.set_axis_off()
    ax.set_frame_on(False)
    return ax

def ax_clear(ax):
    ax.set_axis_off()
    ax.set_frame_on(False)
    return ax

def split_line(line, length):
    line = str(line).split(' ')
    line = reduce_dim([i.split('/') for i in line])
    c = 0
    for i,v in enumerate(line):
        c+=len(v)
        if '\n' in v: c=0
        if c>=length and i!=0:
            line[i] = f'\n{line[i]}'
            c = len(v)
    return ' '.join(line)


def datashow(data, labels=None, title=None, figsize=None, nrows=None, ncols=None, max_size=1024, fit=True, resize:Optional[int | tuple[int, int]]=None, **kwargs):
    NoneType = type(None)

    if isinstance(data, tuple): data = list(data)

    # Check if figsize is an integer and convert it to a tuple of (figsize, figsize)
    if isinstance(figsize, int):
        figsize = (figsize, figsize)

    # Convert data to a list if it is not already a list
    # data = data_list(data)

    # Loop through each element in the data list
    for i in range(len(data)):
        # Check if the element is a tensor or numpy array
        if isinstance(data[i], torch.Tensor): data[i] = data[i].detach().cpu()
        if isinstance(data[i], (np.ndarray, torch.Tensor)) and data[i].ndim != 0:
            # Resize the element if its size is larger than max_size
            if max_size:
                factor = int(math.ceil(max(data[i].shape) / max_size))
                if factor > 1:
                    if data[i].ndim in (2, 3):
                        data[i] = data[i][::factor, ::factor]
                    if data[i].ndim == 1:
                        data[i] = data[i][::factor]

            # Normalize the element if fit=True
            if data[i].ndim in (2, 3) and fit:
                data[i] = fToRange(data[i], 0, 1)

            # Resize the element if resize is provided
            if data[i].ndim in (2, 3) and resize:
                resize = (resize, resize) if isinstance(resize, int) else resize
                #data[i] = fResize(data[i], *resize)

            if data[i].ndim == 3:
                data_shape = data[i].shape
                if data_shape[0] <= 4 and data_shape[2] > 4: # channels first normal image
                    data[i] = fToChannels(data[i], 3)
                    data[i] = torch.permute(data[i], (1,2,0))
                elif data_shape[0] > 4 and data_shape[2] <= 4: # channels last normal image
                    if data_shape[2] == 4: data[i] = data[i][:,:,:3]
                    elif data_shape[2] == 2:
                        data[i] = torch.permute(data[i], (2,0,1))
                        data[i] = fToChannels(data[i], 3)
                        data[i] = torch.permute(data[i], (1,2,0))
                else: # both are > 4 or < 4 we assume channels first
                    data[i] = fToChannels(data[i], 3)
                    data[i] = torch.permute(data[i], (1,2,0))



    # Calculate the number of rows and columns for subplots
    if nrows is None:
        if ncols is None:
            fsize = (1, 1) if figsize is None else figsize
            nrows = int(math.ceil(len(data) ** (fsize[1] / sum(fsize))))
        else:
            nrows = int(math.ceil(len(data) / ncols))
    if ncols is None:
        ncols = int(math.ceil(len(data) / nrows))

    # Create subplots
    _, axes = plt.subplots(nrows, ncols, layout="compressed", figsize=figsize)

    # Flatten the axes array if it's a numpy array
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    # Iterate through each subplot
    for i, ax in enumerate(axes):
        # Check if there is data available for the subplot
        if len(data) > i:
            elem = data[i]

            # Plot image if element is an instance of PIL.Image.Image
            if isinstance(elem, Image.Image):
                ax_imshow(ax, elem, **kwargs)

            # Plot image or line plot if element is a tensor or numpy array
            elif isinstance(elem, (torch.Tensor, np.ndarray)):
                if elem.ndim in [2, 3]:
                    ax_imshow(ax, elem, **kwargs)
                elif elem.ndim == 1:
                    ax_plot(ax, elem, **kwargs)
                elif elem.ndim == 0:
                    ax_print(ax, elem)
                else:
                    raise ValueError(f'Image {i} can\'t have ndim = {elem.ndim}')

            # Print element if it is a string, int, float, None, bool, or range
            elif isinstance(elem, (str, int, float, NoneType, bool, range)) or (isinstance(elem, (list, tuple)) and len(elem) == 0):
                ax_print(ax, elem)

            # plot if elem is a list or a tuple
            elif isinstance(elem, (list, tuple)) and len(elem) > 0:
                # list of values
                if isinstance(elem[0], (int, float)):
                    ax_plot(ax, elem, **kwargs)
                # list of lists
                else:
                    # go through each list
                    for j in elem:
                        # list is inside
                        if isinstance(j, (list, tuple)):
                            # list of values, plot it
                            if isinstance(j[0], (int, float)): ax_plot(ax, j, **kwargs)
                            # 2 lists of x and y values
                            elif isinstance(j[0], (list, tuple)): ax_plot(ax, j[0], j[1], **kwargs)
                        # list is a dict of x and y values
                        elif isinstance(j, dict): ax_plot(ax, j.keys(), j.values(), **kwargs)

            elif isinstance(elem, dict) and len(elem) > 0:
                ax_plot(ax, elem.keys(), elem.values(), **kwargs)
            # Raise an error if the element type is unknown
            else: raise ValueError(f"Sample {i} has type={type(elem)} and idk how to plot it...")

        # Clear the subplot if there is no data
        else: ax_clear(ax)

        # add labels
        if labels is not None:
            if len(labels) > i: ax.set_title(f'{split_line(labels[i], 15)}', fontsize=int(max((12-ncols), 5)))

    #plt.subplots_adjust(wspace=0, hspace=0)
    if title: plt.suptitle(f'{title}')
    #fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,wspace=0)
    #plt.tight_layout(pad=0.05, w_pad=0.2, h_pad=0.2)
    plt.show()

def batch_preview(batch):
    ...


def to_cpu_tensor(x):
    if isinstance(x, np.ndarray): x = torch.as_tensor(x)
    elif isinstance(x, torch.Tensor): x = x.detach().cpu()
    return x

class Visualizer:
    def __init__(self):
        self.data = {}
        self.cur = 0

    def __len__(self): return len(self.data)

    def imshow(self, elem, mode = 'auto', label = None, **kwargs):
        elem = to_cpu_tensor(elem)
        mode = mode.lower()

        if mode == "auto":
            if elem.ndim == 2: mode = "hw"
            elif elem.ndim == 3:
                if elem.shape[0] > elem.shape[2]: mode = "hwc"
                else: mode = "chw"
            elif elem.ndim == 4:
                if elem.shape[1] > elem.shape[3]: mode = "bhwc"
                else: mode = "bchw"

        if mode is None or mode == 'image': mode == 'hw' # pylint: disable=W0104 # type: ignore
        elif mode == 'batch': mode = 'bchw'
        #channels last
        if 'hwc' in mode or mode == 'hw':
            if mode == 'bhwc': elem = elem[0]
            if elem.ndim == 3:
                if elem.shape[2] == 2: elem = torch.cat((elem, torch.ones(elem[:,:,:1].shape, device=elem.device)*elem.min()), dim=2)
                elif elem.shape[2] > 4: elem = elem[:,:,:4]
        # channels first
        elif 'chw' in mode:
            if mode == 'bchw' or  elem.ndim == 4: elem = elem[0]
            if elem.shape[0] == 2: elem = torch.cat((elem, elem[:1]))
            elif elem.shape[0] > 4: elem = elem[:4]
            if elem.shape[0] in (3, 4): elem = torch.permute(elem, (1,2,0))
            else: elem = elem[0]

        self.data[self.cur] = {'elem': elem, 'type': 'imshow', 'label':label, 'kwargs':kwargs}
        self.cur += 1

    def imshow_grid(self, elem, mode = 'bchw', nelems = 4, grid_rows = None, label = None, **kwargs):
        elem = to_cpu_tensor(elem)
        mode = mode.lower()
        if isinstance(elem, (torch.Tensor, np.ndarray)):
            if mode is None or mode == 'image': mode == 'bhwc' # pylint: disable=W0104 # type: ignore
            elif mode == 'batch':
                if elem.ndim == 3: mode = 'bhw'
                elif elem.ndim == 4: mode = 'bchw'
            if mode == 'bchw': # batch - channnel - height - width. Takes first n channels of the first element.
                if elem.shape[1] == 2: elem = torch.cat((elem, torch.ones(elem[:,:1].shape, device=elem.device)*elem.min()), dim=1)
                elif elem.shape[1] > 4: elem = elem[:,:4]
                nelems = min(nelems, elem.shape[0])
                if grid_rows is None: grid_rows = max(1, int(math.ceil(nelems**0.5)))
                elem = torch.permute(torchvision.utils.make_grid(elem[:nelems], nrow = grid_rows), (1,2,0))
            elif mode == 'bhw': # batch - height - width. Takes first n elements (?).
                if elem.ndim == 4: elem = elem[0]
                elem = torch.unsqueeze(elem, 1)
                nelems = min(nelems, elem.shape[0])
                if grid_rows is None: grid_rows = max(1, int(math.ceil(nelems**0.5)))
                elem = torch.permute(torchvision.utils.make_grid(elem[:nelems], nrow = grid_rows), (1,2,0))
            elif mode == 'bhwc':
                ...
            elif mode == 'filters':
                if elem.ndim == 2: pass # nothing needs to be done
                if elem.ndim == 4 and elem.shape[1] == 1: elem = elem.squeeze(1)
                if elem.ndim == 3:
                    if grid_rows is None: grid_rows = max(1, int(math.ceil(nelems**0.5)))
                    elem = torch.permute(torchvision.utils.make_grid(elem[:nelems].unsqueeze(1), nrow = grid_rows, normalize=True), (1,2,0))
                elif elem.ndim == 4:  # weights are not batched; for example, conv2d can have size (out=16, in=3, kerel=5, 5)
                    if elem.shape[1] == 2: elem = torch.cat((elem, torch.ones(elem[:,:1].shape, device=elem.device)*elem.min()), dim=1)
                    if grid_rows is None: grid_rows = max(1, int(math.ceil(nelems**0.5)))
                    elem = elem = torch.permute(torchvision.utils.make_grid(elem[:nelems, :3], nrow = grid_rows, normalize=True), (1,2,0))
        else: ...

        self.data[self.cur] = {'elem': elem, 'type': 'imshow', 'label':label, 'kwargs':kwargs}
        self.cur += 1

    def plot(self, elem, mode = '1', label = None, **kwargs):
        elem = to_cpu_tensor(elem)
        mode = mode.lower()
        if mode == '1': pass
        elif mode == 'xy': ...
        self.data[self.cur] = {'elem': elem, 'type': 'plot', 'label':label, 'kwargs':kwargs}
        self.cur += 1

    def show(self, nrows = None, ncols = None, figsize = None, title = None, fontsize = None):
        # Calculate the number of rows and columns for subplots
        if nrows is None:
            if ncols is None:
                fsize = (1, 1) if figsize is None else figsize
                nrows = int(math.ceil(len(self.data) ** (fsize[1] / sum(fsize))))
            else:
                nrows = int(math.ceil(len(self.data) / ncols))
        if ncols is None:
            ncols = int(math.ceil(len(self.data) / nrows))

        # Create subplots
        _, axes = plt.subplots(nrows, ncols, layout="compressed", figsize=figsize)

        # Flatten the axes array if it's a numpy array
        axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
        if fontsize is None: fontsize = int(max((12-ncols), 5))
        for k, v in self.data.items():
            plot_type = v['type']
            if plot_type == 'imshow': ax_imshow(axes[k], fToRange(v['elem'], 0, 1), **v['kwargs'])
            elif plot_type == 'plot': ax_plot(axes[k], v['elem'], **v['kwargs'])
            if v['label'] is not None: axes[k].set_title(f'{split_line(v["label"], 15)}', fontsize=fontsize) # pyright:ignore
        if len(axes) > len(self.data):
            for i in range(len(self.data), len(axes)):
                ax_clear(axes[i])
        if title is not None: plt.suptitle(f'{title}')
        plt.show()
    def clear(self):
        self.data = {}
        self.cur = 0

def vis_imshow(elem, mode = 'auto', label = None, **kwargs):
    v = Visualizer()
    v.imshow(elem=elem, mode=mode, label=label, **kwargs)
    v.show()

def vis_imshow_grid(elem, mode = 'bchw', nelems = 4, grid_rows = None, label = None, **kwargs):
    v = Visualizer()
    v.imshow_grid(elem=elem, mode=mode, nelems = nelems, grid_rows=grid_rows, label=label, **kwargs)
    v.show()
