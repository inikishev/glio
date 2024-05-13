"""Инструменты для среды разработки Jupyter"""
from typing import TYPE_CHECKING
import sys, gc, traceback, torch
if TYPE_CHECKING:
    def get_ipython(): pass

def clean_ipython_hist():
    """Источник - """
    # Code in this function mainly copied from IPython source
    if  'get_ipython' not in globals(): return
    ip = get_ipython() # type: ignore #pylint:disable=E1111
    user_ns = ip.user_ns # type:ignore
    ip.displayhook.flush()# type:ignore
    pc = ip.displayhook.prompt_count + 1# type:ignore
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager# type:ignore
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''#pylint:disable=W0212
def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')
def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()



def is_jupyter():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return True
    except:# type:ignore #pylint:disable=W0702
        return False


def markdown_if_jupyter(string):
    if is_jupyter():
        from IPython.display import Markdown, display
        display(Markdown(string))
        return None
    else: return string

def show_slices(sliceable):
    from .visualize import vis_imshow
    from .python_tools import shape, ndims
    from ipywidgets import interact

    kwargs = {f"s{i}":(0,v-1) for i,v in enumerate(shape(sliceable)[:-2])}
    stats = dict(orig_shape = shape(sliceable))
    def f(color, **kwargs):
        nonlocal sliceable
        view = sliceable
        for v in list(kwargs.values())[:-1] if color else kwargs.values():
            view = view[v]
        vis_imshow(view, cmap="gray")
        return dict(**stats, view_shape=view.shape, view_min = view.min(), view_max = view.max())
    return interact(f, color=False, **kwargs)

def show_slices_arr(sliceable):
    from .visualize import vis_imshow
    from ipywidgets import interact
    import numpy as np
    if hasattr(sliceable, "detach"): sliceable = sliceable.detach().cpu()
    sliceable = np.array(sliceable)
    shape = sliceable.shape
    kwargs = {f"s{i}":(0, max(shape)-1) for i in range(len(shape)-2)}
    permute = " ".join([str(i) for i in range(len(kwargs)+2)])
    stats = dict(orig_shape = sliceable.shape, dtype=sliceable.dtype, min=sliceable.min(), max=sliceable.max(), mean=sliceable.mean(), std=sliceable.std())
    def f(color, permute:str,**kwargs):
        nonlocal sliceable
        view = np.transpose(sliceable, [int(i) for i in permute.split(" ")])
        for v in list(kwargs.values())[:-1] if color else kwargs.values():
            view = view[v]
        vis_imshow(view, cmap="gray")
        return dict(**stats, view_shape=view.shape)
    return interact(f, permute=permute,color=False, **kwargs)

def show_slices_func(func, range_):
    from .visualize import vis_imshow
    from .python_tools import shape
    from ipywidgets import interact
    vals = list(range(*range_))
    test = func(vals[0])
    kwargs = {"n": 0, **{f"s{i}":(0,v-1) for i,v in enumerate(shape(test)[:-2])}}
    stats = dict(orig_shape = shape(test))
    def f(color, **kwargs):
        nonlocal test
        view = test
        for v in list(kwargs.values())[:-1] if color else kwargs.values():
            view = view[v]
        vis_imshow(view, cmap="gray")
        return dict(**stats, view_shape=view.shape)
    return interact(f, color=False, **kwargs)
