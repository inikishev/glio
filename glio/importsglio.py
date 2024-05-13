import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from .jupyter_tools import show_slices, show_slices_arr, clean_mem
from .torch_tools import lr_finder, summary, count_parameters, seeded_rng, seed0_kwargs
from .python_tools import (
    type_str,
    get__name__,
    CacheRepeatIterator,
    get_all_files,
    listdir_fullpaths,
    get0,
    get1,
    getlast,
    printargs as printa,
    reduce_dim,
    flatten,
    perf_counter_context,
)
from . import plt_colormaps as _plt_colormaps

from .train2 import *
from .plot import *
from .progress_bar import PBar
from .data import DSClassification, DSRegression, DSBasic, DSToTarget

from . import nn as gnn
from .nn import conv, convt, linear, seq, block
from .helpers import cnn_output_size, tcnn_output_size
from .transforms import norm_to01, norm_to11, z_normalize, z_normalize_channels, fUnNormalize, norm_torange
