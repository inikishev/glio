import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from .jupyter_tools import show_slices, show_slices_arr, clean_mem
from .torch_tools import lr_finder, summary as gsummary, count_parameters, seeded_rng, seed0_kwargs, BatchInputTransforms
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
    identity,
    identity_if_none,
    reduce_dim,
    flatten,
    perf_counter_context,
    find_file_containing,
)
from .train import *
from .plot import *
from .progress_bar import PBar
from .data import DSClassification, DSRegression, DSBasic, DSToTarget

from . import nn as gnn
from .helpers import conv_outsize, convtranpose_outsize
from .transforms.intensity import norm, znorm, znormch, unnomalizech

from torchzero.nn.quick import *
from torchzero.nn.layers.conv import *
from torchzero.nn.layers.linear import *
from torchzero.nn.cubes import *