#pylint:disable=C0413
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# region builtins

# BUILTINS
# typing
from typing import Optional, Any, Literal
from collections import Counter, deque
from collections.abc import Iterable, Sequence, Mapping

# files
import shutil
import pickle
import pathlib
from pathlib import Path

# math
import math
import random

# tools
from functools import partial
from contextlib import nullcontext
from copy import copy, deepcopy

# region Libs
# pytorch
import torch
from torch import Tensor, as_tensor # pylint:disable=E0611
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
from torchvision import ops as tvops

# torch related
from torchinfo import summary as __
summary = partial(__,  depth=100, col_names = ("input_size", "output_size", "kernel_size", "num_params", "trainable"), mode='train', row_settings =["depth", "var_names",])

# other
import numpy as np
import joblib
import matplotlib.pyplot as plt

# region mine
from torchzero.nn.cubes import *
from torchzero.nn.layers.sequential import *
from torchzero.nn.layers.conv import *
from torchzero.nn.layers.linear import *
from torchzero import nn as tznn


#region glio

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
    Compose,
)
from .train import *
from .plot import *
from .progress_bar import PBar
from .data import DSClassification, DSRegression, DSBasic, DSToTarget

from . import nn as gnn
from .helpers import conv_outsize, convtranpose_outsize
from .transforms.intensity import norm, znorm, znormch, unznormch

from .loaders.image import imread, imreadtensor, imwrite
CUDA = torch.device('cuda')
CPU = torch.device('cpu')