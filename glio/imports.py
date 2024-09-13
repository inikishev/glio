#pylint:disable=C0413
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# region builtins

# math
import math
import pathlib
import pickle
import random
# files
import shutil
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from copy import copy, deepcopy
# tools
from functools import partial
from pathlib import Path
# BUILTINS
# typing
from typing import Any, Literal, Optional

# region Libs
# pytorch
import torch
import torchvision
from torch import Tensor, as_tensor, nn, optim  # pylint:disable=E0611
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# torch related
from torchinfo import summary as __
from torchvision import ops as tvops
from torchvision.transforms import v2

summary = partial(__,  depth=100, col_names = ("input_size", "output_size", "kernel_size", "num_params", "trainable"), mode='train', row_settings =["depth", "var_names",])

import joblib
import matplotlib.pyplot as plt
# other
import numpy as np
from torchzero import nn as tznn
# region mine
from torchzero.nn.cubes import *
from torchzero.nn.layers.conv import *
from torchzero.nn.layers.linear import *
from torchzero.nn.layers.sequential import *

from . import nn as gnn
from .data import DSBasic, DSClassification, DSRegression, DSToTarget
from .helpers import conv_outsize, convtranpose_outsize
from .jupyter_tools import clean_mem, show_slices, show_slices_arr
from .loaders.image import imread, imreadtensor, imwrite
from .plot import *
from .progress_bar import PBar
from .python_tools import (CacheRepeatIterator, Compose, find_file_containing,
                           flatten, get0, get1, get__name__, get_all_files,
                           getlast, identity, identity_if_none,
                           listdir_fullpaths, perf_counter_context)
from .python_tools import printargs as printa
from .python_tools import reduce_dim, type_str
from .torch_tools import (BatchInputTransforms, count_parameters, lr_finder,
                          seed0_kwargs, seeded_rng)
from .torch_tools import summary as gsummary
from .train import *
from .transforms.intensity import norm, unznormch, znorm, znormch

#region glio



CUDA = torch.device('cuda')
CPU = torch.device('cpu')