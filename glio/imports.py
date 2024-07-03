import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# BUILTINS
# typing
from typing import Optional, Any, Literal
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
from collections import Counter, deque
from copy import copy, deepcopy

# LIBS
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

# mine
from torchzero.nn.cubes import *
from torchzero.nn.layers.conv import ConvBlock, ConvTransposeBlock
from torchzero.nn.layers.linear import LinearBlock