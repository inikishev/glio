import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from .imports import *
from .importsglio import *

CUDA = torch.device('cuda')
CPU = torch.device('cpu')