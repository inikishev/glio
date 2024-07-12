"""Callbacks"""

from .Learner import *
from .cbs_set import *
from .cbs_log import *
from .cbs_debug import *
from .cbs_grad import *
from .cbs_metrics import *
from .cbs_hooks import *
from .cbs_updatestats import *
from .cbs_save import *
from .cbs_priming import *
from .cbs_summary import *
from .cbs_monai import *
from .cbs_torcheval import *
from .cbs_liveplot import *
from .cbs_simpleprogress import *
from .cbs_default_overrides import *
from .cbs_performance import *
from .cbs_optim import *

from ..design.EventModel import Callback, ConditionCallback, BasicCallback, EventCallback, MethodCallback

try:
    import accelerate as __
    from .cbs_accelerate import *
except ModuleNotFoundError: pass

try:
    import fastprogress as __
    from .cbs_fastprogress import *
except ModuleNotFoundError: pass

# try:
#     import hiddenlayer as __
#     from .cbs_hiddenlayer import HLCanvas
# except ModuleNotFoundError: pass

import types # pylint:disable=C0411
__all__ = [name for name, thing in globals().items() # type:ignore
          if not (name.startswith('_') or isinstance(thing, types.ModuleType))]
del types