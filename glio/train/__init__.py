"""Callbacks"""
from .learner import Learner
from .cbs_set import Set_LossFn, Set_Optimizer, Set_Scheduler
from .cbs_log import Log_Preds, Log_PredsSep, Write_Preds, Log_Time
from .cbs_debug import PrintLoss, GCCollect
from .cbs_grad import GradientClipNorm, GradientClipValue
from .cbs_metrics import Metric_Accuracy, Metric_Loss, Metric_Fn
from .cbs_hooks import Log_SignalDistribution, Log_SignalHistorgram, Log_GradDistribution, Log_GradHistorgram
from .cbs_save import Save_Best, Save_Last
from .cbs_priming import *

try:
    import accelerate as __
    from .cbs_accelerate import Accelerate
except ModuleNotFoundError: pass

try:
    import fastprogress as __
    from .cbs_fastprogress import FastProgressBar
except ModuleNotFoundError: pass
