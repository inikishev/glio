"""Callbacks"""
from .Learner import Learner, Learner_DebugPerformance
from .cbs_set import Set_LossFn, Set_Optimizer, Set_Scheduler
from .cbs_log import Log_Preds, Log_PredsSep, Write_Preds, Log_Time, Log_LR, Log_PredsTo
from .cbs_debug import Debug_GCCollect, Debug_PrintFirstParam, Debug_PrintLoss
from .cbs_grad import GradientClipNorm, GradientClipValue
from .cbs_metrics import CBMetric, Metric_Accuracy, Metric_Loss, Metric_PredsTargetsFn, Metric_LearnerFn
from .cbs_hooks import Log_LayerSignalDistribution, Log_LayerSignalHistorgram, Log_LayerGradDistribution, Log_LayerGradHistorgram
from .cbs_updatestats import (Log_ParamDist, Log_GradDist, Log_UpdateDist,
                              Log_GradUpdateAngle,Log_LastGradsAngle, Log_LastUpdatesAngle,
                              Log_ParamPath, Log_GradPath, Log_UpdatePath)

from .cbs_save import Save_Best, Save_Last
from .cbs_priming import LRFinderPriming, IterLR
from .cbs_summary import Summary
from .cbs_monai import *
from .cbs_torcheval import Torcheval_Precision, Torcheval_Recall, Torcheval_AURPC, Torcheval_AUROC, Torcheval_Dice
from .cbs_liveplot import LivePlot, LivePlot2, PlotSummary
from .cbs_simpleprogress import SimpleProgressBar, PrintMetrics, PrintLoss, PrintInverseDLoss
from .cbs_default_overrides import (
    OneBatch_Closure,
    OneBatch_ClosureWithNoBackward,
    GradientFree,
    GradientFreeWithZeroGrad,
    PassLossToOptimizerStep,
    SimpleMomentum,
    CallTrainAndEvalOnOptimizer,
    AddLossReturnedByModelToLossInGetLoss,
    AddLossReturnedByModelToLossInBackward,
)
from .cbs_performance import PerformanceTweaks


from ..design.EventModel import Callback, CBCond, CBContext, CBEvent, CBMethod

try:
    import accelerate as __
    from .cbs_accelerate import Accelerate
except ModuleNotFoundError: pass

try:
    import fastprogress as __
    from .cbs_fastprogress import FastProgressBar
except ModuleNotFoundError: pass

try:
    import hiddenlayer as __
    from .cbs_hiddenlayer import HLCanvas
except ModuleNotFoundError: pass
