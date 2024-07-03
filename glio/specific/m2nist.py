from collections.abc import Sequence
import torch
from glio.train import *
from glio.plot import Figure
from glio.torch_tools import one_hot_mask

TITLE = "M2NIST"

_CBSTUFF = lambda:  [
    SaveLastCB('M2NIST checkpoints'), 
    PerformanceTweaksCB(True), 
    AccelerateCB("no"),
    SimpleProgressBarCB(step=1),
    LivePlotCB(128, plot_keys = ("4plotsplot01","10metrics01"),path_keys=("4plotspath250",)),
    PrintSummaryCB(),
    PlotSummaryCB(),
    ]

_CBMETRICS = lambda: [
    LogLossCB(),
    MetricAccuracyCB(True, True, False, step=8),
    MONAIIoUCB(11, True, True, step=8),
    TorchevalPrecisionCB(11, True, True, step=8),
    TorchevalRecallCB(11, True, True, step=8),
    TorchevalF1CB(11, True, True, step=8),
    TorchevalAURPCCB(11, True, step=8),
    TorchevalRocAucCB(11, True, step=8),
    LogTimeCB(),
    #Log_LR(),
    ]

_CBUPDATEMETRICS = lambda: [
    # Log_GradDist(16),
    # Log_GradUpdateAngle(16),
    # Log_LastGradsAngle(16),
    # Log_GradPath(1),
    LogUpdateDistCB(16),
    LogLastUpdatesAngleCB(16),
    LogParamDistCB(16),
    LogParamsPathCB(1),
    LogUpdatePathCB(1),
    ]

def CALLBACKS(extra = ()):
    if not isinstance(extra, (Sequence)): extra = [extra]
    return _CBSTUFF() + _CBMETRICS() + _CBUPDATEMETRICS() + list(extra)

def plot_preds(learner:Learner, sample:tuple[torch.Tensor,torch.Tensor]):
    input, target = sample
    preds = learner.inference(input.unsqueeze(0))
    fig = Figure()
    fig.add().imshow(input).style_img("input")
    fig.add().imshow_batch(target).style_img("target")
    fig.add().imshow_batch(preds[0]).style_img("preds")
    fig.add().imshow_batch(one_hot_mask(preds[0].argmax(0), 11)).style_img("preds binary")
    fig.show(1, figsize=(14,14))