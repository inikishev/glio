from ..train2 import *

def get_cbs(title, cbs = ()):
    CALLBACKS =  (Metric_Loss(), # Log_GradHistorgram(16), Log_SignalHistorgram(16), Log_LastGradsAngle(128), Log_GradPath(1)
                #Log_UpdateDist(128), Log_GradDist(128), Log_GradUpdateAngle(128), Log_ParamDist(128),
                #Log_LastUpdatesAngle(128),
                #Log_ParamPath(32), Log_UpdatePath(32),
                Log_Time(), Save_Best(title), Save_Last(title), Log_LR(), PerformanceTweaks(True), Accelerate("no"),
                Metric_Accuracy(True, True, False, name = 'accuracy', step=4),
                MONAI_IoU(4, True, True, step=32, name='iou'),
                Torcheval_Precision(4, True, True, step=16),
                Torcheval_Recall(4, True, True, step=16),
                Torcheval_Dice(4, True, True, step=8, name='f1'),
                Torcheval_AURPC(4, True, step=32),
                Torcheval_AUROC(4, True, step=32),
                FastProgressBar(step_batch=128, plot=True),
                Summary(),
                PlotSummary(path='summaries'),
                # CallTrainAndEvalOnOptimizer(),
                ) + cbs

    return CALLBACKS