import statistics
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn.functional as F
import monai.metrics

from .Learner import Learner
from ..design.EventModel import Callback, EventModel, CBMethod
from ..torch_tools import one_hot_mask
from .cbs_metrics import CBMetric

__all__ = ["MONAI_Dice", "MONAI_GeneralizedDice", "MONAI_IoU", "MONAI_ROCAUC", "MONAI_ConfusionMatrix", "MONAI_ConfusionMatrixMetrics", "MONAI_ConfusionMatrixMetricsMean"]

class MONAI_Dice(CBMetric):
    def __init__(self, num_classes, argmax_preds = True, argmax_targets = False, ignore_bg = False, step=1, name="dice"):
        """Dice (2 overlaps divided by sum), also same as F1 score.

        Args:
            num_classes (_type_): _description_
            argmax_preds (bool, optional): _description_. Defaults to True.
            argmax_targets (bool, optional): _description_. Defaults to False.
            ignore_bg (bool, optional): _description_. Defaults to False.
            step (int, optional): _description_. Defaults to 1.
            name (str, optional): _description_. Defaults to "dice".
        """
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.include_background = not ignore_bg
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds: preds = learner.preds.argmax(1)
        else: preds = learner.preds
        if self.argmax_targets: targets = learner.targets.argmax(1)
        else: targets = learner.targets
        return float(monai.metrics.compute_dice(preds, targets, include_background=self.include_background, num_classes=self.num_classes).nanmean()) # type:ignore #pylint:disable=E1101

class MONAI_GeneralizedDice(CBMetric):
    def __init__(self, num_classes, argmax_preds = True, argmax_targets = False, ignore_bg = False, step=1, name="generalized dice"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.include_background = not ignore_bg
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds: preds = learner.preds.argmax(1)
        else: preds = learner.preds
        if self.argmax_targets: targets = learner.targets.argmax(1)
        else: targets = learner.targets
        return float(monai.metrics.compute_generalized_dice(preds, targets, include_background=self.include_background).nanmean()) # type:ignore #pylint:disable=E1101


class MONAI_IoU(CBMetric):
    def __init__(self, num_classes, argmax_preds = True, argmax_targets = False, ignore_bg = False, step=1, name="iou"):
        """Jaccard index, intersection over union.

        Args:
            num_classes (_type_): _description_
            argmax_preds (bool, optional): _description_. Defaults to True.
            argmax_targets (bool, optional): _description_. Defaults to False.
            ignore_bg (bool, optional): _description_. Defaults to False.
            step (int, optional): _description_. Defaults to 1.
            name (str, optional): _description_. Defaults to "iou".
        """
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.include_background = not ignore_bg
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds: preds = learner.preds.argmax(1)
        else: preds = learner.preds
        if self.argmax_targets: targets = learner.targets.argmax(1)
        else: targets = learner.targets
        preds = F.one_hot(preds, num_classes=self.num_classes).swapaxes(-1, 2) # pylint:disable=E1102
        targets = F.one_hot(targets, num_classes=self.num_classes).swapaxes(-1, 2) # pylint:disable=E1102
        return float(monai.metrics.compute_iou(preds, targets, include_background=self.include_background).nanmean()) # type:ignore


class MONAI_ROCAUC(CBMetric):
    def __init__(self, num_classes, to_onehot_targets = False, ignore_bg = False, average='macro', step=1, teststep = 1, name="roc auc"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.to_onehot_targets = to_onehot_targets
        self.ignore_bg = ignore_bg
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.test_cond = None if teststep<=1 else lambda _, i: i%teststep==0
        self.average = average
        self.metric = name

    def __call__(self, learner: Learner):
        preds = learner.preds.flatten(1, -1)
        if self.to_onehot_targets: targets =  F.one_hot(learner.targets, num_classes=self.num_classes) # pylint:disable=E1102
        else:targets = learner.targets.flatten(1, -1)
        if self.ignore_bg:
            preds = preds[:, 1:]
            targets = targets[:, 1:]
        return float(monai.metrics.compute_roc_auc(preds, targets, average=self.average)) # type:ignore


def _aggregate_list_of_confusion_matrix(x:list[np.ndarray]):
    return torch.from_numpy(np.sum(x, 0))
class MONAI_ConfusionMatrix(CBMetric):
    def __init__(self, include_bg = True, name="confusion matrix", step=1):
        """confusion matrix. Y and Y_hat must be in one-hot format!

        Args:
            include_bg (bool, optional): _description_. Defaults to True.
            name (str, optional): _description_. Defaults to "confusion matrix".
            step (int, optional): _description_. Defaults to 1.
        """
        super().__init__(train = False, test = True, aggregate_func = _aggregate_list_of_confusion_matrix)
        self.include_bg = include_bg
        self.metric = name
        self.test_cond = None if step<=1 else lambda _, i: i%step==0


    def __call__(self, learner: Learner):
        return (
            monai.metrics.get_confusion_matrix( # type:ignore
                one_hot_mask(learner.preds.argmax(1), learner.preds.shape[1]).swapaxes(0, 1),
                learner.targets,
                include_background=self.include_bg,
            ).sum(0).detach().cpu().numpy()
        )


MONAI_CM_METRICS = (
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "negative predictive value",
    "miss rate",
    "fall out",
    "false discovery rate",
    "false omission rate",
    "prevalence threshold",
    "threat score",
    "accuracy",
    "balanced accuracy",
    "f1 score",
    "matthews correlation coefficient",
    "fowlkes mallows index",
    "informedness",
    "markedness",
)
class MONAI_ConfusionMatrixMetricsMean(CBMethod):
    order = 1
    def __init__(self, metrics = MONAI_CM_METRICS, key = 'confusion matrix', prefix=''):
        super().__init__()
        self.metrics = metrics
        self.key = key
        self.prefix = prefix

    # def after_train_batch(self, learner:Learner):
    #     cm = learner.logger.last('train confusion matrix')
    #     for metric in self.metrics:
    #         learner.log(f'train {metric}', monai.metrics.compute_confusion_matrix_metric(metric, cm).mean()) # type:ignore

    def after_test_epoch(self, learner:Learner):
        cm = learner.logger.last(f'test {self.key}')
        for metric in self.metrics:
            learner.log(f'test {self.prefix}{metric}', monai.metrics.compute_confusion_matrix_metric(metric, cm).mean()) # type:ignore


class MONAI_ConfusionMatrixMetrics(CBMethod):
    order = 1
    def __init__(self, classes:Sequence, metrics = MONAI_CM_METRICS, prefix='', include_bg=True):
        super().__init__()
        self.metrics = metrics
        self.prefix = prefix
        self.classes = classes
        self.include_bg = include_bg

        self.list_of_cm = []

    # def after_train_batch(self, learner:Learner):
    #     cm = learner.logger.last('train confusion matrix')
    #     for metric in self.metrics:
    #         learner.log(f'train {metric}', monai.metrics.compute_confusion_matrix_metric(metric, cm).mean()) # type:ignore

    def before_test_epoch(self, learner:Learner):
            self.list_of_cm = []

    def after_test_batch(self, learner:Learner):
        self.list_of_cm.append(
            monai.metrics.get_confusion_matrix( # type:ignore
                one_hot_mask(learner.preds.argmax(1), learner.preds.shape[1]).swapaxes(0, 1),
                learner.targets,
                include_background=self.include_bg,
            )
        )
    def after_test_epoch(self, learner:Learner):
        cm = torch.cat(self.list_of_cm, 0)
        for metric in self.metrics:
            values = monai.metrics.compute_confusion_matrix_metric(metric, cm).nanmean(0) # type:ignore
            for cls, val in zip(self.classes, values):
                if val is not None:
                    learner.log(f'test {self.prefix}{metric} - {cls}', val) # type:ignore

            learner.log(f'test {self.prefix}{metric} - mean', values.nanmean()) # type:ignore