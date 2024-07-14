from collections.abc import Sequence
import numpy as np
import torch
import torch.nn.functional as F

from torchzero.metrics.dice import dice, softdice
from torchzero.metrics.iou import iou
from torchzero.metrics.accuracy import accuracy

from ..Learner import Learner
from ...design.EventModel import MethodCallback
from ...torch_tools import batched_raw_preds_to_one_hot
from .metric_callback import PerClassMetricCallback

__all__ = [
    "TorchzeroDiceCB",
    "TorchzeroSoftdiceCB",
    "TorchzeroIoUCB",
    "TorchzeroAccuracyCB",
]

class TorchzeroDiceCB(PerClassMetricCallback):
    def __init__(self, class_labels:Sequence, include_bg = False, train=True, test=True, step=1, name='dice'):
        """Sørensen–Dice coefficient often used for segmentation.
        Defined as two intersections over sum.
        Equivalent to F1 score in the case of binary segmentation.

        preds must be raw BC(*) format, targets - one-hot BC(*).
        """
        super().__init__(class_labels = class_labels, agg_ignore_bg = not include_bg, train = train, test = test)
        self.class_labels = class_labels

        self.batch_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        return dice(y = learner.targets, yhat = batched_raw_preds_to_one_hot(learner.preds), reduction = 'none').detach().cpu()

class TorchzeroSoftdiceCB(PerClassMetricCallback):
    def __init__(self, class_labels:Sequence, include_bg = False, train=True, test=True, step=1, name= 'softdice'):
        """Sørensen–Dice coefficient often used for segmentation.
        Defined as two intersections over sum.
        Equivalent to F1 score in the case of binary segmentation.

        preds must be raw BC(*) format, targets - one-hot BC(*).
        """
        super().__init__(class_labels = class_labels, agg_ignore_bg = not include_bg, train = train, test = test)
        self.class_labels = class_labels

        self.batch_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        return softdice(y = learner.targets, yhat = learner.preds, reduction = 'none').detach().cpu()

class TorchzeroIoUCB(PerClassMetricCallback):
    def __init__(self, class_labels:Sequence, include_bg = False, train=True, test=True, step=1, name='iou'):
        """Intersection over union metric often used for segmentation, also known as Jaccard index.

        preds must be raw BC(*) format, targets - one-hot BC(*).
        """
        super().__init__(class_labels = class_labels, agg_ignore_bg = not include_bg, train = train, test = test)
        self.class_labels = class_labels

        self.batch_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        return iou(y = learner.targets, yhat = batched_raw_preds_to_one_hot(learner.preds), reduction = 'none').detach().cpu()

class TorchzeroAccuracyCB(PerClassMetricCallback):
    def __init__(self, class_labels:Sequence, include_bg = False, train=True, test=True, step=1, name='accuracy'):
        """Accuracy metric. Defined as number of correct predictions divided by total number of predictions.

        preds must be raw BC(*) format, targets - one-hot BC(*).
        """
        super().__init__(class_labels = class_labels, agg_ignore_bg = not include_bg, train = train, test = test)
        self.class_labels = class_labels

        self.batch_cond = None if step<=1 else lambda _, i: i%step==0
        self.metric = name

    def __call__(self, learner: Learner):
        return accuracy(y = learner.targets, yhat = batched_raw_preds_to_one_hot(learner.preds), reduction = 'none').detach().cpu()