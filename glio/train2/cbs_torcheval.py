import statistics

import torch.nn.functional as F
import torcheval.metrics.functional as tmf

from .Learner import Learner
from ..design.EventModel import Callback, EventModel
from .cbs_metrics import CBMetric

class Torcheval_Precision(CBMetric):
    """Точность (правильно предсказанные / общее кол-во предсказанных)"""
    def __init__(self, num_classes, argmax_preds = True, argmax_targets = False, average='micro', step=1, name="precision"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.average = average
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds: preds = learner.preds.argmax(1).flatten()
        else: preds = learner.preds.flatten()
        if self.argmax_targets: targets = learner.targets.argmax(1).flatten()
        else: targets = learner.targets.flatten()
        return float(tmf.multiclass_precision(preds, targets, average = self.average, num_classes=self.num_classes))

class Torcheval_Recall(CBMetric):
    """Полнота (правильно предсказанные / общее кол-во пикселей с классом)"""
    def __init__(self, num_classes, argmax_preds = True, argmax_targets = False, average='micro', step=1, name="recall"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.average = average
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds: preds = learner.preds.argmax(1).flatten()
        else: preds = learner.preds.flatten()
        if self.argmax_targets: targets = learner.targets.argmax(1).flatten()
        else: targets = learner.targets.flatten()
        return float(tmf.multiclass_recall(preds, targets, average = self.average, num_classes=self.num_classes))

class Torcheval_Dice(CBMetric):
    """Коэффициент Сёренсена (пересечение к общему кол-ву)"""
    def __init__(self, num_classes, argmax_preds = True, argmax_targets = False, average='micro', step=1, name="dice"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_preds, self.argmax_targets = argmax_preds, argmax_targets
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.average = average
        self.metric = name

    def __call__(self, learner: Learner):
        if self.argmax_preds: preds = learner.preds.argmax(1).flatten()
        else: preds = learner.preds.flatten()
        if self.argmax_targets: targets = learner.targets.argmax(1).flatten()
        else: targets = learner.targets.flatten()
        return float(tmf.multiclass_recall(preds, targets, average = self.average, num_classes=self.num_classes))


class Torcheval_AURPC(CBMetric):
    """Средняя точность (площадь под кривой точность-полнота)"""
    def __init__(self, num_classes, argmax_targets = False, average='macro', step=1, teststep = 1, name="average precision"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_targets = argmax_targets
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.test_cond = None if teststep<=1 else lambda _, i: i%step==0
        self.average = average
        self.metric = name

    def __call__(self, learner: Learner):
        preds = learner.preds.swapaxes(1, -1).flatten(0,-2)
        if self.argmax_targets: targets =  learner.targets.argmax(1).flatten()
        else: targets = learner.targets.flatten()
        return float(tmf.multiclass_auprc(preds, targets, average = self.average, num_classes=self.num_classes))

class Torcheval_AUROC(CBMetric):
    """Площадь под ROC - кривой (истинно-положительные результаты к ложно-положительным)"""
    def __init__(self, num_classes, argmax_targets = False, average='macro', step=1,teststep = 1, name="roc auc"):
        super().__init__(train = True, test = True, aggregate_func = statistics.mean)
        self.num_classes = num_classes
        self.argmax_targets = argmax_targets
        self.train_cond = None if step<=1 else lambda _, i: i%step==0
        self.test_cond = None if teststep<=1 else lambda _, i: i%step==0
        self.average = average
        self.metric = name

    def __call__(self, learner: Learner):
        preds = learner.preds.swapaxes(1, -1).flatten(0,-2)
        if self.argmax_targets: targets =  learner.targets.argmax(1).flatten()
        else: targets = learner.targets.flatten()
        return float(tmf.multiclass_auroc(preds, targets, average = self.average, num_classes=self.num_classes))
