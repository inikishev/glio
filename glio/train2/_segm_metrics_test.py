
import statistics
from stuff.found.segmentation_metrics import SegmentationMetrics
from .Learner import Learner
from .cbs_metrics import CBMetric

class Metrics_Segm(CBMetric):
    def __init__(self, every = 1):
        self.metrics = SegmentationMetrics()
        self.pixel_acc = []
        self.dice = []
        self.precision = []
        self.recall = []
        super().__init__(train = True, test = True, aggregate_func = statistics.mean, train_cond= None if every == 1 else lambda _,x: x % every == 0)

    def __call__(self, learner:Learner): ...
    def batch(self, learner: "Learner"):
        # training
        pixel_acc, dice, precision, recall = self.metrics(learner.preds, learner.targets)
        if learner.status == "train" and self.train:
            learner.log("train pixel accuracy", pixel_acc)
            learner.log("train dice similarity", dice)
            learner.log("train precision", precision)
            learner.log("train recall", recall)
        # testing
        elif learner.status == "test" and self.test:
            self.pixel_acc.append(pixel_acc)
            self.dice.append(dice)
            self.precision.append(precision)
            self.recall.append(recall)

    def epoch(self, learner: "Learner"):
        if len(self.test_metrics) > 0:
            learner.log("test pixel accuracy", self.aggregate_func(self.pixel_acc))
            learner.log("test dice similarity", self.aggregate_func(self.dice))
            learner.log("test precision", self.aggregate_func(self.precision))
            learner.log("test recall", self.aggregate_func(self.recall))
            self.pixel_acc = []
            self.dice = []
            self.precision = []
            self.recall = []