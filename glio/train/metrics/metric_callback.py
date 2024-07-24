"metrics"
from collections.abc import Callable, Mapping, Sequence, Iterable
import itertools
from typing import Literal
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import torch
from ..Learner import Learner
from ...design.EventModel import Callback, EventModel
from ...python_tools import get__name__

__all__ = [
    "MetricCallback",
    "PerClassMetricCallback",
]
class MetricCallback(Callback, ABC):
    metric:str

    def __init__( # pylint:disable=W0102
        self,
        train=True,
        test=True,
        aggregate_funcs: dict[str, Callable] = {"": np.nanmean, "min": np.nanmin, "max": np.nanmax, "median": np.nanmedian},
        batch_cond: Optional[Callable] = None,
        test_epoch_cond: Optional[Callable] = None,
        train_step = 1,
        test_step = 1,
    ):
        self.train, self.test = train, test
        self.aggregate_funcs = aggregate_funcs.copy()
        self.batch_cond, self.test_epoch_cond = batch_cond, test_epoch_cond
        self.train_step, self.test_step = train_step, test_step

        self.test_metrics = []

    @abstractmethod
    def __call__(self, learner: "Learner"):
        raise NotImplementedError(f"{self.__class__.__name__} is missing `__call__`")

    def batch(self, learner: "Learner"):
        # training
        if learner.status == "train" and self.train:
            with torch.no_grad():
                learner.log(f"train {self.metric}", self(learner))

        # testing
        elif learner.status == "test" and self.test:
            self.test_metrics.append(self(learner))

    def epoch(self, learner: "Learner"):
        if len(self.test_metrics) > 0:
            for name, func in self.aggregate_funcs.items():
                learner.log(f"test {self.metric}{(' ' + name) if len(name) > 0 else ''}", func(self.test_metrics))
            self.test_metrics = []

    def attach(self, learner:"EventModel"):
        learner._attach(event = "after_batch", fn = self.batch, cond = self.batch_cond, name = get__name__(self), ID = id(self))
        if self.test:
            learner._attach(event =  "after_test_epoch", fn = self.epoch, cond = self.test_epoch_cond, name = get__name__(self), ID = id(self))

    def attach_default(self, learner:"EventModel"):
        learner._attach_default(event = "after_batch", fn = self.batch, cond = self.batch_cond, name = get__name__(self), ID = id(self))
        if self.test:
            learner._attach_default(event =  "after_test_epoch", fn = self.epoch, cond = self.test_epoch_cond, name = get__name__(self), ID = id(self))


class PerClassMetricCallback(Callback, ABC):
    metric:str

    def __init__( # pylint:disable=W0102
        self,
        class_labels: Optional[Iterable[Any]],
        train=True,
        test=True,
        test_aggregate_funcs: dict[str, Callable] = {"": np.nanmean,},
        class_aggregate_func: dict[str, Callable] = {"mean": np.nanmean},
        train_step = 1,
        test_step = 1,
        agg_ignore_bg = False,
        log_per_class:bool | Literal['auto'] = 'auto',
        batch_cond: Optional[Callable] = None,
        test_epoch_cond: Optional[Callable] = None,
    ):
        """_summary_

        Args:
            class_labels (Sequence[str]): labels of each class, including background if it exists.
            train (bool, optional): Whether to log train metric. Defaults to True.
            test (bool, optional): Whether to log aggregated test metric. Defaults to True.
            test_aggregate_funcs (_type_, optional): Functions that aggregate all test values into one. Defaults to {"": np.nanmean,}.
            class_aggregate_func (_type_, optional): Functions that aggregate all class values into one. Defaults to {"mean": np.nanmean}.
            train_step (int, optional): Calculate metric every `train_step` on training. Defaults to 1.
            test_step (int, optional): Calculate metric every `test_step` on testing. Defaults to 1.
            agg_ignore_bg (bool, optional): Whether `class_aggregate_func` should ignore first class, which is assumed to be background. Defaults to False.
            log_per_class (bool, optional): Whether per-class metric should be logged, otherwise only aggregated channels metric is logged. Defaults to True.
            batch_cond (Optional[Callable], optional): after_batch condition. Defaults to None.
            test_epoch_cond (Optional[Callable], optional): after_test_epoch condition. Defaults to None.
        """
        if class_labels is None: self.class_labels = iter(lambda: None, 1)
        else: self.class_labels = class_labels
        self.agg_ignore_bg = agg_ignore_bg
        self.train, self.test = train, test
        self.test_aggregate_funcs = test_aggregate_funcs.copy()
        self.class_aggregate_func = class_aggregate_func.copy()
        self.batch_cond, self.test_epoch_cond = batch_cond, test_epoch_cond
        if log_per_class == 'auto': log_per_class = class_labels is not None
        self.log_per_class = log_per_class
        self.train_step, self.test_step = train_step, test_step
        self.cur = 0

        self.test_metrics = []

    @abstractmethod
    def __call__(self, learner: "Learner"):
        raise NotImplementedError(f"{self.__class__.__name__} is missing `__call__`")

    def batch(self, learner: "Learner"):
        # training
        all_values = []
        if learner.status == "train" and self.train and self.cur % self.train_step == 0:
            with torch.no_grad():
                for label, value in zip(self.class_labels, self(learner)):
                    all_values.append(value)
                    if self.log_per_class: learner.log(f"train {self.metric} - {label}", value)
                start = int(self.agg_ignore_bg)
                for name, fn in self.class_aggregate_func.items():
                    learner.log(f"train {self.metric} {name}", fn(all_values[start:]))

        # testing
        elif learner.status == "test" and self.test and self.cur % self.test_step == 0:
            self.test_metrics.append(self(learner))

        self.cur += 1

    def epoch(self, learner: "Learner"):
        if len(self.test_metrics) > 0:
            start = int(self.agg_ignore_bg)
            for agg_fn_name, agg_fn in self.test_aggregate_funcs.items():
                all_values = []
                for label, values in zip(self.class_labels, zip(*self.test_metrics)):
                    if not np.isnan(values).all():
                        aggregated = agg_fn(values)
                        all_values.append(aggregated)
                        if self.log_per_class: learner.log(f"test {self.metric} - {label}{(' (' + agg_fn_name + ')') if len(agg_fn_name) > 0 else ''}", aggregated)
                for ch_fn_name, ch_fn in self.class_aggregate_func.items():
                    learner.log(f"test {self.metric} {ch_fn_name}{(' (' + agg_fn_name + ')') if len(agg_fn_name) > 0 else ''}", ch_fn(all_values[start:]))
            self.test_metrics = []

    def attach(self, learner:"EventModel"):
        learner._attach(event = "after_batch", fn = self.batch, cond = self.batch_cond, name = get__name__(self), ID = id(self))
        if self.test:
            learner._attach(event =  "after_test_epoch", fn = self.epoch, cond = self.test_epoch_cond, name = get__name__(self), ID = id(self))

    def attach_default(self, learner:"EventModel"):
        learner._attach_default(event = "after_batch", fn = self.batch, cond = self.batch_cond, name = get__name__(self), ID = id(self))
        if self.test:
            learner._attach_default(event =  "after_test_epoch", fn = self.epoch, cond = self.test_epoch_cond, name = get__name__(self), ID = id(self))