"""Docstring """
import os, shutil
from collections.abc import Mapping, Sequence
from ..design.EventModel import ConditionCallback, EventCallback
from .Learner import Learner
from ..python_tools import int_at_beginning

__all__ = [
    "SaveBestCB",
    "SaveLastCB",
]
class SaveBestCB(EventCallback):
    event = "after_test_epoch"
    order = 1 # needs to run after metrics are calculated
    def __init__(
        self,
        dir="runs",
        keep_old = False,
        serialize=False,
        metrics: Mapping[str, str] = {"test loss": "low", "test accuracy": "high"},
    ):  # pylint:disable=W0102
        super().__init__()
        self.dir = dir
        self.keep_old = keep_old
        self.serialize = serialize
        if isinstance(metrics, Sequence): metrics = {m: ("low" if "loss" in m else "high") for m in metrics}
        if not isinstance(metrics, Mapping): metrics = {m: ("low" if "loss" in m else "high") for m in (metrics, )}

        self.metrics = {k:v.lower() for k,v in metrics.items()}
        self.best_metrics = {k:float("inf") if v == "low" else -float("inf") for k,v in metrics.items()}

        self.best_paths = {}


    def __call__(self, learner: Learner):
        # make folders
        checkpoint_path = f'{learner.get_prefix_epochbatch_dir(self.dir,  "checkpoints")}'

        # avoid saving multiple checkpoints if multiple metrics improved
        is_already_saved = False

        # iterate over metrics
        for met, target in self.metrics.items():
            # check if metric in logger
            if met in learner.logger:
                # get last value
                val = learner.logger.last(met)
                # if lowest means best
                if target == "low":
                    # if last value is lower
                    if  val < self.best_metrics[met]:
                        # if not keep old, remove last checkpoint for this metric
                        if (not self.keep_old) and met in self.best_paths: shutil.rmtree(self.best_paths[met])
                        # save checkpoint
                        if not is_already_saved: learner.checkpoint(dir = checkpoint_path, serialize=self.serialize, mkdir=True)
                        # save new best value
                        self.best_metrics[met] = val
                        # save path to this metrics checkpoint
                        self.best_paths[met] = checkpoint_path
                        is_already_saved = True
                else:
                    if val > self.best_metrics[met]:
                        # if not keep old, remove last checkpoint for this metric
                        if (not self.keep_old) and met in self.best_paths: shutil.rmtree(self.best_paths[met])
                        # save checkpoint
                        if not is_already_saved: learner.checkpoint(dir = checkpoint_path, serialize=self.serialize, mkdir=True)
                        # save new best value
                        self.best_metrics[met] = val
                        # save path to this metrics checkpoint
                        self.best_paths[met] = checkpoint_path
                        is_already_saved = True

class SaveLastCB(EventCallback):
    event = "after_fit"
    order = 1 # needs to run after metrics are calculated
    def __init__(self, dir = "runs", serialize=False): #pylint:disable=W0102
        super().__init__()
        self.dir = dir
        self.serialize = serialize

    def __call__(self, learner: Learner):
        learner.checkpoint(dir = f'{learner.get_prefix_epochbatch_dir(self.dir,  "checkpoints")}', serialize=self.serialize, mkdir=True)
