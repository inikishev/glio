"""Docstring """
from typing import Literal, Optional
from collections.abc import Sequence
import torch
from ..design.event_model import MethodCallback
from .Learner import Learner

__all__ = [
    "SchedulerOneCycleLRCB",
    "SchedulerStopCB",
]


class SchedulerOneCycleLRCB(MethodCallback):
    def __init__(self,
    max_lr: float | list[float],
    pct_start: float = 0.3,
    anneal_strategy: Literal['cos', 'linear'] = 'cos',
    cycle_momentum: bool = True,
    base_momentum: float | list[float] = 0.85,
    max_momentum: float | list[float] = 0.95,
    div_factor: float = 25,
    final_div_factor: float = 1e4,
    three_phase: bool = False,
    last_epoch: int = -1,
    verbose: bool = False,
) -> None:
        super().__init__()

        self.kwargs = dict(
            max_lr=max_lr,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def before_fit(self, learner:Learner):
        learner.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            learner.optimizer, # type:ignore
            epochs = learner.num_epochs,
            steps_per_epoch = len(learner.dltrain), # type:ignore
            **self.kwargs # type:ignore
        )

class SchedulerStopCB(MethodCallback):
    def __init__(self,
    epoch: Optional[int] = None,
    batch: Optional[int] = None,
    total_batch: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.epoch = epoch
        self.batch = batch
        self.total_batch = total_batch

    def after_batch(self, learner:Learner):
        if (learner.cur_epoch == self.epoch and learner.cur_batch == self.batch) or learner.total_batch == self.total_batch:
            learner.scheduler = None