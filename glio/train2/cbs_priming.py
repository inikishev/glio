import torch
from itertools import zip_longest
from ..design.EventModel import CBCond
from .Learner import Learner
from ..torch_tools import get_lr, set_lr as set_lr_, change_lr, copy_state_dict, smart_to_float
from ..python_tools import EndlessContinuingIterator

class LRFinderPriming(CBCond):
    def __init__(
        self,
        dl,
        start=1e-6,
        mul=1.3,
        add=0,
        stop=1,
        max_increase=3,
        niter=2,
        set_model=True,
        set_lr=True,
        load_best_state=False,
        save_curve = True,
    ):
        super().__init__()
        self.dl, self.start, self.mul, self.add, self.stop, self.max_increase, self.niter = dl, start, mul, add, stop, max_increase, niter
        self.set_model, self.set_lr, self.load_best_state = set_model, set_lr, load_best_state
        self.save_curve = save_curve

    def __call__(self, learner:Learner):

        backup = copy_state_dict(
            learner.state_dict(
                attrs_state_dict = ("model", "optimizer"),
            )
        )

        dl_iter = EndlessContinuingIterator(learner.dl_train)

        best_lr = get_lr(learner.optimizer) # type:ignore
        min_loss = float("inf")
        max_dloss = -float("inf")
        best_state = backup
        iter_losses = []

        for _ in range(self.niter):
            losses = []
            set_lr_(learner.optimizer, self.start) # type:ignore
            for inputs, targets in dl_iter:
                learner.one_batch(inputs, targets, True)
                loss = smart_to_float(learner.loss)
                if len(losses) > 0:
                    if max_dloss < losses[-1] - loss:
                        max_dloss = losses[-1] - loss
                        best_lr = get_lr(learner.optimizer) # type:ignore
                losses.append(loss)

                if loss < min_loss:
                    min_loss = loss
                    best_state = copy_state_dict(learner.state_dict())

                change_lr(learner.optimizer, lambda x: x * self.mul + self.add) # type:ignore
                if (self.stop is not None and get_lr(learner.optimizer) > self.stop) or (self.max_increase is not None and loss/min_loss > self.max_increase): # type:ignore
                    break
            iter_losses.append(losses)
            if self.load_best_state: learner.load_state_dict(copy_state_dict(best_state))
            else: learner.load_state_dict(copy_state_dict(backup))

        if self.set_model: learner.load_state_dict(best_state)
        else: learner.load_state_dict(backup)
        if self.set_lr: set_lr_(learner.optimizer, best_lr) # type:ignore
        if self.save_curve:
            avg_losses = [[j for j in i if j is not None] for i in zip_longest(*iter_losses)]
            avg_losses = [sum(i)/len(i) for i in avg_losses]
            learner.log("lr finder losses", avg_losses)

class IterLR(CBCond):
    def __init__(self, lrs = (0.5, 1, 1.5)):
        super().__init__()
        self.lrs = lrs

    def __call__(self, learner:Learner):
        with learner.without(["IterLR", "FastProgressBar"]):
            backup = copy_state_dict(learner.state_dict(attrs_state_dict = ("model", "optimizer"),))
            start_lr = get_lr(learner.optimizer) # type:ignore
            start_loss = smart_to_float(learner.loss)
            max_dloss = -float("inf")
            best_lr = start_lr

            for lr in self.lrs:
                if callable(lr): lr = lr()
                new_lr = start_lr * lr
                set_lr_(learner.optimizer, new_lr) # type:ignore pylint:disable=W0640
                learner.one_batch(learner.inputs, learner.targets, True) # do an optimization step, has loss from before the optimization step
                learner.one_batch(learner.inputs, learner.targets, False) # calculate the loss after optimization step
                dloss = start_loss - smart_to_float(learner.loss)

                if dloss > max_dloss:
                    max_dloss = dloss
                    best_lr = new_lr # type:ignore

                learner.load_state_dict(copy_state_dict(backup))

            set_lr_(learner.optimizer, best_lr) # type:ignore pylint:disable=W0640
            # learner.one_batch(learner.inputs, learner.targets, True)