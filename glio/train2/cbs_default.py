
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING, Optional
from contextlib import nullcontext
import torch, torch.utils.data
from ..design.EventModel import Callback, CBEvent
from ..python_tools import SupportsIter
from ..torch_tools import to_device, smart_detach, smart_detach_cpu
if TYPE_CHECKING:
    from .Learner import Learner


class Default_Forward(CBEvent):
    event = "forward"
    def __call__(self, learner: "Learner", inputs: torch.Tensor):
        learner.preds = learner.model(inputs)
        return learner.preds

class Default_GetLoss(CBEvent):
    event = "get_loss"
    def __call__(self, learner: "Learner", preds:torch.Tensor, targets: torch.Tensor):
        learner.loss = learner.loss_fn(preds, targets) # type:ignore
        return learner.loss

class Default_Backward(CBEvent):
    event = "backward"
    def __call__(self, learner: "Learner"):
        if learner.accelerator is None: learner.loss.backward()
        else: learner.accelerator.backward(learner.loss)

class Default_OptimizerStep(CBEvent):
    event = "optimizer_step"
    def __call__(self, learner: "Learner", *args, **kwargs):
        learner.optimizer.step(*args, **kwargs) # type:ignore

class Default_ZeroGrad(CBEvent):
    event = "zero_grad"
    def __call__(self, learner: "Learner"):
        learner.optimizer.zero_grad() # type:ignore

class Default_SchedulerStep(CBEvent):
    event = "scheduler_step"
    def __call__(self, learner: "Learner"):
        if learner.scheduler is not None:
            learner.scheduler.step()

# class Default_SetMode(CBEvent):
#     event = "set_mode"
#     def __call__(self, learner: "Learner", train=True):
#         if hasattr(learner.model, "train") and hasattr(learner.optimizer, "eval"):
#             if train: learner.model.train()
#             else: learner.model.eval()
#         # if hasattr(learner.optimizer, "train") and hasattr(learner.optimizer, "eval"):
#         #     if train: learner.optimizer.train() # type:ignore
#         #     else: learner.optimizer.eval() # type:ignore

class Default_Train(CBEvent):
    event = "train"
    def __call__(self, learner: "Learner"):
        if hasattr(learner.model, "train") and callable(learner.model.train): learner.model.train()
        if hasattr(learner.optimizer, "train") and callable(learner.optimizer.train): learner.optimizer.train() # type:ignore

class Default_Eval(CBEvent):
    event = "eval"
    def __call__(self, learner: "Learner"):
        if hasattr(learner.model, "eval") and callable(learner.model.eval): learner.model.eval()
        if hasattr(learner.optimizer, "eval") and callable(learner.optimizer.eval): learner.optimizer.eval() # type:ignore

class Default_OneBatch(CBEvent):
    event = "one_batch"
    def __call__(self, learner: "Learner", inputs: torch.Tensor, targets: torch.Tensor, train=True):
        learner.train()
        if learner.accelerator is None: inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        with nullcontext() if train else torch.no_grad():

            # get predictions
            learner.forward(inputs)

            # calculate loss
            learner.get_loss(learner.preds, targets)

            # backprop
            if train:
                learner.zero_grad() # type:ignore
                learner.backward()
                learner.optimizer_step()
                learner.scheduler_step()


class Default_Inference(CBEvent):
    event = "inference"
    def __call__(self, learner: "Learner", batch: torch.Tensor| Any, to_cpu = True):
        learner.eval()
        batch = to_device(batch, learner.device)
        with torch.no_grad():
            if to_cpu: return smart_detach_cpu(learner.forward(batch))
            return smart_detach(learner.forward(batch))

class Default_OneEpoch(CBEvent):
    event = "one_epoch"
    def __call__(self, learner: "Learner", dl: torch.utils.data.DataLoader | SupportsIter, train=True):
        for learner.cur_batch, (inputs, targets) in enumerate(dl): # type:ignore
            learner.one_batch(inputs, targets, train=train)


class Default_Fit(CBEvent):
    event = "fit"
    def __call__(
        self,
        learner:"Learner",
        epochs_iterator,
        dltrain: Optional[torch.utils.data.DataLoader | Any] = None,
        dltest: Optional[torch.utils.data.DataLoader | Any] = None,
        test_first = False,
        test_every: int = 1,
    ):

        for learner.cur_epoch in epochs_iterator:
            learner.event("before_epoch")
            with learner.context("full_epoch"):
            # тестирование в начале
                if learner.cur_epoch == 0 and test_first and dltest is not None:
                    learner.one_epoch(dltest, train=False)

                # обучение
                if dltrain is not None:
                    learner.one_epoch(dltrain, train=True)

                # тестирование
                if dltest is not None and learner.cur_epoch % test_every == 0:
                    learner.one_epoch(dltest, train=False)

                learner.total_epoch += 1
            learner.event("after_epoch")



class Default_Log(CBEvent):
    event = "log"
    def __call__(self, learner:"Learner", metric:str, value):
        learner.logger.add(metric, value, learner.total_batch)
