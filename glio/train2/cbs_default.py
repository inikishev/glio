
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING, Optional
from contextlib import nullcontext
import torch, torch.utils.data
from ..design.EventModel import Callback, CBEvent
from ..python_tools import SupportsIter
from ..torch_tools import to_device, smart_detach, smart_detach_cpu
if TYPE_CHECKING:
    from .Learner import Learner

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

class Default_SetMode(CBEvent):
    event = "set_mode"
    def __call__(self, learner: "Learner", train=True):
        if hasattr(learner.model, "train") and hasattr(learner.optimizer, "eval"):
            if train: learner.model.train()
            else: learner.model.eval()
        # if hasattr(learner.optimizer, "train") and hasattr(learner.optimizer, "eval"):
        #     if train: learner.optimizer.train() # type:ignore
        #     else: learner.optimizer.eval() # type:ignore

class Default_OneBatch(CBEvent):
    event = "one_batch"
    def __call__(self, learner: "Learner", inputs: torch.Tensor, targets: torch.Tensor, train=True):
        learner.set_mode(train)
        if learner.accelerator is None: inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        with nullcontext() if train else torch.no_grad():

            # get predictions
            learner.preds = learner.model(inputs)

            # calculate loss
            learner.loss = learner.loss_fn(learner.preds, targets) # pylint: disable=E1102 # type:ignore

            # backprop
            if train:
                learner.zero_grad() # type:ignore
                learner.backward()
                learner.optimizer_step()
                learner.scheduler_step()


class Default_Inference(CBEvent):
    event = "inference"
    def __call__(self, learner: "Learner", batch: torch.Tensor| Any, to_cpu = True):
        learner.set_mode(False)
        batch = to_device(batch, learner.device)
        with torch.no_grad():
            if to_cpu: return smart_detach_cpu(learner.model(batch))
            return smart_detach(learner.model(batch))

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
        num_epochs: int,
        dl_train: Optional[torch.utils.data.DataLoader | Any] = None,
        dl_test: Optional[torch.utils.data.DataLoader | Any] = None,
        test_first=True,
        catch_interrupt=True,
        test_on_interrupt = True,
        always_after_fit = True,
        extra:Optional[Callback | Iterable[Callback]] = None,
        without:Optional[str | Iterable[str]] = None
    ):
        # Присваиваем аттрибуты
        learner.dl_train = dl_train
        learner.dl_test = dl_test
        learner.test_first = test_first
        learner.catch_interrupt = catch_interrupt
        if catch_interrupt: learner.catch_fit_exceptions = KeyboardInterrupt
        else: learner.catch_fit_exceptions = ()
        learner.test_on_interrupt = test_on_interrupt
        learner.always_after_fit = always_after_fit
        learner.num_epochs = num_epochs
        learner.epochs_iterator = range(learner.num_epochs)
        learner.cur_epoch = 0
        if learner.accelerator is None: learner.model = to_device(learner.model, learner.device) # type:ignore


        # Запускаем цикл обучения и тестирования
        with learner.context("fit", extra=extra, without=without):
            try:
                learner.event("before_fit")
                # итерация по обучающей выборке и тестированию, если они есть.
                for learner.cur_epoch in learner.epochs_iterator:
                    learner.event("before_epoch")
                    with learner.context("full_epoch"):
                    # тестирование в начале
                        if learner.cur_epoch == 0 and learner.test_first and learner.dl_test is not None:
                            learner.one_epoch(learner.dl_test, train=False)

                        # обучение
                        if learner.dl_train is not None:
                            learner.one_epoch(learner.dl_train, train=True)

                        # тестирование
                        if learner.dl_test is not None:
                            learner.one_epoch(learner.dl_test, train=False)

                        learner.total_epoch += 1
                    learner.event("after_epoch")

            except learner.catch_fit_exceptions:
                if learner.test_on_interrupt and learner.status == "train" and learner.dl_test is not None:
                    print("Keyboard interrupt, testing one last time... Press stop again to cancel.")
                    try: learner.one_epoch(learner.dl_test, train=False)
                    except learner.catch_fit_exceptions: print("Keyboard interrupt, stopping testing...")
                else: print("Keyboard interrupt, stopping the training...")
            finally: 
                if learner.always_after_fit or (not learner.catch_interrupt): learner.event("after_fit")

class Default_Log(CBEvent):
    event = "log"
    def __call__(self, learner:"Learner", metric:str, value):
        learner.logger.add(metric, value, learner.total_batch)
