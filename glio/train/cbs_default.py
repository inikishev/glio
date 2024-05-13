from typing import Optional,TYPE_CHECKING, Any
from collections.abc import Iterable
from contextlib import nullcontext
import torch, torch.utils.data
from ..torch_tools import to_device
from ..python_tools import SupportsIter
from ..design.CallbackModel import Callback
if TYPE_CHECKING:
    from .learner import Learner



class Default_Backward(Callback):
    DEFAULT = True
    def backward(self, learner: "Learner"):
        if learner.accelerator is None: learner.loss.backward()
        else: learner.accelerator.backward(learner.loss)

class Default_OptimizerStep(Callback):
    DEFAULT = True
    def optimizer_step(self, learner: "Learner"):
        learner.optimizer.step() # type:ignore

class Default_ZeroGrad(Callback):
    DEFAULT = True
    def zero_grad(self, learner: "Learner"):
        learner.optimizer.zero_grad() # type:ignore

class Default_SchedulerStep(Callback):
    DEFAULT = True
    def scheduler_step(self, learner: "Learner"):
        if learner.scheduler is not None:
            learner.scheduler.step()

class Default_OneBatch(Callback):
    DEFAULT = True
    def one_batch(self, learner: "Learner", inputs: torch.Tensor, targets: torch.Tensor, train=True):
        if train: learner.model.train()
        else: learner.model.eval()
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

class Default_Inference(Callback):
    DEFAULT = True
    def inference(self, learner: "Learner", batch: torch.Tensor| Any, to_cpu = True):
        learner.model.eval()
        batch = to_device(batch, learner.device)
        with torch.no_grad():
            if to_cpu: return learner.model(batch).cpu().detach()
            return learner.model(batch).detach()

class Default_OneEpoch(Callback):
    DEFAULT = True
    def one_epoch(self, learner: "Learner", dl: torch.utils.data.DataLoader | SupportsIter, train=True):
        for learner.cur_batch, (inputs, targets) in enumerate(dl): # type:ignore
            learner.one_batch(inputs, targets, train=train)

class Default_Fit(Callback):
    DEFAULT = True

    def fit(
        self,
        learner:"Learner",
        num_epochs: int,
        dl_train: Optional[torch.utils.data.DataLoader | Any] = None,
        dl_test: Optional[torch.utils.data.DataLoader | Any] = None,
        test_first=True,
        test_on_interrupt = True,
        extra:Optional[Callback | Iterable[Callback]] = None,
        without:Optional[str | Iterable[str]] = None
    ):
        # Присваиваем аттрибуты
        learner.dl_train = dl_train
        learner.dl_test = dl_test
        learner.test_first = test_first
        learner.test_on_interrupt = test_on_interrupt
        learner.num_epochs = num_epochs
        learner.epochs_iterator = range(learner.num_epochs)
        learner.cur_epoch = 0
        if learner.accelerator is None: learner.model = to_device(learner.model, learner.device) # type:ignore

        # Запускаем цикл обучения и тестирования
        with learner.context("fit", extra=extra, without=without):
            try:
                # итерация по обучающей выборке и тестированию, если они есть.
                for learner.cur_epoch in learner.epochs_iterator:
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

            except KeyboardInterrupt:
                if learner.test_on_interrupt and learner.status == "train" and learner.dl_test is not None:
                    print("Keyboard interrupt, testing one last time... Press stop again to cancel.")
                    try: learner.one_epoch(learner.dl_test, train=False)
                    except KeyboardInterrupt: print("Keyboard interrupt, stopping testing...")
                else: print("Keyboard interrupt, stopping the training...")

class Default_Log(Callback):
    DEFAULT = True
    def log(self, learner:"Learner", metric:str, value):
        learner.logger.add(metric, value, learner.total_batch)
