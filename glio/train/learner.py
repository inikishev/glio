"""Whatever"""
from collections.abc import Callable, Iterable, Sequence
from typing import Optional,TYPE_CHECKING, Any
import os
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import torch, torch.utils.data, numpy as np

from ..design.CallbackModel import CallbackModel, Callback
from ..logger.logger import Logger
from ..torch_tools import CUDA_IF_AVAILABLE, copy_state_dict, to_device
from..python_tools import try_copy, type_str, SupportsIter

from .cbs_default import (
    Default_Backward,
    Default_OptimizerStep,
    Default_ZeroGrad,
    Default_SchedulerStep,
    Default_OneBatch,
    Default_Inference,
    Default_OneEpoch,
    Default_Fit,
    Default_Log,
)

if TYPE_CHECKING:
    from accelerate import Accelerator

DEFAULT_CBS = (
    Default_Backward(),
    Default_OptimizerStep(),
    Default_ZeroGrad(),
    Default_SchedulerStep(),
    Default_OneBatch(),
    Default_Inference(),
    Default_OneEpoch(),
    Default_Fit(),
    Default_Log(),
)

class Learner(CallbackModel):
    """Learner"""

    def __init__(
        self,
        model: torch.nn.Module,
        name: str,
        cbs: Optional[Iterable[Callback]],
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional["Accelerator"] = None,
        device: Optional[torch.device] = CUDA_IF_AVAILABLE,
        logger: Optional[Logger] = None,
        default_cbs: Optional[Iterable[Callback]] = DEFAULT_CBS,
    ):
        self.model: torch.nn.Module = model
        self.name: str = name
        self.optimizer: Optional[torch.optim.Optimizer] = optimizer
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = scheduler
        self.loss_fn: Optional[Callable] = loss_fn
        self.accelerator: Optional["Accelerator"] = accelerator
        self.device: Optional[torch.device] = device
        self.logger: Logger = logger if logger is not None else Logger()

        self.cur_batch:int = 0
        """Current batch in epoch."""
        self.cur_epoch:int = 0
        """Current epoch in fit."""

        self.total_batch:int = 0
        """Total batches this has trained on."""
        self.total_epoch:int = 0
        """Total epochs this has trained for."""

        self.status:str = "init"
        """Current status of the Learner: `init`, `train`, `test`, `inference`"""

        # we init after so that all callbacks have access to the model
        super().__init__(cbs, default_cbs)

    def __type_hinting(self): #pylint:disable=W0238
        self.preds: torch.Tensor | Any = None
        self.inputs: torch.Tensor | Any = None
        self.targets: torch.Tensor | Any = None
        self.loss:torch.Tensor | Any = None
        self.dl_train: torch.utils.data.DataLoader | SupportsIter | Any = None
        self.dl_test: torch.utils.data.DataLoader | SupportsIter | Any = None
        self.test_first: bool = True
        self.test_on_interrupt: bool = True
        self.num_epochs: int = 10
        self.epochs_iterator: Iterable = range(10)

    def backward(self):
        self._run_main("backward")

    def zero_grad(self):
        self._run_main("zero_grad")

    def optimizer_step(self):
        self.event("before_optimizer_step")
        self._run_main("optimizer_step")

    def scheduler_step(self):
        self._run_main("scheduler_step")

    def one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, train=True):
        # Перемещение на графический ускоритель, если это не производится Hugging Face Accelerate
        if self.accelerator is None:
            self.inputs = to_device(inputs, self.device)
            self.targets = to_device(targets, self.device)
        else:
            self.inputs, self.targets = inputs, targets

        # получаем предсказания
        self.event("before_batch")
        results = self._run_main("one_batch", self.inputs, self.targets, train=train) # type:ignore

        # присваиваем предсказания
        if results is not None:
            if isinstance(results, tuple): self.preds, self.loss = results # pylint:disable=W0632
            elif isinstance(results, dict):
                for k, v in results.items(): setattr(self, k, v) # type:ignore
            else: raise ValueError(f"one_batch должен возвращать кортеж, словарь, или None, а не {type(results)}")
        self.event("after_batch")

        if self.status == "train": self.total_batch += 1

    def inference(self, batch, to_cpu = True):
        self.status = "inference"
        return self._run_main("inference", batch, to_cpu)[-1]

    def one_epoch(self, dl: torch.utils.data.DataLoader | SupportsIter, train=True):
        self.dl = dl
        self.status = "train" if train else "test"

        with self.context("epoch"):
            self._run_main("one_epoch", self.dl, train)

    def fit(
        self,
        num_epochs,
        dl_train: Optional[torch.utils.data.DataLoader | SupportsIter] = None,
        dl_test: Optional[torch.utils.data.DataLoader | SupportsIter] = None,
        test_first=True,
        test_on_interrupt = True,
        extra:Optional[Callback | Iterable[Callback]] = None,
        without:Optional[str | Iterable[str]] = None
    ):
        self._run_main("fit", num_epochs, dl_train, dl_test, test_first, test_on_interrupt, extra, without)

    def log(self, metric:str, value):
        self._run_main("log", metric, value)

    # ------------------------------ normal methods ------------------------------ #

    def summary(self, size: Sequence | torch.Tensor):
        from ..torch_tools import summary
        self.model = self.model.to(self.device)
        print(f"Summary of {self.name}:")
        summary(self.model, size, self.device)

    def state_dict(self, warn = False, filt:Optional[Callable]=None, copy=False):
        state_dict = {}
        for attr in dir(self):
            if not attr.startswith("_"):
                if filt is None or filt(attr):
                    if isinstance(getattr(self, attr), (int,float,str,bool,torch.Tensor,np.ndarray)) or attr in ("train_preds_log", "test_preds_log"):
                        state_dict[attr] = try_copy(getattr(self, attr)) if copy else getattr(self, attr)
                    elif hasattr(getattr(self, attr), "state_dict"):
                        state = getattr(self, attr).state_dict()
                        if copy: state = copy_state_dict(state)
                        state_dict[f"STATE DICT {attr}"] = state
                    elif warn: print(f"Невозможно сохранить {attr}")
        return state_dict


    def save_state_dict(self, path:str, warn = False, filt:Optional[Callable]=None):
        torch.save(self.state_dict(warn, filt=filt), path)

    def load_state_dict(
        self,
        state_dict: dict,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        warn=True,
        **kwargs,
    ):
        if model is not None: kwargs["model"] = model
        if optimizer is not None: kwargs["optimizer"] = optimizer
        if scheduler is not None: kwargs["scheduler"] = scheduler
        if "logger" not in kwargs and self.logger is None: kwargs["logger"] = Logger()

        for key, value in state_dict.items():
            # ключ - state_dict, поэтому он загружается методом "load_state_dict" объекта, который должен быть передан в аргументе
            if key.startswith("STATE DICT "):
                key = key.replace("STATE DICT ", "")
                # если есть аргумент с объектом, используем его
                if key in kwargs:
                    kwargs[key].load_state_dict(value)
                    setattr(self, key, kwargs[key])
                # иначе если данный аттрибут присутствует в self и имеет метод load_state_dict, используем этот метод
                elif hasattr(self, key) and hasattr(getattr(self, key), "load_state_dict"):
                    getattr(self, key).load_state_dict(value) # мутирует объект и возвращает None, поэтому не присваивается
                # в противном случае данный ключ невозможно загрузить
                elif warn: print(f"Невозможно загрузить {key}. Создайте объект необходимого типа и подайте в качестве аргумента к этому методу, например `optimizer = optim.AdamW(model.parameters(), 1e-3)`.")
            else: setattr(self, key, value)


    def load_state_dict_file(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        warn=True,
        **kwargs,
    ):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint, model, optimizer, scheduler, warn, **kwargs)

    @classmethod
    def from_state_dict_file(cls,
        path:str,
        model: torch.nn.Module,
        cbs: Optional[list[Callback]],
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional["Accelerator"] = None,
        device: Optional[torch.device] = None,
        logger: Logger = Logger(),
        default_cbs: Optional[list[Callback]] = None, warn=False, **kwargs):
        learner: Learner = cls(
            model=model,
            name="",
            cbs=cbs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            device=device,
            logger=logger,
            default_cbs=default_cbs,
        )
        learner.load_state_dict_file(path, warn=warn, **kwargs)
        return learner

    def save_checkpoint(self,folder:str = r"checkpoints", cp_name:Optional[str] = None, name:Optional[str] = None, warn=False):
        """Сохраняет чекпоинт в папке checkpoints, имя чекпоинта задается как объединение имени модели и архитектуры, а также текущего количества эпох и количества батчей.
        Если есть метрика тестовой точности, то она добавляется в имя чекпоинта.
        Если есть метрика тренировочной точности, то она добавляется в имя чекпоинта."""
        if name is None: name = self.name

        if cp_name is None:
            cp_name = f"{datetime.now().strftime(r'%d.%m.%Y %H-%M-%S')} ({self.total_epoch}-{self.total_batch}"
            if "test loss" in self.logger:
                cp_name += f"; testloss={self.logger.last('test loss'):.5f}"
            elif "train loss" in self.logger:
                cp_name += f"; trainloss={self.logger.last('train loss'):.5f}"
            if "test accuracy" in self.logger:
                cp_name += f"; testacc={self.logger.last('test accuracy'):.5f}"
            elif "train accuracy" in self.logger:
                cp_name += f"; trainacc={self.logger.last('train accuracy'):.5f}"

            cp_name += ")"

        if not os.path.exists(folder): os.mkdir(folder)
        model_path = os.path.join(folder, name)
        if not os.path.exists(model_path): os.mkdir(model_path)
        cp_path = os.path.join(model_path, cp_name)
        if not os.path.exists(cp_path): os.mkdir(cp_path)

        # Словарь модели и архитектура сохраняються в папку чекпоинта
        self.save_state_dict(
            path = os.path.join(cp_path, "checkpoint.pt"),
            warn = warn,
            filt = lambda x: x
            not in (
                "model" if hasattr(self.model, "state_dict") else None,
                "optimizer" if hasattr(self.optimizer, "state_dict") else None,
                "scheduler" if hasattr(self.scheduler, "state_dict") else None,
                "logger",
                "inputs", "targets", "preds"
            ),
        )
        if hasattr(self.model, "state_dict"): torch.save(self.model.state_dict(), os.path.join(cp_path, "model.pt"))
        if hasattr(self.optimizer, "state_dict"): torch.save(self.optimizer.state_dict(), os.path.join(cp_path, "optimizer.pt")) # type:ignore
        if hasattr(self.scheduler, "state_dict"): torch.save(self.scheduler.state_dict(), os.path.join(cp_path, "scheduler.pt")) # type:ignore

        with open(os.path.join(cp_path, "model.txt"), "w", encoding="utf8") as f:
            f.write(str(self.model))
        with open(os.path.join(cp_path, "optimizer.txt"), "w", encoding="utf8") as f:
            f.write(str(self.optimizer))

        with open(os.path.join(cp_path, "info.yaml"), "w", encoding="utf8") as f:
            text = ""
            text += f"name: {self.name}\n"
            text += f"total_epoch: {self.total_epoch}\n"
            text += f"total_batch: {self.total_batch}\n"
            text += f"model: {type_str(self.model)}\n"
            text += f"loss_fn: {self.loss_fn.__name__ if hasattr(self.loss_fn, '__name__') else type_str(self.loss_fn)}\n" # type:ignore
            text += f"optimizer: {type_str(self.optimizer)}\n"
            text += f"scheduler: {type_str(self.scheduler)}\n"
            text += "callbacks:\n    - "
            text += "\n    - ".join([str(i) for i in self.callbacks])
            f.write(text)

        with open(os.path.join(cp_path, "logs.yaml"), "w", encoding="utf8") as f:
            text = ""
            for key in sorted(self.logger.keys()):
                last = self.logger.last(key)
                text += f"{key}:\n"
                text += f"    count: {len(self.logger[key])}\n"
                text += f"    type: {type(last)}\n"
                if isinstance(last, (torch.Tensor, np.ndarray)) and last.ndim > 0:
                    text += f"    last dtype: {last.dtype}\n"
                    text += f"    last ndim: {last.ndim}\n"
                    text += f"    last shape: {last.shape}\n"
                    text += f"    last min: {last.min()}\n"
                    text += f"    last max: {last.max()}\n"
                    text += f"    last mean: {last.mean()}\n"
                    text += f"    last var: {last.var()}\n"
                    text += f"    last std: {last.std()}\n"
                    text += f"    elements: {last.numel() if isinstance(last, torch.Tensor) else last.size}\n"
                elif isinstance(last, (int, float)) or (isinstance(last, (torch.Tensor, np.ndarray)) and last.ndim == 0):
                    values = self.logger.toarray(key)
                    text += f"    last value: {float(last)}\n"
                    text += f"    lowest: {values.min()}\n"
                    text += f"    highest: {values.max()}\n"
                    text += f"    mean: {values.mean()}\n"
                text += "\n"
            f.write(text)


        # логи сохраняются в папку модели
        self.logger.save(os.path.join(model_path, "logger.npz"))

    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        warn=True,
        **kwargs,
    ):
        self.load_state_dict_file(f'{path}/checkpoint.pt', model, optimizer, scheduler, warn, **kwargs)

        if os.path.exists(f'{path}/model.pt'):
            if model is not None: self.model = model
            self.model.load_state_dict(torch.load(f'{path}/model.pt'))

        if os.path.exists(f'{path}/optimizer.pt'):
            if optimizer is not None: self.optimizer = optimizer
            if self.optimizer is not None:
                self.optimizer.load_state_dict(torch.load(f'{path}/optimizer.pt'))
            elif warn: print("Невозможно загрузить оптимизатор, его нужно передать в виде объекта в аргумент `optimizer`.")

        if os.path.exists(f'{path}/optimizer.pt'):
            if scheduler is not None: self.scheduler = scheduler
            if self.scheduler is not None:
                self.scheduler.load_state_dict(torch.load(f'{path}/scheduler.pt'))
            elif warn: print("Невозможно загрузить планировщик, его нужно передать в виде объекта в аргумент `scheduler`.")


        self.logger.load(f"{os.path.join(str(Path(path).parent.absolute()), 'logger.npz')}")
        self.logger.rollback(self.total_batch)


    @classmethod
    def from_checkpoint(cls,
        path:str,
        model: torch.nn.Module,
        cbs: Optional[list[Callback]],
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional["Accelerator"] = None,
        device: Optional[torch.device] = None,
        logger: Logger = Logger(),
        default_cbs: Optional[list[Callback]] = None, warn=False, **kwargs):
        learner: Learner = cls(
            model=model,
            name="",
            cbs=cbs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            device=device,
            logger=logger,
            default_cbs=default_cbs,
        )
        learner.load_checkpoint(path, warn=warn, **kwargs)
        return learner
