"""sss"""
from collections.abc import Iterable, Callable, Sequence
from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING, final, Literal, overload
import os, pathlib, shutil
from datetime import datetime
from contextlib import contextmanager
import torch, torch.utils.data
import numpy as np

from ..design.event_model import EventModel, EventModelWithPerformanceDebugging, Callback, ConditionCallback, BasicCallback, EventCallback, MethodCallback
from ..logger import Logger
from ..torch_tools import CUDA_IF_AVAILABLE, ensure_device, copy_state_dict
from ..python_tools import SupportsIter, try_copy, type_str, get__name__, int_at_beginning, to_valid_fname

from .cbs_default import (
    DefaultForwardCB,
    DefaultGetLossCB,
    DefaultBackwardCB,
    DefaultOptimizerStepCB,
    DefaultZeroGradCB,
    DefaultSchedulerStepCB,
    DefaultTrainCB,
    DefaultEvalCB,
    DefaultOneBatchCB,
    DefaultInferenceCB,
    DefaultOneEpochCB,
    DefaultFitCB,
    DefaultLogCB,
)

if TYPE_CHECKING:
    from accelerate import Accelerator

__all__ = [
    "Learner",
    "LearnerWithPerformanceDebugging",
    ]

DEFAULT_CBS = (
    DefaultForwardCB(),
    DefaultGetLossCB(),
    DefaultBackwardCB(),
    DefaultOptimizerStepCB(),
    DefaultZeroGradCB(),
    DefaultSchedulerStepCB(),
    DefaultTrainCB(),
    DefaultEvalCB(),
    DefaultOneBatchCB(),
    DefaultInferenceCB(),
    DefaultOneEpochCB(),
    DefaultFitCB(),
    DefaultLogCB(),
)

def _serialize_joblib_compress(fname, obj,):
    import joblib
    return joblib.dump(value=obj, filename=fname, compress=3)

def _load_joblib(fname):
    import joblib
    return joblib.load(fname)

def _serialize_torch_save_dill(fname, obj,):
    import dill
    return torch.save(obj=obj, f=fname, pickle_module=dill)

def _load_torch_dill(fname, map_location=None):
    import dill
    return torch.load(f=fname, map_location=map_location, pickle_module=dill)

class Learner(EventModel):
    """Learner"""

    def __init__(
        self,
        model: torch.nn.Module,
        name: str,
        cbs: Optional[Iterable[Callback]] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer | Any] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler | Any] = None,
        note: Any = None,
        accelerator: Optional["Accelerator"] = None,
        device: Optional[torch.device] = CUDA_IF_AVAILABLE,
        logger: Optional[Logger] = None,
        wandb_run = None,
        default_cbs: Optional[Iterable[Callback]] = DEFAULT_CBS,
    ):
        self.model: torch.nn.Module = model
        self.name: str = name
        self.note = note
        self.optimizer: Optional[torch.optim.Optimizer | Any] = optimizer
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler | Any] = scheduler
        self.loss_fn: Optional[Callable] = loss_fn
        self.accelerator: Optional["Accelerator"] = accelerator
        self.device: Optional[torch.device] = device
        self.logger: Logger = logger if logger is not None else Logger()
        self.wandb_run = wandb_run

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

        self.training:bool = True
        """Self.model mode, `True` if model in `train` mode, `False` if in `eval` mode."""

        self.workdirs: dict[str, str] = {}

        # we init after so that all callbacks have access to the model
        super().__init__(cbs, default_cbs)

    def __type_hinting(self): #pylint:disable=W0238
        self.preds: torch.Tensor | Any = None
        self.inputs: torch.Tensor | Any = None
        self.targets: torch.Tensor | Any = None
        self.loss:torch.Tensor | Any = None
        self.loss_returned_by_model: torch.Tensor | Any = None

    def attach_metric(self, metric: Callable, name: str, train = True, test = True): ...

    def forward(self, inputs:torch.Tensor):
        return self.event("forward", inputs=inputs)[0]

    def get_loss(self, preds:torch.Tensor, targets: torch.Tensor):
        return self.event("get_loss", preds=preds, targets=targets)[0]

    def backward(self):
        self.event("backward")

    def zero_grad(self):
        self.event("zero_grad")

    def optimizer_step(self, *args, **kwargs):
        self.event("optimizer_step", *args, **kwargs)

    def scheduler_step(self):
        self.event("scheduler_step")

    def train(self):
        self.event("train")
        self.training = True

    def eval(self):
        self.event("eval")
        self.training = False

    def one_batch(self, inputs: torch.Tensor, targets: torch.Tensor, train=True):
        # move to gpu
        self.inputs = ensure_device(inputs, self.device)
        self.targets = ensure_device(targets, self.device)

        # get preds
        self.event("before_batch")
        self.event("before_any_batch")
        if train: self.event("before_train_batch")
        else: self.event("before_test_batch")
        results = self.event("one_batch", self.inputs, self.targets, train=train)[-1] # type:ignore

        # get preds
        if results is not None:
            if isinstance(results, tuple): self.preds, self.loss = results # pylint:disable=W0632
            elif isinstance(results, dict):
                for k, v in results.items(): setattr(self, k, v) # type:ignore
            else: raise ValueError(f"one_batch must return tuple, dict, or None, but it returned {type(results)}")

        if train: self.event("after_train_batch")
        else: self.event("after_test_batch")
        self.event("after_batch")
        self.event("after_any_batch")
        if self.status == "train": self.total_batch += 1

    def inference(self, inputs, to_cpu = True, mode = 'eval', status = 'inference', grad=False, unsqueeze = False):
        if status is not None: self.status = status
        return self.event("inference", inputs = inputs, to_cpu = to_cpu, mode = mode, grad = grad, unsqueeze = unsqueeze)[-1]

    def one_epoch(self, dl: torch.utils.data.DataLoader | SupportsIter, train=True):
        self.dl = dl
        self.status = "train" if train else "test"

        with self.context("epoch"):

            self.event("before_any_epoch")
            self.event(f"before_{self.status}_epoch")
            self.event("one_epoch", self.dl, train)
            self.event("after_any_epoch")
            self.event(f"after_{self.status}_epoch")

    def fit(
        self,
        num_epochs: int,
        dltrain: Optional[torch.utils.data.DataLoader | SupportsIter] = None,
        dltest: Optional[torch.utils.data.DataLoader | SupportsIter] = None,
        test_first=True,
        test_every: int = 1,
        catch_interrupt=True,
        test_on_interrupt = True,
        after_fit_on_interrupt = True,
        extra:Optional[Callback | Iterable[Callback]] = None,
        without:Optional[str | Iterable[str]] = None,
        start_epoch = 0,
    ):
        """Fit

        Args:
            num_epochs (int): number of epochs,
            dltrain (Optional[torch.utils.data.DataLoader  |  Any], optional): Train dataloader. Defaults to None.
            dltest (Optional[torch.utils.data.DataLoader  |  Any], optional): Test dataloader. Defaults to None.
            test_first (bool, optional): Whether to test before first epoch. Defaults to True.
            test_every (int, optional): Test every x epochs. Defaults to 1.
            catch_interrupt (bool, optional): Whether to catch KeyboardInterrupt. Defaults to True.
            test_on_interrupt (bool, optional): Whether to test on iterrupt, requires `catch_interrupt`. Defaults to True.
            after_fit_on_interrupt (bool, optional): Whether to run `after_fit` even on iterrupt, requires `catch_interrupt`. Defaults to True.
            extra (Optional[Callback  |  Iterable[Callback]], optional): Extra callbacks to use during fitting. Defaults to None.
            without (Optional[str  |  Iterable[str]], optional): Callbacks that won't be used during fitting. Defaults to None.
        """
        # attributes
        self.num_epochs = num_epochs
        self.dltrain = dltrain
        self.dltest = dltest
        self.test_first = test_first
        self.test_every = test_every

        # catching interrupt
        self.catch_interrupt = catch_interrupt
        if catch_interrupt: self.catch_fit_exceptions = KeyboardInterrupt
        else: self.catch_fit_exceptions = ()
        self.test_on_interrupt = test_on_interrupt
        self.atfer_fit_on_iterrupt = after_fit_on_interrupt

        # epochs
        self.epochs_iterator = range(start_epoch, self.num_epochs)
        self.cur_epoch = 0

        # accelerate the model if needed
        if self.accelerator is None: self.model = ensure_device(self.model, self.device) # type:ignore
        if self.accelerator is None: self.model = self.model.to(self.device)

        # run the train loop
        with self.context("fit", extra=extra, without=without):
            try:
                self.event("before_fit")
                self.event(
                    "fit",
                    epochs_iterator = self.epochs_iterator,
                    dltrain = self.dltrain,
                    dltest = self.dltest,
                    test_first = self.test_first,
                    test_every = self.test_every,
                )

            # catch keyboard interrupt
            except self.catch_fit_exceptions:
                # test on iterrupt
                if self.test_on_interrupt and self.status == "train" and self.dltest is not None:
                    print("Keyboard interrupt, testing one last time... Press stop again to cancel.")
                    try: self.one_epoch(self.dltest, train=False)
                    except self.catch_fit_exceptions: print("Keyboard interrupt, stopping testing...")
                else: print("Keyboard interrupt, stopping the training...")
                # after fit on interrupt
                if self.atfer_fit_on_iterrupt: self.event("after_fit")

            # if no exceptions raised, do `after_fit`
            else: self.event("after_fit")

    def log(self, metric:str, value):
        self.event("log", metric, value)

    # ------------------------------ normal methods ------------------------------ #

    def summary(self, size: Sequence | torch.Tensor):
        from ..torch_tools import summary
        self.model = self.model.to(self.device)
        print(f"Summary of {self.name}:")
        summary(self.model, size, self.device)

    def state_dict(self, copy=False,
                   attrs_state_dict = ("model", "optimizer", "scheduler", "logger"),
                   attrs_values = ("cur_batch", "cur_epoch", "total_batch", "total_epoch", "status", "training", "_workdir"),):
        state_dict:dict[str, Any] = {}
        # attributes with state dict
        for attr_name in attrs_state_dict:
            # get attribute
            attr = getattr(self, attr_name, None)
            # if it has state dict
            if hasattr(attr, "state_dict"):
                # add it to learner state dict
                if copy: state_dict[f'SD {attr_name}'] = copy_state_dict(getattr(self, attr_name).state_dict()) # type:ignore
                else: state_dict[f'SD {attr_name}'] = getattr(self, attr_name).state_dict() # type:ignore
        # attributes with no state dict
        for attr_name in attrs_values:
            if copy: state_dict[attr_name] = try_copy(getattr(self, attr_name, None))
            else: state_dict[attr_name] = getattr(self, attr_name, None)

        return state_dict

    @contextmanager
    def freeze(self,
            attrs_state_dict = ("model", "optimizer", "scheduler", "logger"),
            attrs_values = ("cur_batch", "cur_epoch", "total_batch", "total_epoch", "status", "training", "_workdir"),
            ):
        state_dict = copy_state_dict(self.state_dict(copy=True, attrs_state_dict=attrs_state_dict, attrs_values=attrs_values))
        yield
        self.load_state_dict(state_dict)

    def save_state_dict(self, path:str):
        torch.save(self.state_dict(), path)

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
            # load a state dict key
            if key.startswith("SD "):
                key = key.replace("SD ", "")
                # get value from kwargs
                if key in kwargs:
                    kwargs[key].load_state_dict(value)
                    setattr(self, key, kwargs[key])

                # else get value from self
                elif hasattr(self, key) and hasattr(getattr(self, key), "load_state_dict"):
                    getattr(self, key).load_state_dict(value) # this mutates in place

                # otherwise, no object was provided to load the state dict
                elif warn: print(f"Unable to load {key}. Create an object of the same type and pass it to this method, e.g. `optimizer = optim.AdamW(model.parameters(), 1e-3)`.")

            # load a non state dict key
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

    def info(self):
        info = f"""name: {self.name}
current batch: {self.cur_batch}
total batches: {self.total_batch}
current epoch: {self.cur_epoch}
total epochs: {self.total_epoch}
status: {self.status}
model class: {type_str(self.model)}
optimizer class: {type_str(self.optimizer)}
scheduler class: {type_str(self.scheduler)}
loss_fn class: {type_str(self.loss_fn)}
accelerator class: {type_str(self.accelerator)}
device: {self.device}

callbacks:
   - """
        info += '\n   - '.join([type_str(cb) for cb in self.cbs])

        return info

    def checkpoint(self, dir:str,
                        serialize=True,
                        serialize_cbs = True,
                        save_logger=True,
                        attrs_state_dict = ("model", "optimizer", "scheduler"),
                        attrs_values = ("name", "cur_batch", "cur_epoch", "total_batch", "total_epoch", "status", "training", "_workdir"),
                        attrs_serialize = ("model", "optimizer", "scheduler", "loss_fn", "accelerator"),
                        state_dict_serialize_fn = _serialize_torch_save_dill,
                        object_serialize_fn = _serialize_joblib_compress,
                        save_info = True,
                        mkdir = True,
                        ):
        if mkdir and not os.path.exists(dir): os.mkdir(dir)

        # save state dict
        state_dict = self.state_dict(attrs_state_dict=attrs_state_dict, attrs_values=attrs_values)
        learner_state = {}
        for attr, value in state_dict.items():
            # save state dict of an attribute
            if attr.startswith("SD "):
                attr_name = attr.replace("SD ", "")
                filename = os.path.join(dir, f"{attr_name}.pt")
                state_dict_serialize_fn(filename, value)
            else:
                learner_state[attr] = value

        # save singular attrributes
        filename = os.path.join(dir, "learner.pt")
        object_serialize_fn(filename, learner_state)

        # save logger
        if save_logger:
            if hasattr(self.logger, "save"):
                self.logger.save(os.path.join(dir, "logger.npz"))

        # serialize
        if serialize:
            for attr in attrs_serialize:
                if hasattr(self, attr):
                    filename = os.path.join(dir, f"{attr}.pkl")
                    try: object_serialize_fn(filename, getattr(self, attr))
                    except Exception: pass

            # serialize callbacks
            if serialize_cbs:

                # callbacks
                os.mkdir(os.path.join(dir, "cbs"))
                for i, cb in enumerate(self.cbs):
                    name = ''.join([c for c in type_str(cb) if c.isalnum() or c.isalpha() or c.isspace() or c in ('.', '_', '-')])
                    filename = os.path.join(dir, "cbs", f"{i} - {name}.pkl")
                    try: object_serialize_fn(filename, cb)
                    except Exception: pass

                # default callbacks
                os.mkdir(os.path.join(dir, "default_cbs"))
                for i, cb in enumerate(self.default_cbs):
                    name = ''.join([c for c in type_str(cb) if c.isalnum() or c.isalpha() or c.isspace() or c in ('.', '_', '-')])
                    filename = os.path.join(dir, "default_cbs", f"{i} - {name}.pkl")
                    try: object_serialize_fn(filename, cb)
                    except Exception: pass

        if save_info:
            with open(os.path.join(dir, "info.yaml"), "w", encoding='utf8') as f: f.write(self.info())
            with open(os.path.join(dir, "model.txt"), "w", encoding='utf8') as f: f.write(str(self.model))
            with open(os.path.join(dir, "optimizer.txt"), "w", encoding='utf8') as f: f.write(str(self.optimizer))
            if hasattr(self.logger, "info"):
                with open(os.path.join(dir, "logger.yaml"), "w", encoding='utf8') as f: f.write(self.logger.info_str())

    def load_checkpoint(
        self,
        dir,
        cbs: Optional[Iterable[Callback]] = None,
        model = None,
        optimizer = None,
        scheduler = None,
        loss_fn = None,
        accelerator = None,
        device = None,
        logger = None,
        default_cbs: Optional[Iterable[Callback]] = DEFAULT_CBS,
        state_dict_load_fn = _load_torch_dill,
        object_load_fn = _load_joblib,
        warn = True,
        **kwargs
    ):
        # set kwargs
        for attr, attr_name in ((model, "model"),(optimizer, "optimizer"),(scheduler, "scheduler"),
                                (loss_fn, "loss_fn"),(accelerator, "accelerator"),(device, "device"), (logger,"logger")):
            if attr is None: attr = getattr(self, attr_name, None)
            kwargs[attr_name] = attr

        # load main objects
        exceptions = []
        for attr in ("model", "optimizer", "scheduler", "loss_fn", "logger"):
            # first try to load state dict
            failed = False
            filename = os.path.join(dir, f"{attr}.pt")
            if os.path.isfile(filename):
                try: kwargs[attr].load_state_dict(state_dict_load_fn(filename))
                except Exception as e:
                    failed = True
                    exceptions.append(e)

            # if failed, try to load pickled object
            failed = False
            if failed or kwargs[attr] is None:
                filename = os.path.join(dir, f"{attr}.pkl")
                if os.path.isfile(filename):
                    try: kwargs[attr] = object_load_fn(filename)
                    except Exception as e:
                        failed = True
                        exceptions.append(e)
            if failed and warn: print(f"Failed to load {attr}:\n{exceptions}")

        # load logger
        if kwargs['logger'] is None and os.path.isfile(os.path.join(dir, "logger.npz")):
            try: kwargs['logger'] = Logger.from_file(os.path.join(dir, "logger.npz"))
            except Exception as e:
                print(f"Failed to load logger:\n{e}")
                kwargs['logger'] = Logger()


        # load various attributes
        if os.path.isfile(os.path.join(dir, "learner.pt")):
            try:
                learner_state = object_load_fn(os.path.join(dir, "learner.pt"))
                for k,v in learner_state.items(): setattr(self, k, v)
            except Exception as e:
                if warn: print(f"Failed to load learner state:\n{e}")

        # load callbacks
        if cbs is None:
            cbs = []
            if os.path.isdir(os.path.join(dir, "cbs")):
                for cb in os.listdir(os.path.join(dir, "cbs")):
                    filename = os.path.join(dir, "cbs", cb)
                    try: cbs.append(object_load_fn(filename))
                    except Exception as e:
                        if warn: print(f"Failed to load callback {cb}:\n{e}")

        if default_cbs is None:
            default_cbs = []
            if os.path.isdir(os.path.join(dir, "cbs")):
                for cb in os.listdir(os.path.join(dir, "default_cbs")):
                    filename = os.path.join(dir, "default_cbs", cb)
                    try: default_cbs.append(object_load_fn(filename))
                    except Exception as e:
                        if warn: print(f"Failed to load default callback {cb}:\n{e}")

        # set all attributes
        self.clear()
        for k,v in kwargs.items():
            if v is not None: setattr(self, k, v)

        # set callbacks
        self.add(cbs)
        self.add_default(default_cbs)


    @classmethod
    def from_checkpoint(cls,
        dir,
        cbs: Optional[Iterable[Callback]] = None,
        model = None,
        optimizer = None,
        scheduler = None,
        loss_fn = None,
        accelerator = None,
        device = None,
        logger = None,
        default_cbs: Optional[Iterable[Callback]] = DEFAULT_CBS,
        state_dict_load_fn = _load_torch_dill,
        object_load_fn = _load_joblib,
        warn = True,
        **kwargs
    ):
        learner = cls(model, 'TEMP')  # type:ignore
        learner.load_checkpoint(
            dir=dir,
            cbs=cbs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            accelerator=accelerator,
            device=device,
            logger=logger,
            default_cbs=default_cbs,
            state_dict_load_fn=state_dict_load_fn,
            object_load_fn=object_load_fn,
            warn=warn,
            **kwargs,
        )
        return learner


    def draw_hiddenlayer(self, inputs, input_names = None,transforms: str = "default",framework_transforms: str = "default", rankdir: str = "LR"):
        import hiddenlayer as hl # pylint:disable=E0401 # type:ignore
        return hl.build_graph(self.model, inputs, input_names=input_names, transforms=transforms, framework_transforms=framework_transforms, rankdir=rankdir)

    def draw_torchvis(self, inputs, targets=None, loss=lambda x,_: x.mean(), show_attrs=False, show_saved=False, max_attr_chars=10):
        from torchviz import make_dot # pylint:disable=E0401 # type:ignore
        preds = self.model(inputs)
        return make_dot(loss(preds,targets), params=dict(self.model.named_parameters()), show_attrs=show_attrs, show_saved=show_saved, max_attr_chars=max_attr_chars)

    def draw_torchview(self,inputs,**kwargs):
        import torchview # pylint:disable=E0401 # type:ignore
        return torchview.draw_graph(self.model, input_data=inputs, **kwargs).visual_graph

    def draw_torchlens(self,inputs, vis_opt='unrolled', **kwargs):
        import torchlens as tl # pylint:disable=E0401 # type:ignore
        return tl.log_forward_pass(self.model, inputs, vis_opt=vis_opt, **kwargs)

    def get_workdir(self, dir, mkdir=True):
        """Returns path to working directory, and creates it if it doesn't exist.

        Path is `{dir}/{self.name} {run index} - {datetime}`, where `run index` is a number that automatically increments based on what indexes are already saved in the folder.

        If last `run index` folder is empty, it will be deleted and used.
        """
        if dir in self.workdirs: return self.workdirs[dir]

        # create the root directory
        if (not os.path.exists(dir)) and mkdir: os.mkdir(dir)
        # get all runs in the root directory
        runs: dict[int, str] = {int_at_beginning(i.replace(f'{self.name} ', '')):i for i in os.listdir(dir) if (os.path.isdir(os.path.join(dir, i)) and self.name in i and i.replace(f'{self.name} ', '')[0].isnumeric())} # type:ignore
        # if no runs, this is the 1st run
        if len(runs) == 0: run_index = 1
        else:
            # find last run
            last = max(list(runs.keys()))
            # if last run is empty, delete it and use same index
            if len(os.listdir(os.path.join(dir, runs[last]))) == 0:
                shutil.rmtree(os.path.join(dir, runs[last]))
                run_index = last
            # otherwise increment last run index
            else: run_index = last + 1
        # create this learners directory
        now = datetime.now()
        workdir = os.path.join(dir, f"{self.name} {run_index} - {now.year}.{now.month}.{now.day} {now.hour}-{now.minute}")
        if not os.path.exists(workdir): os.mkdir(workdir)
        self.workdirs[dir] = workdir
        return self.workdirs[dir]

    def get_epochbatch_dir(self, dir, mkdir = True, metrics = ('test loss', 'test accuracy')):
        """Return a directory for current epoch/batch - `{self.workdir}/{total epochs} {total batches} {metric details}`.

        For example: `e5 b36385 test-loss_0.22640`

        Workdir is `{dir}/{self.name} {run index} - {datetime}`

        Args:
            dir (_type_): _description_
            mkdir (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        workdir = self.get_workdir(dir, mkdir=mkdir)
        checkpoint_name = f'e{self.total_epoch} b{self.total_batch}'

        for metric in metrics:
            if metric in self.logger:
                checkpoint_name += f" {to_valid_fname(metric.replace(' ', '-'))}_{self.logger.last(metric):.5f}"

        checkpoint_path = os.path.join(workdir, checkpoint_name)
        if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)
        return os.path.join(checkpoint_path)

    def get_prefix_epochbatch_dir(self, dir, prefix, mkdir = True, metrics = ('test loss', 'test accuracy')):
        """Return a directory for current epoch/batch - `{self.workdir}/{prefix}/{total epochs} {total batches} {metric details}`.

        For example: `e5 b36385 test-loss_0.22640`

        Workdir is `{dir}/{self.name} {run index} - {datetime}`

        Args:
            dir (_type_): _description_
            mkdir (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        workdir = os.path.join(self.get_workdir(dir, mkdir=mkdir), prefix)
        if not os.path.exists(workdir): os.mkdir(workdir)
        checkpoint_name = f'e{self.total_epoch} b{self.total_batch}'

        for metric in metrics:
            if metric in self.logger:
                checkpoint_name += f" {to_valid_fname(metric.replace(' ', '-'))}_{self.logger.last(metric):.5f}"

        checkpoint_path = os.path.join(workdir, checkpoint_name)
        if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)
        return os.path.join(checkpoint_path)

    def set_lr(self, lr:float | Callable):
        from ..torch_tools import set_lr_
        set_lr_(self.optimizer, lr)  # type:ignore

    def wandb_init(
        self,
        project: str,
        name: Optional[str],
        config: dict,
        notes: Optional[str] = None,
        tags: Sequence | None = None,
        magic: bool = True,
        save_code: bool = True,
        watch: bool = True,
        watch_step: int = 32,
    ):
        """Initialize a wandb run.

        Args:
            project (str): _description_
            name (Optional[str]): _description_
            config (dict): _description_
            notes (Optional[str], optional): _description_. Defaults to None.
            tags (Sequence | None, optional): _description_. Defaults to None.
            magic (bool, optional): _description_. Defaults to True.
            save_code (bool, optional): _description_. Defaults to True.
            watch (bool, optional): _description_. Defaults to True.
            watch_step (int, optional): _description_. Defaults to 32.
        """
        import wandb
        self.project = project
        self.config = config
        self.last_wandb_logged_batch = 0

        self.wandb_run = wandb.init(
            project = self.project,
            name = name,
            config = self.config,
            notes = notes,
            tags = tags,
            magic = magic,
            save_code = save_code,
        )

        if watch: self.wandb_watch(watch_step=watch_step)

    def wandb_watch(self, source:Literal["gradients", "parameters", "all"] | None = "all",  watch_step = 32):
        """Watch model with wandb.watch.

        Args:
            source: _description_. Defaults to "all".
            watch_step (int, optional): _description_. Defaults to 32.
        """
        self.wandb_run.watch(self.model, log = source, log_freq=watch_step) # type:ignore

    def wandb_log(self, metric, value, commit=False):
        """Log a metric into wandb run.

        Args:
            metric (_type_): _description_
            value (_type_): _description_
            commit (bool, optional): _description_. Defaults to False.
        """
        self.wandb_run.log({metric: value}, step = self.total_batch, commit=commit) # type:ignore

    def wandb_update(self, commit=False):
        """Push all new metric records since last update to wandb. Does not commit them by default.

        Args:
            commit (bool, optional): _description_. Defaults to False.
        """
        for batch in range(self.last_wandb_logged_batch, self.total_batch+1):
            train_metrics = {}
            test_metrics = {}
            other_metrics = {}
            for key, value in self.logger.logs.items():
                if batch in value:
                    if 'train' in key: train_metrics[key] = value[batch]
                    elif 'test' in key: test_metrics[key] = value[batch]
                    else: other_metrics[key] = value[batch]
                metrics = {"train": train_metrics, "test": test_metrics, **other_metrics}
                self.wandb_run.log(metrics, step=batch, commit=commit) # type:ignore

            self.last_wandb_logged_batch += 1

    def wandb_finalize(self):
        self.wandb_run.finish() # type:ignore

class LearnerWithPerformanceDebugging(Learner, EventModelWithPerformanceDebugging): pass