from typing import Any, Optional
from collections.abc import Callable
from abc import ABC, abstractmethod
from contextlib import nullcontext
import random, time
import os
from datetime import datetime

import yaml
import torch
import optuna

from ..progress_bar import PBar
from ..loaders.image import imreadtensor
from ..torch_tools import copy_state_dict
from ..python_tools import to_valid_fname
from ..transforms.intensity import znorm
from ..logger import Logger, Comparison

class _SafeIndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

_valid_chars = frozenset(" -_.()=")
def _dict_to_valid_fname_str(d:dict):
    return to_valid_fname(' '.join([f'{k}={v}' for k, v in d.items()]), valid_chars=_valid_chars)

# region Benchmark
class Benchmark(ABC):
    model:torch.nn.Module
    def __init__(self):
        super().__init__()
        self.logger = Logger()
        self.num_evals = 0
        self._workdir = None

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def log(self, metric, value):
        self.logger.add(metric, value, cur_batch=self.num_evals)

    def one_step(self):
        """Updates `self.model` and return loss, used in `run`."""
        raise NotImplementedError

    # region .run
    def run(self, optimizer, max_steps = None, max_evals = None, print_loss = False, do_backward = True, trial=None, verbose=False):
        """_summary_

        Args:
            optimizer (_type_): _description_
            max_steps (_type_, optional): _description_. Defaults to None.
            max_evals (_type_, optional): _description_. Defaults to None.
            print_loss (bool, optional): _description_. Defaults to False.
            do_backward (bool, optional): _description_. Defaults to True.
            trial (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            optuna.TrialPruned: _description_

        Returns:
            _type_: _description_
        """
        if max_steps is None and max_evals is None: raise ValueError('Either max_steps or max_evals must be provided.')

        cur_iter = 0
        while True:

            # construct closure
            def closure(backward = True):
                optimizer.zero_grad()
                loss = self.one_step()
                if backward and do_backward: loss.backward()
                return loss

            # evaluate closure
            loss = optimizer.step(closure)

            if isinstance(loss, torch.Tensor): loss = loss.detach().cpu()

            self.log('loss', loss)
            if verbose: print(cur_iter, self.num_evals, loss)
            if trial is not None:
                trial.report(loss, cur_iter)
                if trial.should_prune(): raise optuna.TrialPruned()

            if max_evals is not None and self.num_evals >= max_evals: break
            if max_steps is not None and cur_iter >= max_steps: break
            cur_iter += 1

        if print_loss: print(f'reached {self.logger.min('loss')}')
        return loss  # type:ignore
        #endregion

    def workdir(self, name: Optional[str] = None, dir:str = 'runs', mkdir = True, suffix = 'hparams', hparams=None, check_exists = True):
        """Creates and returns `dir/{name} {suffix}`, where `suffix` is hyperparams.

        Args:
            name (Optional[str], optional): _description_. Defaults to None.
            dir (str, optional): _description_. Defaults to 'runs'.
            mkdir (bool, optional): _description_. Defaults to True.
            suffix (str, optional): _description_. Defaults to 'hparams'.
            hparams (_type_, optional): _description_. Defaults to None.
            check_exists (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
            ValueError: _description_
            FileExistsError: _description_

        Returns:
            _type_: _description_
        """
        # make parent directory
        if mkdir and not os.path.exists(dir): os.mkdir(dir)

        # if hasn't been created
        if self._workdir is None:

            # then it needs a name
            if name is None: raise ValueError()

            # create a suffix
            if suffix == 'datetime': suffix_str = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
            elif suffix == 'hparams':
                if isinstance(hparams, dict): suffix_str = _dict_to_valid_fname_str(hparams)
                else: suffix_str = to_valid_fname(str(hparams))
            else: raise ValueError(f'Invalid suffix: {suffix}')

            # assign _workdir
            self._workdir = os.path.join(dir, f'{name} {suffix_str}')

            # create _workdir or raise if it exists
            if not os.path.exists(self._workdir): os.mkdir(self._workdir)
            elif check_exists: raise FileExistsError(f'Workdir already exists: {self._workdir}')

        return self._workdir

    def save_vis(self, dir):
        pass

    def save(self, opt=None, hparams = None, name: Optional[str] = None, dir = 'runs', ):
        hyperparameters = {}

        # save optimizer name
        if opt is not None:
            if isinstance(opt, str): hyperparameters['optimizer'] = opt
            elif isinstance(opt, type): hyperparameters['optimizer'] = opt.__name__
            else: hyperparameters['optimizer'] = opt.__class__.__name__

        # save optimizer hyperparameters
        if hparams is not None:
            if isinstance(hparams, dict): hyperparameters.update(hparams)
            else: hyperparameters['hparams'] = hparams

        # create folder name from optimizer name
        if name is None:
            if opt is None: raise ValueError("Either opt or name must be provided.")
            name = to_valid_fname(hyperparameters['optimizer'])

        # save logger
        self.logger.save(os.path.join(self.workdir(name = name, dir = dir, hparams=hparams), 'logger.npz'))

        # save hyperparams to a yaml
        with open(os.path.join(self.workdir(name = name, dir = dir), 'hyperparameters.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(hyperparameters, f, sort_keys=False, Dumper = _SafeIndentDumper)

        # save a visualization (if this method does something)
        self.save_vis(self.workdir(name = name, dir = dir))

    # region .objective
    @classmethod
    def objective(cls, bench_kwargs, max_steps = None, max_evals = None, opt_cls:Any = ..., fixed_kwargs = {}, search_kwargs = {}, do_backward = True, save = True): # pylint:disable=W0102
        """
        Create an optuna objective.

        Args:
            bench_kwargs (_type_): Same kwargs as passed to this class __init__.
            max_evals (_type_): Maximum evals per each trial.
            opt_cls (_type_): Optimizer class, e.g. `torch.optim.SGD`.
            fixed_kwargs (dict, optional): Dictionary of fixed kwargs. Defaults to {}.
            search_kwargs (dict, optional): Dictionary of kwargs to optimize and their domains. Defaults to {}.
            save (bool, optional): Whether to save a checkpoint on each trial. Defaults to True.

        Returns:
            Callable: An optuna objective.

        Example:
        ```py

        search_kwargs = dict(lr = (0., 10000.), momentum = (0., 1.))
        study = optuna.create_study()
        objective = SomeBenchmark.objective(dict(...), max_evals = 10, opt_cls = torch.optim.SGD, search_kwargs = search_kwargs, save=True)
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        ```
        """
        if len(search_kwargs) == 0: raise ValueError("search_kwargs must not be empty")

        # create a benchmark to save state_dict to make sure we always start from the same initial point
        bench = cls(**bench_kwargs)
        state_dict = copy_state_dict(bench.state_dict())
        del bench

        # optuna objective
        def objective(trial:optuna.trial.Trial):

            # construct hyperparams
            hparams = {}

            # add fixed params
            hparams.update(fixed_kwargs)

            # generate search params
            for k,v in search_kwargs.items():

                # tuple - range
                if isinstance(v, tuple):
                    if len(v) != 2: raise ValueError(f"Tuple size must be 2, got {len(v)}")
                    if isinstance(v[0], int): hparams[k] = trial.suggest_int(k, v[0], v[1])
                    elif isinstance(v[0], float): hparams[k] = trial.suggest_float(k, v[0], v[1])
                    else: raise ValueError(f"Unsupported type tuple[{type(v[0])}]")

                # list - selection
                elif isinstance(v, list): hparams[k] = trial.suggest_categorical(k, v)

                # bool - True / False
                elif isinstance(v, bool): hparams[k] = trial.suggest_categorical(k, [True, False])
                else: raise ValueError(f"Unsupported type {type(v)}")

            # create benchmark of self class
            bench = cls(**bench_kwargs)
            bench.load_state_dict(copy_state_dict(state_dict))

            # create an optimizer
            opt = opt_cls(bench.parameters(), **hparams)

            # run the benchmark for `max_evals`
            loss = bench.run(opt, max_steps=max_steps, max_evals=max_evals, do_backward=do_backward, trial=trial)

            # save results
            if save: bench.save(opt = opt, hparams = hparams)
            if isinstance(loss, torch.Tensor): loss = loss.detach().cpu()

            # return loss
            return float(loss)

        # return optuna objective
        return objective
        # endregion



class InputModel(torch.nn.Module):
    def __init__(self, input:torch.Tensor):
        super().__init__()
        self.input = torch.nn.Parameter(input, requires_grad=True)

    def forward(self): return self.input