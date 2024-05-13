"""2s"""
from typing import Any, Sequence, Mapping, Iterable
from functools import partial
import numpy as np
import scipy.optimize
import torch
from ..python_tools import reduce_dim
class Param:
    def vec_to_param(self, vec:np.ndarray) -> Any: ...

# scalar
class Param_Scalar(Param):
    def __init__(self, param: int|float):
        self.len = 1
        self.vec = np.array([param])
    def vec_to_param(self, vec:np.ndarray) -> float: return float(vec[0])

# array
class Param_Numpy(Param):
    def __init__(self, param: np.ndarray):
        self.shape = param.shape
        self.vec = param.ravel()
        self.len = self.vec.size
    def vec_to_param(self, vec: np.ndarray) -> np.ndarray: return vec.reshape(self.shape)

class Param_Tensor(Param):
    def __init__(self, param: torch.Tensor):
        self.len = param.numel()
        self.shape = param.shape
        self.requires_grad = param.requires_grad
        self.vec = param.detach().ravel().cpu().numpy()
    def vec_to_param(self, vec:np.ndarray) -> torch.Tensor: return torch.as_tensor(vec).reshape(self.shape).requires_grad_(self.requires_grad)

# params
class Param_ListParam(Param):
    def __init__(self, param: list | Sequence | Iterable):
        self.params = [to_param(i) for i in param]
        self.lens = [p.len for p in self.params]
        self.splits = [0] + np.cumsum(self.lens).tolist()[:-1]
        self.vec = np.array(reduce_dim([i.vec for i in self.params]))
        self.len = self.vec.size
    def vec_to_param(self, vec:np.ndarray) -> list[Any]: return [i.vec_to_param(vec[s:s+i.len]) for s, i in zip(self.splits, self.params)]

class Param_TupleParam(Param):
    def __init__(self, param: list | Sequence):
        self.params = [to_param(i) for i in param]
        self.lens = [p.len for p in self.params]
        self.splits = [0] + np.cumsum(self.lens).tolist()[:-1]
        self.vec = np.array(reduce_dim([i.vec for i in self.params]))
        self.len = self.vec.size
    def vec_to_param(self, vec:np.ndarray) -> tuple[Any]: return tuple([i.vec_to_param(vec[s:s+i.len]) for s, i in zip(self.splits, self.params)])

class Param_DictParam(Param):
    def __init__(self, param: dict | Mapping):
        self.param_keys = list(param.keys())
        self.param_vals = [to_param(i) for i in param.values()]
        self.lens = [p.len for p in self.param_vals]
        self.splits = [0] + np.cumsum(self.lens).tolist()[:-1]
        self.vec = np.array(reduce_dim([i.vec for i in self.param_vals]))
        self.len = self.vec.size
    def vec_to_param(self, vec:np.ndarray) -> dict[str, Any]:
        return {k:i.vec_to_param(vec[s:s+i.len]) for k, s, i in zip(self.param_keys, self.splits, self.param_vals)}

class Param_ArgsKwargs(Param):
    def __init__(self, args: tuple, kwargs: dict):
        self.args_params = Param_ListParam(args)
        self.kwargs_params = Param_DictParam(kwargs)
        self.len = self.args_params.len + self.kwargs_params.len
        self.vec = np.zeros(self.len)
        self.vec[:self.args_params.len] = self.args_params.vec
        self.vec[self.args_params.len:] = self.kwargs_params.vec
    def vec_to_param(self, vec:np.ndarray) -> tuple[list[Any], dict[str, Any]]: return self.args_params.vec_to_param(vec), self.kwargs_params.vec_to_param(vec)

class Param_TorchModule(Param):
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.params = Param_DictParam(module.state_dict())
        self.len = self.params.len
        self.vec = self.params.vec
    def vec_to_param(self, vec:np.ndarray) -> torch.nn.Module:
        self.module.load_state_dict(self.params.vec_to_param(vec))
        return self.module

def to_param(param:Any):
    if isinstance(param, torch.nn.Module): return Param_TorchModule(param)

    if isinstance(param, (int, float)):return Param_Scalar(param)

    if isinstance(param, np.ndarray): return Param_Numpy(param)

    if isinstance(param, torch.Tensor): return Param_Tensor(param)

    if isinstance(param, list): return Param_ListParam(param)
    if isinstance(param, tuple): return Param_TupleParam(param)
    if isinstance(param, dict): return Param_DictParam(param)

    if isinstance(param, (Sequence, Iterable)): return Param_ListParam(param)
    if isinstance(param, Mapping): return Param_DictParam(param)


    raise ValueError(f"Unknown param type {type(param)}")

def to_trainable_fn(fn, trainable_args = (), trainable_kwargs = {}, fixed_args = (), fixed_kwargs = {}): #pylint:disable=W0102,W1113
    params = Param_ArgsKwargs(trainable_args, trainable_kwargs)
    def trainable_fn(vec):
        trainable_args, trainable_kwargs = params.vec_to_param(vec)
        return fn(*trainable_args, *fixed_args, **trainable_kwargs, **fixed_kwargs)
    return trainable_fn

def _to_vector_fn_from_x0_params(fn, params:Param, fixed_args, fixed_kwargs1, fixed_kwargs2, eval_callback):
    if eval_callback is None:
        def trainable_fn(vec) -> float:
            x0 = params.vec_to_param(vec)
            return fn(x0, *fixed_args, **fixed_kwargs1, **fixed_kwargs2)
        return trainable_fn
    else:
        def trainable_fn(vec) -> float:
            x0 = params.vec_to_param(vec)
            scalar = fn(x0, *fixed_args, **fixed_kwargs1, **fixed_kwargs2)
            value = eval_callback(x0, vec, scalar)
            if isinstance(value, (int,float)): scalar = value
            return scalar
        return trainable_fn

def _to_vector_fn_from_kw_params(fn, params:Param_ArgsKwargs, fixed_args, fixed_kwargs1, fixed_kwargs2, eval_callback):
    if eval_callback is None:
        def trainable_fn(vec) -> float:
            trainable_args, trainable_kwargs = params.vec_to_param(vec)
            return fn(*trainable_args,*fixed_args,  **trainable_kwargs, **fixed_kwargs1, **fixed_kwargs2)
        return trainable_fn
    else:
        def trainable_fn(vec) -> float:
            trainable_args, trainable_kwargs = params.vec_to_param(vec)
            scalar = fn(*trainable_args,*fixed_args,  **trainable_kwargs, **fixed_kwargs1, **fixed_kwargs2)
            value = eval_callback([trainable_args, trainable_kwargs], vec, scalar)
            if isinstance(value, (int,float)): scalar = value
            return scalar
        return trainable_fn

class OptimizeResultWithParams:
    def __init__(self, res:scipy.optimize.OptimizeResult, params:Any):
        self.res: scipy.optimize.OptimizeResult = res
        self.params = params
        self.x = self.res.x
        self.success = self.res.success
        self.message = self.res.message
        self.status = self.res.status
    def __getattr__(self, name): return self.res.__getattr__(name)
    def __str__(self): return self.res.__str__()
    def __repr__(self): return self.res.__repr__()

class ScipyMinimizer:
    def __init__(self, x0, fixed_args = (), fixed_kwargs = {}): #pylint:disable=W0102,W1113
        self._params = to_param(x0)
        self.x0 =self._params.vec
        self.fixed_args = fixed_args
        self.fixed_kwargs = fixed_kwargs

    def minimize(self, fun, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, eval_callback = None, update_x0 = True, kwargs = {}) -> OptimizeResultWithParams: #pylint:disable=W0102
        vector_fun = _to_vector_fn_from_x0_params(fun, self._params, self.fixed_args, self.fixed_kwargs, kwargs, eval_callback)
        self.res = scipy.optimize.minimize(vector_fun, self.x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)
        if update_x0: self.x0 = self.res.x
        self.params: Any = self._params.vec_to_param(self.res.x)
        self.res_with_params = OptimizeResultWithParams(self.res, self.params)
        return self.res_with_params

    def get_params(self) -> Any:
        return self._params.vec_to_param(self.res.x)


class ScipyMinimizerArgs:
    def __init__(self, *trainable_args, fixed_args = (), fixed_kwargs = {}, **trainable_kwargs): #pylint:disable=W0102,W1113
        self._params = Param_ArgsKwargs(trainable_args, trainable_kwargs)
        self.x0 =self._params.vec
        self.fixed_args = fixed_args
        self.fixed_kwargs = fixed_kwargs

    def minimize(self, fun, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, eval_callback = None, update_x0 = True, kwargs = {}) -> OptimizeResultWithParams: #pylint:disable=W0102
        vector_fun = _to_vector_fn_from_kw_params(fun, self._params, self.fixed_args, self.fixed_kwargs, kwargs, eval_callback)
        self.res = scipy.optimize.minimize(vector_fun, self.x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)
        if update_x0: self.x0 = self.res.x
        self.params: tuple[list, dict[str, Any]] = self._params.vec_to_param(self.res.x)
        self.res_with_params = OptimizeResultWithParams(self.res, self.params)
        return self.res_with_params

    def get_params(self) -> Any:
        return self._params.vec_to_param(self.res.x)

def scipy_minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, eval_callback = None, kwargs = {}): #pylint:disable=W0102,W1113
    return ScipyMinimizer(x0).minimize(fun, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options, eval_callback = eval_callback, kwargs=kwargs)



class RandomOptimizer:
    def __init__(self, x0:Any, init_lr = 0.05, lookback = 100, lradd = 0, lrmul=1, lrpow=0.8, rand_fn = partial(np.random.uniform, low=-1, high=1), callback = None):
        self.rand_fn = rand_fn
        self.x = x0
        self._params = to_param(x0)
        self.xvec = self._params.vec
        self.new_solution = self.xvec
        self.len = self._params.len
        self.lr = init_lr
        self.lookback = lookback
        self.lradd = lradd
        self.lrmul = lrmul
        self.lrpow = lrpow
        self.callback = callback
        self.losses = []
        self.last_loss = float("inf")
        self.cur_iter = 0

    def step(self, loss):
        if loss < self.last_loss:
            self.xvec = self.new_solution
            self.last_loss = loss
            self.cur_iter += 1
            if self.cur_iter > self.lookback: self.lr = abs((abs(self.losses[-self.lookback] - self.losses[-1]) ** self.lrpow) * self.lrmul + self.lradd)
        self.losses.append(loss)
        self.new_solution = self.xvec + self.rand_fn(size=self.len) * self.lr
        self.x = self._params.vec_to_param(self.new_solution)
        if self.callback is not None: self.callback(self.x, self.xvec)
        return self.x

