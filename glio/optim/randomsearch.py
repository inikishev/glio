from collections.abc import Iterable, Callable
from functools import partial
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from .optimizer import ClosureOptimizer, clone_params, set_params_

class RandomSearch(ClosureOptimizer):
    """
    Random search optimizer.
    On each iteration:

    1. Generates random `params`.
    2. Evaluate `loss = closure(params)`
    3. If loss became smaller, set `best_params` to `params`.

    Make sure to call `set_best()` method after optimizing to apply the best found parameters.
    """
    def __init__(self, params:Iterable[torch.Tensor | torch.nn.parameter.Parameter], rand:Callable = torch.randn_like, set_best_each_time=True):
        self.params = torch.nn.ParameterList(params)
        super().__init__(self.params, {})
        self.rand = rand
        self.set_best_each_time = set_best_each_time

        self.lowest_loss = float('inf')
        self.best_params = clone_params(params)

    def step(self, closure: Callable):#type:ignore
        # 1. Generates random `params`.
        set_params_(self.params, [self.rand(p, device=p.device) for p in self.params])
        # 2. Evaluate `loss = closure(params)`
        loss = closure()
        # 3. If loss became smaller
        if loss < self.lowest_loss:
            # set `best_params` to `params`.
            self.lowest_loss = loss
            self.best_params = clone_params(self.params)

        if self.set_best_each_time: set_params_(self.params, self.best_params)
        return self.lowest_loss

    def set_best(self): set_params_(self.params, self.best_params)


class RandomWalk(ClosureOptimizer):
    """
    Random walk optimizer.
    On each iteration:

    1. Apply random petrubation to best params: `params = best_params + rand * lr`.
    2. Evaluate `loss = f(params)`
    3. If loss became smaller, set `best_params` to `params`.

    Make sure to call `set_best()` method after optimizing to apply the best found parameters.
    """
    def __init__(self, params:Iterable[torch.Tensor | torch.nn.parameter.Parameter], lr = 1e-3, rand:Callable = torch.randn_like):
        self.params = torch.nn.ParameterList(params)
        super().__init__(self.params, {})
        self.lr = lr
        self.rand = rand

        self.lowest_loss = float('inf')
        self.best_params = clone_params(params)

    def step(self, closure: Callable):
        # 1. Apply random petrubation to best params: `params = best_params + rand * lr`.
        set_params_(self.params, [p + self.rand(p, device=p.device) * self.lr for p in self.best_params])
        # 2. Evaluate `loss = closure(params)`
        loss = closure()
        # 3. If loss became smaller
        if loss < self.lowest_loss:
            # set `best_params` to `params`.
            self.lowest_loss = loss
            self.best_params = clone_params(self.params)

        return loss

    def set_best(self): set_params_(self.params, self.best_params)


class RandomWalk2(ClosureOptimizer):
    """
    Second order random walk optimizer.
    On each iteration:

    1. Apply random petrubation to best delta: `delta = best_delta + rand * lr`
    2. Add delta to params to `params = best_params + delta`.
    3. Evaluate `loss = closure(params)`
    4. If loss became smaller, set `best_params` to `params`, set `best_delta` to `delta`, else generate new `delta`.

    Make sure to call `set_best()` method after optimizing to apply the best found parameters.
    """
    def __init__(self, params:Iterable[torch.Tensor | torch.nn.parameter.Parameter], lr = 1e-3, rand = torch.randn_like, init=torch.zeros_like):
        self.params = torch.nn.ParameterList(params)
        super().__init__(self.params, {})
        self.lr = lr
        self.rand = rand
        self.init = init

        self.lowest_loss = float('inf')
        self.best_params = clone_params(params)

        self.delta = torch.nn.ParameterList([self.init(p, device=p.device) for p in self.params])
        self.best_delta = clone_params(self.delta)

    def step(self, closure: Callable):
        # 1. Apply random petrubation to best delta: `delta = best_delta + rand * lr`
        self.delta = [m + self.rand(m, device = m.device) * self.lr for m in self.best_delta]
        # 2. Add delta to params to `params = best_params + delta`.
        set_params_(self.params, [p + m for p, m in zip(self.best_params, self.delta)])
        # 2. Evaluate `loss = closure(params)`
        loss = closure()
        # 3. If loss became smaller,
        if loss < self.lowest_loss:
            # set `best_params` to `params`,
            self.lowest_loss = loss
            self.best_params = clone_params(self.params)
            # set `best_delta` to `delta`
            self.best_delta = clone_params(self.delta)
        else:
            set_params_(self.delta, [self.init(p, device=p.device) for p in self.params])
        return loss

    def set_best(self): set_params_(self.params, self.best_params)


class RandomConvexWalk(ClosureOptimizer):
    """
    Second order random walk optimizer that tries to walk in the direction where loss descrease is accelerating (but usually fails)
    On each iteration:

    1. Apply random petrubation to best delta: `delta = best_delta + rand * lr`
    2. Add delta to params to `params = best_params + delta`.
    3. Evaluate `loss = closure(params)`
    4. If loss became smaller, set `best_params` to `params`, else reset `highest_delta_loss` to -inf and generate new `delta`.
    5. `loss_delta = previous_loss - loss`
    6. If delta loss became bigger, set `best_delta` to `delta`

    Make sure to call `set_best()` method after optimizing to apply the best found parameters.
    """
    def __init__(self, params:Iterable[torch.Tensor | torch.nn.parameter.Parameter], lr = 1e-3, rand = torch.randn_like, init=torch.zeros_like):
        self.params = torch.nn.ParameterList(params)
        super().__init__(self.params, {})
        self.lr = lr
        self.rand = rand
        self.init = init

        self.lowest_loss = float('inf')
        self.previous_loss = float('inf')
        self.highest_delta_loss = -float('inf')
        self.best_params = clone_params(params)

        self.delta = torch.nn.ParameterList([self.init(p, device=p.device) for p in self.params])
        self.best_delta = clone_params(self.delta)

    def step(self, closure: Callable):
        # 1. Apply random petrubation to best delta: `delta = best_delta + rand * lr`
        set_params_(self.delta, [m + self.rand(m, device = m.device) * self.lr for m in self.best_delta])
        # 2. Add delta to params to `params = best_params + delta`.
        set_params_(self.params, [p + m for p, m in zip(self.best_params, self.delta)])
        # 3. Evaluate `loss = closure(params)`
        loss = closure()
        # 4. If loss became smaller,
        if loss < self.lowest_loss:
            # set `best_params` to `params`,
            self.lowest_loss = loss
            self.best_params = clone_params(self.params)
            # 5. `loss_delta = previous_loss - loss`
            loss_delta = self.previous_loss - loss
            # 6. If delta loss became bigger, set `best_delta` to `delta`
            if loss_delta > self.highest_delta_loss:
                self.highest_delta_loss = loss_delta
                self.best_delta = clone_params(self.delta)

        # else reset `highest_delta_loss` to -inf generate new random delta
        else:
            self.highest_delta_loss = -float('inf')
            set_params_(self.delta, [self.init(p, device=p.device) for p in self.params])
        self.previous_loss = loss

        return loss

    def set_best(self): set_params_(self.params, self.best_params)


class RandomWalkMomentum(ClosureOptimizer):
    """
    Random walk with momentum. Same as random walk but keeps decaying average of last updates and uses to update the model.
    On each iteration:

    1. Generate random petrubation `petrubation = rand * lr`
    2. Apply petrubation to best params: `params = best_params + petrubation`.
    3. Evaluate `loss = f(params)`
    4. If loss became smaller, add petrubation to momentum: `momentum = momentum + petrubation`.
    5. Apply momentum to best params: `best_params = best_params + momentum`.
    6. Decay momentum: `momentum = momentum * decay`

    Make sure to call `set_best()` method after optimizing to apply the best found parameters.
    """
    def __init__(self, params:Iterable[torch.Tensor | torch.nn.parameter.Parameter], lr = 1e-3, decay = 0.5, rand = torch.randn_like):
        self.params = torch.nn.ParameterList(params)
        super().__init__(self.params, {})
        self.lr = lr
        self.decay = decay
        self.rand = rand

        self.lowest_loss = None
        self.best_params = clone_params(params)

        self.momentum = None
        self.petrubation = torch.nn.ParameterList([torch.zeros_like(p, device=p.device) for p in self.params])

    def step(self, closure: Callable):
        if self.momentum is None: self.momentum = torch.nn.ParameterList([torch.zeros_like(p, device=p.device) for p in self.params])
    
        # 1. Generate random petrubation `petrubation = rand * lr`
        set_params_(self.petrubation, [self.rand(p, device=p.device) * self.lr for p in self.params])
        # 2. Apply petrubation to best params: `params = best_params + petrubation`.
        set_params_(self.params, [p + pet for p, pet in zip(self.best_params, self.petrubation)])
        # 3. Evaluate `loss = closure(params)`
        loss = closure()
        if self.lowest_loss is None: self.lowest_loss = loss
        # 4. If loss became smaller
        if loss < self.lowest_loss:
            # add petrubation to momentum: `momentum = momentum + petrubation`.
            self.lowest_loss = loss
            set_params_(self.momentum, [m+p for m,p in zip(self.momentum, self.petrubation)])
        # 5. Apply momentum to best params: `best_params = best_params + momentum`.
        set_params_(self.best_params, [p + m for p, m in zip(self.best_params, self.momentum)])
        # 6. Decay momentum: `momentum = momentum * decay`
        set_params_(self.momentum, [m * self.decay for m in self.momentum])
        return loss

    def set_best(self): set_params_(self.params, self.best_params)

class AdaRandomWalk(ClosureOptimizer):
    """
    Adaptive random walk optimizer.

    Generates a random update, evaluates the loss, if loss decreased, we add the update and increase LR, if loss increased discard the update and decrease LR.

    On each iteration:

    1. Generate random update `update = rand * lr`
    2. Apply `update` to best params: `params = best_params + update`.
    3. Evaluate `loss = f(params)`
    4. If loss became smaller, apply update (`best_params` = `params`) and increase LR, else discard update and decrease LR.

    Optionally, if loss decreases, apply `- update` to best params: `best_params = best_params - update`.

    Make sure to call `set_best()` method after optimizing to apply the best found parameters.
    """
    def __init__(self,
                 params:Iterable[torch.Tensor | torch.nn.parameter.Parameter],
                 lr_init = 1e-3,
                 lr_delta = (0.975, 1.1),
                 lr_op = lambda lr, delta: lr * delta,
                 lr_min = 1e-5,
                 apply_neg = False,
                 log_lr = True,
                 rand:Callable = torch.randn_like):
        self.params = torch.nn.ParameterList(params)
        super().__init__(self.params, {})
        self.lr = lr_init
        self.lr_dec, self.lr_inc = lr_delta
        self.lr_min = lr_min
        self.lrop = lr_op
        self.apply_neg = apply_neg

        self.rand = rand
        self.lowest_loss = float('inf')
        self.best_params = clone_params(params)
        self.update = torch.nn.ParameterList([torch.zeros_like(p, device=p.device) for p in self.params])
        self.log_lr = log_lr
        self.lr_history = []

    @property
    def lr_delta(self): return self.lr_dec, self.lr_inc
    @lr_delta.setter
    def lr_delta(self, value): self.lr_dec, self.lr_inc = value

    def step(self, closure: Callable):
        # 1. Generate random update `update = rand * lr`
        set_params_(self.update, [self.rand(p, device=p.device) * self.lr for p in self.params])
        # 2. Apply `update` to best params: `params = best_params + update`.
        set_params_(self.params, [p + u for p, u in zip(self.best_params, self.update)])
        # 2. Evaluate `loss = closure(params)`
        loss = closure()
        # 3. If loss became smaller
        if loss < self.lowest_loss:
            # apply update (`best_params` = `params`)
            self.lowest_loss = loss
            self.best_params = clone_params(self.params)
            # increase LR
            self.lr = self.lrop(self.lr, self.lr_inc)
        # else discard update and decrease LR.
        else:
            self.lr = max(self.lr_min, self.lrop(self.lr, self.lr_dec))
            # Optionally, if loss decreases, apply `- update` to best params: `best_params = best_params - update`.
            if self.apply_neg: set_params_(self.best_params, [p - u for p, u in zip(self.best_params, self.update)])

        if self.log_lr: self.lr_history.append(self.lr)
        return loss

    def set_best(self): set_params_(self.params, self.best_params)

class ShotgunMeta: ... # optimizes and restarts from a new initial condition multiple times.

class MomentumMeta: ...


#  I could decouple various steps of the optimizers, like the momentum... How do I do that???