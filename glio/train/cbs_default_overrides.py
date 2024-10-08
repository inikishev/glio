from typing import TYPE_CHECKING, Literal
from contextlib import nullcontext
import torch, torch.utils.data
from ..design.event_model import EventCallback, MethodCallback
if TYPE_CHECKING:
    from .Learner import Learner

__all__ = [
    'OneBatchClosureCB',
    "OneBatchClosureWithNoBackwardCB",
    "DisableTorchGradCB",
    "DisableTorchGradAndBackwardCB",
    "PassLossToOptimizerStepCB",
    "SimpleMomentumCB",
    "CallTrainAndEvalOnOptimizerCB",
    "AddLossReturnedByModelToLossInGetLossCB",
    "AddLossReturnedByModelToLossInBackwardCB",
]
class OneBatchClosureCB(EventCallback):
    event = "one_batch"
    def __call__(self, learner: "Learner", inputs: torch.Tensor, targets: torch.Tensor, train=True):
        if train: learner.train()
        else: learner.eval()
        if learner.accelerator is None: inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        with nullcontext() if train else torch.no_grad():
            if train:
                def closure(backward=True):
                    learner.preds = learner.forward(inputs)
                    learner.loss = learner.get_loss(learner.preds, targets) # type:ignore
                    if backward:
                        learner.zero_grad() # type:ignore
                        learner.backward()
                    return learner.loss
                learner.optimizer_step(closure) # type:ignore
                learner.scheduler_step()
            else:
                learner.preds = learner.forward(inputs)
                learner.loss = learner.get_loss(learner.preds, targets) # type:ignore


class OneBatchClosureWithNoBackwardCB(EventCallback):
    event = "one_batch"
    def __call__(self, learner: "Learner", inputs: torch.Tensor, targets: torch.Tensor, train=True):
        if train: learner.train()
        else: learner.eval()
        if learner.accelerator is None: inputs, targets = inputs.to(learner.device), targets.to(learner.device)
        with nullcontext() if train else torch.no_grad():
            if train:
                def closure():
                    learner.zero_grad() # type:ignore
                    learner.preds = learner.forward(inputs)
                    learner.loss = learner.get_loss(learner.preds, targets) # type:ignore
                    #learner.backward()
                    return learner.loss
                learner.optimizer_step(closure) # type:ignore
                learner.scheduler_step()
            else:
                learner.preds = learner.forward(inputs)
                learner.loss = learner.get_loss(learner.preds, targets) # type:ignore


class DisableTorchGradCB(MethodCallback):
    order = 100
    def zero_grad(self, learner: "Learner"): pass
    def backward(self, learner: "Learner"): pass
    def enter(self, learner:"Learner"):
        torch.set_grad_enabled(False)

    def exit(self, learner: "Learner"):
        torch.set_grad_enabled(True)

    def before_batch(self, learner: "Learner"):
        torch.set_grad_enabled(False)

class DisableTorchGradAndBackwardCB(MethodCallback):
    order = 100
    def backward(self, learner: "Learner"): pass
    def enter(self, learner:"Learner"):
        torch.set_grad_enabled(False)

    def exit(self, learner: "Learner"):
        torch.set_grad_enabled(True)

    def before_batch(self, learner: "Learner"):
        torch.set_grad_enabled(False)

class PassLossToOptimizerStepCB(MethodCallback):
    def optimizer_step(self, learner: "Learner"):
        learner.optimizer.step(learner.loss) # type:ignore

class SimpleMomentumCB(MethodCallback):
    def __init__(self, momentum=0.9):
        self.momentum = momentum
    @torch.no_grad
    def zero_grad(self, learner: "Learner"):
        for p in learner.model.parameters():
            if p.grad is not None: p.grad *= self.momentum

class CallTrainAndEvalOnOptimizerCB(MethodCallback):
    def train(self, learner: "Learner"):
        if hasattr(learner.model, "train") and callable(learner.model.train): learner.model.train()
        if hasattr(learner.optimizer, "train") and callable(learner.optimizer.train): learner.optimizer.train() # type:ignore

    def eval(self, learner: "Learner"):
        if hasattr(learner.model, "eval") and callable(learner.model.eval): learner.model.eval()
        if hasattr(learner.optimizer, "eval") and callable(learner.optimizer.eval): learner.optimizer.eval() # type:ignore


class AddLossReturnedByModelToLossInGetLossCB(MethodCallback):
    def forward(self, learner: "Learner", inputs: torch.Tensor):
        returned_value = learner.model(inputs)
        if isinstance(returned_value, torch.Tensor):
            learner.preds = returned_value
            learner.loss_returned_by_model = None
        else: learner.preds, learner.loss_returned_by_model = returned_value
        return learner.preds

    def get_loss(self, learner: "Learner", preds:torch.Tensor, targets:torch.Tensor):
        if learner.loss_returned_by_model is not None: learner.loss = learner.loss_fn(preds, targets) + learner.loss_returned_by_model # type:ignore
        else: learner.loss = learner.loss_fn(preds, targets) # type:ignore
        return learner.loss

class AddLossReturnedByModelToLossInBackwardCB(MethodCallback):
    def forward(self, learner: "Learner", inputs: torch.Tensor):
        returned_value = learner.model(inputs)
        if isinstance(returned_value, torch.Tensor):
            learner.preds = returned_value
            learner.loss_returned_by_model = None
        else: learner.preds, learner.loss_returned_by_model = returned_value
        return learner.preds

    def backward(self, learner: "Learner"):

        if learner.loss_returned_by_model is not None:
            if learner.accelerator is None: (learner.loss + learner.loss_returned_by_model).backward()
            else: learner.accelerator.backward((learner.loss + learner.loss_returned_by_model))
        else:
            if learner.accelerator is None: learner.loss.backward()
            else: learner.accelerator.backward(learner.loss)
