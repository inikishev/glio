from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    import torch
    from . import Logger

from .attachment import Attachment

def accuracy(preds: 'torch.Tensor', targets: 'torch.Tensor'):
    return (preds.argmax(dim=1) == targets).float().mean()

class MetricLogger(Attachment):
    name: str  = '...'
    def __init__(self, model: "torch.nn.Module", logger: "Logger"):
        super().__init__(model, logger)
        self.test_logs: list[torch.Tensor | float] = []
    
    def eval(self, preds: 'torch.Tensor', targets: 'torch.Tensor', loss:'torch.Tensor | float', train = True, **kwargs) -> float: ...
    def step(self, preds: 'torch.Tensor', targets: 'torch.Tensor', loss:'torch.Tensor | float', train = True, **kwargs):
        if train:
            if len(self.test_logs) != 0:
                self.test_logs = [float(i.detach().cpu()) for i in self.test_logs] # pyright: ignore
                self.logger.add(f"test {self.name}", sum(self.test_logs) / len(self.test_logs), self.model.total_batch)
                self.test_logs = []
            
            self.logger.add(f"train {self.name}", self.eval(preds=preds,targets=targets,loss=loss,train=train,**kwargs), self.model.total_batch)
        else:
            self.test_logs.append(self.eval(preds=preds,targets=targets,loss=loss,train=train,**kwargs))
    
class Accuracy(MetricLogger):
    name = 'accuracy'
    def eval(self, preds: 'torch.Tensor', targets: 'torch.Tensor', **kwargs): return accuracy(preds, targets) # pyright:ignore
        
class Loss(MetricLogger):
    name = 'loss'
    def eval(self, loss: 'torch.Tensor', **kwargs): return loss # pyright:ignore