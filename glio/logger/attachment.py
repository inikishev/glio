from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    import torch
    from . import Logger
from typing import Type
class Attachment:
    def __init__(self, model: "torch.nn.Module", logger: "Logger"):
        self.model = model
        self.logger = logger
        
    def step(self, preds, targets, loss, train, **kwargs): ...
    

class Bundle:
    def __init__(self, model: 'torch.nn.Module', logger: 'Logger', attachments: list[Type[Attachment]]):
        self.model = model
        self.logger = logger
        self.attachments: list[Attachment] = [att(model = model, logger = logger) for att in attachments]
    
    def step(self, preds, targets, loss:'torch.Tensor', train = True, **kwargs):
        for attachment in self.attachments:
            attachment.step(preds = preds, targets = targets, loss = loss, train = train, **kwargs)