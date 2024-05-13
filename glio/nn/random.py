import torch
from torch import nn
class RandomChoice(nn.Module):
    """Each forward pass randomly chooses a layer from all provided layers"""
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        return self.layers[torch.randint(len(self.layers), (1,))](x)
    
