# Автор - Никишев Иван Олегович группа 224-31

import torch.nn, torch


class Visualize(torch.nn.Module):
    def __init__(self, step = 16):
        super().__init__()
        self.step = step
        self.current_step = 0
        self.logs = []
    
    def forward(self, x):
        # plot x (i need to make visualizer in the visualize submodule)
        if self.current_step % self.step == 0: pass
        self.current_step += 1
        return x
    
class PrintSize(torch.nn.Module):
    def __init__(self, enabled = True):
        super().__init__()
        self.enabled = enabled
    
    def forward(self, x):
        if self.enabled: print(x.shape)
        return x