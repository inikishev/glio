from torch import nn
import torch.nn.functional as F
import torch
import functools

class GeneralReLU(nn.Module):
    """
    FROM FASTAI COURSES
    """
    def __init__(self, leak = None, sub = None, maxv = None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv
        
    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak else F.relu(x)
        if self.sub is not None: x -= self.sub
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x
    

class SymmetricReLU(nn.Module):
    """Симметричная функция активации на основе ReLU"""
    def __init__(self, leak: float = 0.2, sub: float = 1, maxv: bool = None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub,  maxv
        
    def forward(self, x: torch.Tensor):
        # нижний отрезок
        x = F.leaky_relu(x+self.sub, self.leak) - self.sub
        # верхний отрезок
        x = -F.leaky_relu(-x + self.sub, self.leak) + self.sub
        return x
    

class ElementwiseReLU(nn.Module):
    """Random ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.
    
    `in_size`: `(C, H, W)` for 2d inputs.
    
    Array can be learnable if `learnable` is `True`."""
    def __init__(self, in_size, learnable = False, init = functools.partial(torch.nn.init.uniform_, a=-1, b=1)): 
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
    
    def forward(self, x):
        return torch.maximum(x, self.thresholds)
    
class ElementwiseLeakyReLU(nn.Module):
    """Random leaky ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.
    
    `in_size`: `(C, H, W)` for 2d inputs.
    
    This can't be learnable. (yet)"""
    def __init__(self, in_size, leak=0.1, learnable = False, init = functools.partial(torch.nn.init.uniform_, a=-1, b=1), ): 
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
        self.leak = leak
    
    def forward(self, x):
        x[x < self.thresholds] *= self.leak
        return x
    
if __name__ == "__main__":
    plReLU = ElementwiseLeakyReLU((3, 28, 28), 0.1, True)
    batch = torch.randn(16, 3, 28, 28)
    print(plReLU(batch).shape)
    
class ElementwiseResidualReLU(nn.Module):
    """Random residual leaky ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.
    
    Computes `ElementwiseLeakyReLU(x) + x * residual`
    
    `in_size`: `(C, H, W)` for 2d inputs.
    
    Array can be learnable if `learnable` is `True`. Residual connection means learnable array can be initialized with zeroes."""
    def __init__(self, in_size, residual = 0.5, learnable = False, init = torch.nn.init.zeros_ ): 
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
        self.residual = residual
    
    def forward(self, x):
        return torch.maximum(x, self.thresholds) + x*self.residual 
    
class ElementwiseLeakyResidualReLU(nn.Module):
    """Random residual leaky ReLU: uses a random array as separate ReLU for each individual pixel with random thresholds.
    
    Computes `ElementwiseLeakyReLU(x) + x * residual`
    
    `in_size`: `(C, H, W)` for 2d inputs.
    
    Array can be learnable if `learnable` is `True`. Residual connection means learnable array can be initialized with zeroes."""
    def __init__(self, in_size, leak=0.1, residual = 0.5, learnable = False, init = torch.nn.init.zeros_ ): 
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.zeros(in_size), requires_grad = learnable)
        init(self.thresholds)
        self.residual = residual
        self.leak = leak
    
    def forward(self, x):
        return torch.where(x > self.thresholds, x, x * self.leak) + x*self.residual 

class SinAct(nn.Module):
    """`sin` act function wrapped into an `nn.Module` for convenience, calls `torch.sin(x)`."""
    def __init__(self):
        super().__init__()
    def forward(self, x): return torch.sin(x)

class VAct(nn.Module):
    """A simple piecewise V-shaped activation function"""
    def __init__(self, inverse = False):
        super().__init__()
        self.inverse = inverse
    def forward(self, x):
        if self.inverse: return torch.where(x < 0, x, -x)
        return torch.where(x < 0, -x, x)
    
if __name__ == "__main__":
    plReLU = ElementwiseLeakyReLU((3, 28, 28), 0.1, True)
    batch = torch.randn(16, 3, 28, 28)
    print(plReLU(batch).shape)
    
    
    