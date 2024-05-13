# Автор - Никишев Иван Олегович группа 224-31

from torch import nn

    
class Residual(nn.Module):
    def __init__(self, layers:list[nn.Module]):
        super().__init__()
        self.sequential = nn.Sequential(*layers)
    def forward(self, x):
        return self.sequential(x) + x
    
class Residual_BatchNorm(nn.Module):
    def __init__(self, layers:list[nn.Module], out_channels:int):
        super().__init__()
        batchnorm = nn.BatchNorm3d(out_channels)
        batchnorm.weight.data[:] = 0
        batchnorm.bias.data[:] = 0
        self.sequential = nn.Sequential(*layers, batchnorm)
    def forward(self, x):
        return self.sequential(x) + x
    
from .elementwise import Elementwise
class Residual_Elementwise(nn.Module):
    def __init__(self, layers:list[nn.Module], out_size:int):
        super().__init__()
        piecewise = Elementwise(out_size)
        piecewise.weight.data[:] = 0
        piecewise.bias.data[:] = 0
        self.sequential = nn.Sequential(*layers, piecewise)
    def forward(self, x):
        return self.sequential(x) + x
    
    
def Conv3dBlock(in_channels, out_channels, kernel_size, stride, padding = 0, norm = False, act = True):
    modules = []
    modules.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
    if act: modules.append(nn.ReLU())
    if norm: modules.append(nn.BatchNorm3d(out_channels))
    return nn.Sequential(*modules)

class ResBlock_Conv3d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv3dBlock(in_channels, out_channels, kernel_size=2, stride=1, padding = 1, norm=False)
        self.conv2 = Conv3dBlock(in_channels, out_channels, kernel_size=2, stride=1, padding = 0, norm=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.norm1.bias.data[:] = 0
        self.norm1.weight.data[:] = 0
        self.act = nn.ReLU()
        self.NiN = nn.Conv3d(in_channels, out_channels, 1, stride = 1)

    def forward(self, x): 
        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        x2 = self.norm1(x2)
        x = self.norm2(self.NiN(x))
        return x+x2
