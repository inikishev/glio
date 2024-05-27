import torch
from torch import nn
import torchvision.transforms.v2
import kornia
class AffineKornia(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return kornia.geometry.transform.affine(x, self.theta)
    
class AffineTorchvision(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.translate = nn.Parameter(torch.tensor([0,0], dtype=torch.float32), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.shear = nn.Parameter(torch.tensor([0,0], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return torchvision.transforms.v2.functional.affine(x, angle=self.angle, translate=self.translate, scale=self.scale, shear=self.shear)