import torch
from torch import nn
import kornia
class AffineTransform(nn.Module):
    def __init__(self):
        super(AffineTransform, self).__init__()
        self.theta = nn.Parameter(torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32))

    def forward(self, x):
        return kornia.geometry.transform.affine(x, self.theta)