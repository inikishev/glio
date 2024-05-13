"""Матричное умножение на обучаемую матрицу"""
import torch

class LMLinear(torch.nn.Module):
    def __init__(self, in_size, bias = True, init = torch.nn.init.kaiming_normal_):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, in_size[0]))
        self.bias = torch.nn.Parameter(torch.Tensor(in_size[1])) if bias else None
        init(self.weight)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.bias is not None: return (self.weight @ x + self.bias).squeeze(1)
        return (self.weight @ x).squeeze(1)

if __name__ == "__main__":
    test = torch.randn(16, 10, 5)
    model = LMLinear((10, 5))
    print(model(test).shape)
