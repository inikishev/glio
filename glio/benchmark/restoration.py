import os
import torch
from ..loaders.image import imwrite
from .benchmark import Benchmark, InputModel
from ..transforms.intensity import znorm
from ..transforms.format import EnsureTensor
from ..python_tools import auto_compose
from ..random.generators import make_seeded

def mse(x:torch.Tensor, y:torch.Tensor): return ((x - y)**2).mean()

class RestoreImageBenchmark(Benchmark):
    def __init__(
        self,
        target: torch.Tensor,
        tfm_init = (EnsureTensor(dtype=torch.float32), znorm),
        tfm_preds = None,
        tfm_target = None,
        loss_fn = mse,
        init = make_seeded(torch.randn),
        device = torch.device("cuda"),
        dtype = torch.float32,
    ):
        super().__init__()
        self.target = auto_compose(tfm_init)(target).to(device).to(dtype)
        self.tfm_preds = auto_compose(tfm_preds)
        self.tfm_targets = auto_compose(tfm_target)
        self.loss_fn = loss_fn
        self.model = InputModel(init(target.shape).to(device)).to(device).to(dtype)

    def forward(self) -> torch.Tensor:
        self.num_evals += 1
        return self.model()

    def one_step(self):
        preds = self.forward()
        loss = self.loss_fn(self.tfm_preds(preds), self.tfm_targets(self.target))
        return loss

    def save_vis(self, dir):
        imwrite(self.model(), os.path.join(dir, 'preds.png'))