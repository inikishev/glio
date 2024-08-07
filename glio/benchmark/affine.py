import os
import torch
from torchzero.nn.layers.affine import Affine
from torchzero.nn.layers.learnable_ops import LearnableMulAdd
from ..loaders.image import imwrite
from .benchmark import Benchmark
from ..transforms.intensity import znorm
from ..transforms.format import EnsureTensor, ensure_channel_first
from ..python_tools import auto_compose

def mse(x:torch.Tensor, y:torch.Tensor): return ((x - y)**2).mean()

class AffineBenchmark(Benchmark):
    def __init__(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        init_tfm_input = (EnsureTensor(dtype=torch.float32), ensure_channel_first, znorm),
        init_tfm_target = (EnsureTensor(dtype=torch.float32), ensure_channel_first, znorm),
        tfm_input = None,
        tfm_preds = None,
        tfm_target = None,
        loss_fn = mse,
        learn_norm = True,
        ndim = 2,
        device = torch.device("cuda"),
        dtype = torch.float32,
        unsqueeze = False,
    ):
        super().__init__()
        self.input = auto_compose(init_tfm_input)(input).to(device).to(dtype)
        self.target = auto_compose(init_tfm_target)(target).to(device).to(dtype)
        self.tfm_input = auto_compose(tfm_input)
        self.tfm_preds = auto_compose(tfm_preds)
        self.tfm_target = auto_compose(tfm_target)

        self.loss_fn = loss_fn
        if learn_norm: self.model = torch.nn.Sequential(LearnableMulAdd(), Affine(ndim = ndim))
        else: self.model = Affine(ndim = ndim)
        self.model = self.model.to(device).to(dtype)

        if unsqueeze:
            self.input = self.input.unsqueeze(0)
            self.target = self.target.unsqueeze(0)

        if self.input.ndim - ndim != 2: raise ValueError(f"Input ndim should be {self.input.ndim - 2}, got input of shape {self.input.shape}")

    def forward(self, input) -> torch.Tensor:
        self.num_evals += 1
        return self.model(input)

    def one_step(self):
        self.preds = self.forward(self.tfm_input(self.input))
        loss = self.loss_fn(self.tfm_preds(self.preds), self.tfm_target(self.target))
        return loss

    def save_vis(self, dir):
        imwrite(self.preds[0], os.path.join(dir, 'preds.png'))