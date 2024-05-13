from collections.abc import Callable
import numpy as np, torch
from .optimizer import ClosureOptimizer
from ..transforms import z_normalize
from ..loaders.image import read as imread
from ..python_tools import identity_if_none
from ..progress_bar import PBar
class OptimizerTest: ...


class OTImageReconstruct(OptimizerTest):
    def __init__(self, image, tfm = None, loss:Callable = torch.nn.MSELoss(), init = torch.randn_like, device = None):
        # load image
        if isinstance(image, np.ndarray):
            self.image = z_normalize(torch.tensor(image, dtype=torch.float32))
        elif isinstance(image, torch.Tensor):
            self.image = z_normalize(image.to(torch.float32))
        elif isinstance(image, str):
            self.image = z_normalize(imread(image).to(torch.float32))

        # prep args
        self.tfm = identity_if_none(tfm)
        self.loss = loss
        if device is None: self.device = self.image.device
        else: self.device = device

        # parameters
        self.params = torch.nn.ParameterList([init(self.image, device=self.image.device)])

        # logs
        self.losses = []
        self.preds = []

    @torch.no_grad
    def closure(self):
        preds = self.params[0]
        targets = self.tfm(self.image)
        return self.loss(preds, targets)

    def parameters(self): return self.params


    def __call__(self, optimizer: ClosureOptimizer, niter:int, logloss = True, logpreds = True, pbar = True):
        r = PBar(range(niter)) if pbar else range(niter)
        for _ in r:
            loss = optimizer.step(self.closure)
            if logloss: self.losses.append(loss)
            if logpreds: self.preds.append(self.params[0].clone())