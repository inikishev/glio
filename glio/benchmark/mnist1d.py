import os
import torch
from torchzero.nn.layers.affine import Affine
from torchzero.nn.layers.learnable_ops import LearnableMulAdd
from ..loaders.image import imwrite
from .benchmark import Benchmark
from ..transforms.intensity import znorm
from ..transforms.format import EnsureTensor, ensure_channel_first
from ..python_tools import auto_compose
from ..torch_tools import CUDA_IF_AVAILABLE
from ..datasets.mnist1d import get_mnist1d_classification, get_mnist1d_autoenc
def mse(x:torch.Tensor, y:torch.Tensor): return ((x - y)**2).mean()

class MNIST1DClassificationBenchmark(Benchmark):
    def __init__(
        self,
        path='D:/datasets/mnist1d_data.pkl',
        download=False,
        tfm_train = None,
        tfm_test = None,
        device=CUDA_IF_AVAILABLE,
        dtype=torch.float32,

    ):
        super().__init__()
        self.dstrain, self.dstest = get_mnist1d_classification(
            path=path,
            download=download,
            device=device,
            dtype=dtype,
            tfm_train=tfm_train,
            tfm_test=tfm_test,
        )