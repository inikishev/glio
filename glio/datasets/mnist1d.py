import numpy as np
import torch
import joblib
from stuff.found.torch.datasets.mnist1d.data import get_dataset, get_dataset_args
from ..data import DSClassification, DSToTarget
from ..torch_tools import CUDA_IF_AVAILABLE
from ..transforms.format import EnsureTensor

__all__ = ["get_mnist1d_classification", "get_mnist1d_autoenc"]

def _unsqueeze0(x): return torch.unsqueeze(x, 0)

def get_mnist1d_classification(
    path="D:/datasets/mnist1d_data.pkl",
    download=False,
    device=CUDA_IF_AVAILABLE,
    tfm_train=None,
    tfm_test=None,
    dtype=torch.float32,
):
    args = get_dataset_args()
    data = get_dataset(args, path, download=download)
    dstrain = DSClassification()
    dstest = DSClassification()
    x = data['x']
    y = data['y']
    xtest = data['x_test']
    ytest = data['y_test']

    loader = [EnsureTensor(device=device,dtype=dtype), _unsqueeze0]
    for sx, sy in zip(x, y):
        dstrain.add_sample(
            data=sx,
            target=torch.nn.functional.one_hot(torch.from_numpy(np.array([sy])), 10)[0].to(device, dtype),  # pylint:disable=E1102
            loader=loader,
            transform=tfm_train,
            target_encoder=None,
        )

    for sx, sy in zip(xtest, ytest):
        dstest.add_sample(
            data=sx,
            target=torch.nn.functional.one_hot(torch.from_numpy(np.array([sy])), 10)[0].to(device, dtype),  # pylint:disable=E1102
            loader=loader,
            transform=tfm_test,
            target_encoder=None,
        )  # pylint:disable=E1102

    dstrain.preload(); dstest.preload()
    return dstrain, dstest

def get_mnist1d_autoenc(path='D:/datasets/mnist1d_data.pkl', download=False, device=CUDA_IF_AVAILABLE, dtype=torch.float32,):
    args = get_dataset_args()
    data = get_dataset(args, path, download=download)
    dstrain = DSToTarget()
    dstest = DSToTarget()

    loader = [EnsureTensor(device=device,dtype=dtype), _unsqueeze0]

    dstrain.add_samples(data = data["x"], loader = loader)

    dstest.add_samples(data = data['x_test'], loader = loader)

    dstrain.preload(); dstest.preload()
    return dstrain, dstest
