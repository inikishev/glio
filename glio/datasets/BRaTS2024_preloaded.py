from typing import Any
import torch
import joblib
from glio.torch_tools import one_hot_mask
from glio.python_tools import SliceContainer


def get_ds_2d(path = r"E:\dataset\BRaTS2024-GoAT\train hist.joblib") -> list[tuple[torch.Tensor, torch.Tensor]]:
    ds:list[list[tuple[SliceContainer,SliceContainer]]] = joblib.load(path)
    return [(i[0](), i[1]()) for j in ds for i in j]

def loader_2d(sample:tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return sample[0].to(torch.float32), one_hot_mask(sample[1], 4)

def get_ds_around(path = r"E:\dataset\BRaTS2024-GoAT\train hist.joblib", around=1) -> list[tuple[list[torch.Tensor],torch.Tensor]]:
    ds:list[list[tuple[SliceContainer,SliceContainer]]] = joblib.load(path)
    res = []
    for slices in ds:
        for i in range(around, len(slices) - around):
            stack = slices[i - around : i + around + 1]
            images = [s[0]() for s in stack]
            seg = slices[i][1]()
            res.append((images, seg))
    return res

def loader_around(sample:tuple[list[torch.Tensor],torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(sample[0], 0).to(torch.float32), one_hot_mask(sample[1], 4)

def loader_around_fix(sample:tuple[list[torch.Tensor],torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(sample[0], 0).to(torch.float32)[:,:96,:96], one_hot_mask(sample[1], 4)[:,:96,:96]