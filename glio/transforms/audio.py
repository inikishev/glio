import numpy as np, torch

def to_duration(x:torch.Tensor, sr, sec):
    x = x[:, :int(sr*sec)]
    if x.shape[1] < int(sr*sec):
        x = torch.cat([x, torch.zeros(x.shape[0], int(sr*sec - x.shape[1]))], dim=1)

    return x

def hardclip(x:torch.Tensor | np.ndarray, low=-1, high=1):
    return x.clip(low, high)

def normalize(x:torch.Tensor):
    return x / x.abs().max()

def dcoffset_fix(x):
    return x - x.mean()

def cut(x, sr, start_sec = None, end_sec = None):
    if start_sec is None:
        start_sec = 0
    if end_sec is None:
        end_sec = x.shape[1] / sr
    return x[:, int(sr*start_sec):int(sr*end_sec)]