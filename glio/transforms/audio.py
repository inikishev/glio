import torch

def to_duration(x, sr, sec):
    x = x[:, :int(sr*sec)]
    if x.shape[1] < int(sr*sec):
        x = torch.cat([x, torch.zeros(x.shape[0], int(sr*sec - x.shape[1]))], dim=1)

    return x

def hardclip(x, low=-1, high=1):
    return torch.clamp(x, low, high)

def normalize(x):
    return x / x.abs().max()

def normclip(x, low=-1, high=1, norm = 0.5):
    xmax = x.abs().max()
    if xmax > 1: 
        xmax = xmax - ((xmax - 1) * norm)
        x = x / xmax
    else: return normalize(x) # TODO
    return torch.clamp(x, low, high)

def dcoffset_fix(x):
    return x - x.mean()

def cut(x, sr, start_sec = None, end_sec = None):
    if start_sec is None:
        start_sec = 0
    if end_sec is None:
        end_sec = x.shape[1] / sr
    return x[:, int(sr*start_sec):int(sr*end_sec)]