"""Init"""
from typing import Callable
import torch

from ...torch_tools import apply_init_fn, has_nonzero_weight

def init_uniform(model:torch.nn.Module, a=0.0, b=1.0, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, lambda x: torch.nn.init.uniform_(x, a, b), filt=filt)
def init_normal(model:torch.nn.Module, mean=0.0, std=1.0, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, lambda x: torch.nn.init.normal_(x, mean, std), filt=filt)
def init_constant(model:torch.nn.Module, val, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, lambda x: torch.nn.init.constant_(x, val), filt=filt)
def init_ones(model:torch.nn.Module, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, torch.nn.init.ones_, filt=filt)
def init_zeros(model:torch.nn.Module, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, torch.nn.init.zeros_, filt=filt)
def init_eye(model:torch.nn.Module, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, torch.nn.init.eye_, filt=filt)
def init_dirac(model:torch.nn.Module, groups = 1, filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, lambda x: torch.nn.init.dirac_(x, groups), filt=filt)
def init_xavier_uniform(model:torch.nn.Module, gain = 1., filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, lambda x: torch.nn.init.xavier_uniform_(x, gain), filt=filt)
def init_xavier_normal(model:torch.nn.Module, gain = 1., filt = has_nonzero_weight) -> torch.nn.Module:
    return apply_init_fn(model, lambda x: torch.nn.init.xavier_normal_(x, gain), filt=filt)
