from collections.abc import Callable
from typing import Any
import random
import torch
import joblib
from glio.torch_tools import one_hot_mask, MRISlicer
from glio.python_tools import SliceContainer, reduce_dim, Compose

# RHUH_PATH = r"E:\dataset\RHUH-GBM"
# RHUH_HIST140_TRAIN = rf"{RHUH_PATH}/rhuh hist140 train.joblib"
# RHUH_HIST140_TEST = rf"{RHUH_PATH}/rhuh hist140 test.joblib"
# RHUH_NOHIST140_TRAIN = rf"{RHUH_PATH}/rhuh nohist140 train.joblib"
# RHUH_NOHIST140_TEST = rf"{RHUH_PATH}/rhuh nohist140 test.joblib"

# RHUH_HIST140_NOADC_TRAIN = rf"{RHUH_PATH}/rhuh hist140 noadc train.joblib"
# RHUH_HIST140_NOADC_TEST = rf"{RHUH_PATH}/rhuh hist140 noadc test.joblib"
# RHUH_NOHIST140_NOADC_TRAIN = rf"{RHUH_PATH}/rhuh nohist140 noadc train.joblib"
# RHUH_NOHIST140_NOADC_TEST = rf"{RHUH_PATH}/rhuh nohist140 noadc test.joblib"

# BRATS_PATH = r"E:\dataset\BRaTS2024-GoAT"
# BRATS2024_NOHIST96_TRAIN = rf"{BRATS_PATH}/brats2024 nohist96 train.joblib"
# BRATS2024_NOHIST96_TEST = rf"{BRATS_PATH}/brats2024 nohist96 test.joblib"
# BRATS2024SMALL_HIST96_TRAIN = rf"{BRATS_PATH}/brats2024-small hist train.joblib"
# BRATS2024SMALL_HIST96_TEST = rf"{BRATS_PATH}/brats2024-small hist test.joblib"
# BRATS2024SMALL_NOHIST96_TRAIN = rf"{BRATS_PATH}/brats2024-small nohist train.joblib"
# BRATS2024SMALL_NOHIST96_TEST = rf"{BRATS_PATH}/brats2024-small nohist test.joblib"

__all__ = [
    "RHUH_TRAIN",
    "RHUH_TEST",
    "BRATSGOAT_TRAIN",
    "BRATSGOAT_TEST",
    "BRATSGLI_0_75",
    "BRATSGLI_75_100",
    "BRATSGLI_0_1000",
    "BRATSGLI_1000_1350",
    "get_ds_randslices",
    "get_ds_allsegslices",
    "get_ds_allslices",
    "loader",
    "randcrop",
    'shuffle_channels',
    'shuffle_channels_around',
    'groupwise_tfms',
    'GroupwiseTfms',
    'sliding_inferencen',
    "MRISlicer"
]

RHUH_TRAIN = r"E:\dataset\RHUH-GBM\rhuh full v2 train.joblib"
RHUH_TEST = r"E:\dataset\RHUH-GBM\rhuh full v2 test.joblib"

BRATSGOAT_TRAIN = r"E:\dataset\BRaTS2024-GoAT\brats2024-96 v2 train.joblib"
BRATSGOAT_TEST = r"E:\dataset\BRaTS2024-GoAT\brats2024-96 v2 test.joblib"

BRATSGLI_0_75 = r"E:\dataset\BraTS-GLI v2\brats-gli 0-75.joblib"
BRATSGLI_75_100 = r"E:\dataset\BraTS-GLI v2\brats-gli 75-100.joblib"
BRATSGLI_0_1000 = r"E:\dataset\BraTS-GLI v2\brats-gli 120 0-1000.joblib"
BRATSGLI_1000_1350 = r"E:\dataset\BraTS-GLI v2\brats-gli 120 1000-1350.joblib"

def get_ds_randslices(path, around=1, any_prob = 0.05) -> list[MRISlicer]:
    """Returns one object per study that returns a random slice on call."""
    ds:list[MRISlicer] = joblib.load(path)
    for i in ds: i.set_settings(around = around, any_prob = any_prob)
    return ds

def get_ds_allsegslices(path, around=1, any_prob = 0.05) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a study that contain segmentation + `any_prob` * 100 % objects that return a random slice."""
    MRIs:list[MRISlicer] = joblib.load(path)
    for i in MRIs: i.set_settings(around = around, any_prob = any_prob)
    ds = reduce_dim([i.get_all_seg_slice_callables() for i in MRIs])
    random_slices = reduce_dim([i.get_anyp_random_slice_callables() for i in MRIs])
    ds.extend(random_slices)
    return ds

def get_ds_allslices(path, around=1) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a study that contain segmentation + `any_prob` * 100 % objects that return a random slice."""
    MRIs:list[MRISlicer] = joblib.load(path)
    for i in MRIs: i.set_settings(around = around)
    ds = reduce_dim([i.get_all_slice_callables() for i in MRIs])
    return ds

def loader(x:MRISlicer | Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    return x()

def randcrop(x: tuple[torch.Tensor, torch.Tensor], size = (96,96)):
    if x[0].shape[1] == size[0] and x[0].shape[2] == size[1]: return x
    #print(x[0].shape)
    startx = random.randint(0, (x[0].shape[1] - size[0]) - 1)
    starty = random.randint(0, (x[0].shape[2] - size[1]) - 1)
    return x[0][:, startx:startx+size[0], starty:starty+size[1]].to(torch.float32), one_hot_mask(x[1][startx:startx+size[0], starty:starty+size[1]], 5)

def shuffle_channels(x:torch.Tensor):
    return x[torch.randperm(x.shape[0])]

def shuffle_channels_around(x:torch.Tensor, channels_per = 3):
    num_groups = int(x.shape[0] / channels_per)
    perm = torch.randperm(num_groups, dtype=torch.int32)
    img= x.reshape(num_groups, channels_per, *x.shape[1:])[perm].flatten(0, 1)
    return img

def groupwise_tfms(x:torch.Tensor, tfms, channels_per = 3):
    num_groups = int(x.shape[0] / channels_per)
    groups = x.reshape(num_groups, channels_per, *x.shape[1:]).unbind(0)
    groups = [Compose(tfms)(i) for i in groups]
    return torch.cat(groups, 0)


class GroupwiseTfms:
    def __init__(self, tfms, channels_per = 3):
        self.tfms = tfms
        self.channels_per = channels_per
    def __call__(self, x):
        return groupwise_tfms(x, self.tfms, self.channels_per)



def sliding_inference(inputs:torch.Tensor, inferer, size, overlap=0.5, progress=False, batch_size=32):
    """Input must be a 4D C* tensor, around is 1. Sliding inference using gaussian overlapping."""
    from monai.inferers import SlidingWindowInferer # type:ignore
    inputs = inputs.swapaxes(0,1) # First spatial dimension becomes batch dimension
    # construct the inferer
    sliding = SlidingWindowInferer(size, batch_size, overlap, mode='gaussian', progress=progress)
    # run inference
    results = sliding(inputs, inferer).cpu() # type:ignore
    # return C* tensor
    return results.swapaxes(0,1) # type:ignore

def sliding_inference1(inputs:torch.Tensor, inferer, size, overlap=0.5, progress=False, batch_size=32):
    """Input must be a 4D C* tensor, around is 1. Sliding inference using gaussian overlapping."""
    from monai.inferers import SlidingWindowInferer # type:ignore
    inputs = inputs.swapaxes(0,1) # First spatial dimension becomes batch dimension

    # input is 3 neighbouring slices, this creates a new dimension and flattens it so that each slice contains 12 channels,
    # each channel has 4 modalities and 3 neighbouring slices per modality.
    # this also makes the input smaller by 1 pixel on each side of the first spatial dimension
    inputs = torch.stack(( inputs[:-2], inputs[1:-1],inputs[2:]), 2).flatten(1,2)
    sliding = SlidingWindowInferer(size, batch_size, overlap, mode='gaussian', progress=progress)

    results = sliding(inputs, inferer).cpu() # type:ignore

    # add 1 pixel padding to each side of the first spatial dimension restore the original shape
    padding = torch.zeros((1, *results.shape[1:],)) # type:ignore
    results = torch.cat((padding, results, padding)) # type:ignore

    # return C* tensor
    return results.swapaxes(0,1) # type:ignore

def sliding_inference2(inputs:torch.Tensor, inferer, size, overlap=0.5, progress=False, batch_size=32):
    """Input must be a 4D C* tensor, around is 1. Sliding inference using gaussian overlapping."""
    from monai.inferers import SlidingWindowInferer # type:ignore
    inputs = inputs.swapaxes(0,1) # First spatial dimension becomes batch dimension

    # input is 3 neighbouring slices, this creates a new dimension and flattens it so that each slice contains 12 channels,
    # each channel has 4 modalities and 3 neighbouring slices per modality.
    # this also makes the input smaller by 1 pixel on each side of the first spatial dimension
    inputs = torch.stack(( inputs[:-4], inputs[1:-3], inputs[2:-2], inputs[3:-1],inputs[4:]), 2).flatten(1,2)
    sliding = SlidingWindowInferer(size, batch_size, overlap, mode='gaussian', progress=progress)

    results = sliding(inputs, inferer).cpu() # type:ignore

    # add 2 pixel padding to each side of the first spatial dimension restore the original shape
    padding = torch.zeros((2, *results.shape[1:],)) # type:ignore
    results = torch.cat((padding, results, padding)) # type:ignore

    # return C* tensor
    return results.swapaxes(0,1) # type:ignore

def sliding_inferencen(inputs:torch.Tensor, inferer, size, around, overlap=0.5, progress=False, batch_size=32):
    """Input must be a 4D C* tensor, around is 1. Sliding inference using gaussian overlapping."""
    if around == 0: return sliding_inference(inputs=inputs, inferer=inferer, size=size, overlap=overlap, progress=progress, batch_size=batch_size)

    from monai.inferers import SlidingWindowInferer # type:ignore
    inputs = inputs.swapaxes(0,1) # First spatial dimension becomes batch dimension

    # input is 3 neighbouring slices, this creates a new dimension and flattens it so that each slice contains 12 channels,
    # each channel has 4 modalities and 3 neighbouring slices per modality.
    # this also makes the input smaller by 1 pixel on each side of the first spatial dimension
    channels = around * 2 + 1
    inputs = torch.stack([inputs[i: (- ((channels - 1) - i) if (i < (channels - 1)) else None)] for i in range(channels)], 2).flatten(1,2)
    sliding = SlidingWindowInferer(size, batch_size, overlap, mode='gaussian', progress=progress)

    results = sliding(inputs, inferer).cpu() # type:ignore

    # add 2 pixel padding to each side of the first spatial dimension restore the original shape
    padding = torch.zeros((around, *results.shape[1:],)) # type:ignore
    results = torch.cat((padding, results, padding)) # type:ignore

    # return C* tensor
    return results.swapaxes(0,1) # type:ignore