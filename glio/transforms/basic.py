# Автор - Никишев Иван Олегович группа 224-31
from typing import Optional, Any
import torch, numpy as np
from ..python_tools import flatten

class Compose:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args, (list, tuple)): self.funcs = args[0]
        else: self.funcs = args

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x

    def plot(self): ... # plot with a sample image

    def __len__(self): return len(self.funcs)

    def __str__(self):
        if len(self.funcs) < 3: return f'alien.Compose({self.funcs})'
        newline = '\n'
        return f'alien.Compose({newline}     - {(newline+"     - ").join([repr(i) for i in self.funcs])}{newline})'
    def __repr__(self): return self.__str__()

def compose_if_needed(funcs):
    """If `funcs` is a list, makes a callable `Compose` out of them. Otherwise just returns `funcs`."""
    if isinstance(funcs, (list, tuple)): return Compose(funcs)
    return funcs

from ..python_tools import apply_tree, get_first_recursive
class ComposeTree(Compose):
    """OBSOLETE"""
    def __init__(self, funcs: list):
        self.funcs = funcs

    def __call__(self, x, cached = None):
        return apply_tree(x, self.funcs, cached)

    def first(self, x):
        return get_first_recursive(self.funcs)(x)

    def __len__(self): return len(flatten(self.funcs))




# TO CHANNELS
def fToChannels(img: torch.Tensor, num_channels) -> torch.Tensor:
    """
    Makes sure your image has specified number of channels in the normal C, H, W format. Missing channels will be copied from existing ones, extra channels will be deleted.
    """
    if img.ndim == 2: return img.repeat(num_channels, 1, 1)
    if img.ndim == 3:
        if img.shape[0] == num_channels: return img
        if img.shape[0] == 1: return img.repeat(num_channels, 1, 1)
        if img.shape[0] > num_channels: return img[:num_channels]
        return torch.cat((img, img[:num_channels - img.shape[0]]),)
    else: raise NotImplementedError

class ToChannels(torch.nn.Module):
    def __init__(self, num_channels):
        """
        Makes sure all images have specified number of channels in the normal C, H, W format. Missing channels will be copied from existing ones, extra channels will be deleted.
        """
        super().__init__()
        if not isinstance(num_channels, int): raise TypeError(f"num_channels must be an integer but it is {type(num_channels)}")
        self.num_channels = num_channels

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fToChannels(img, self.num_channels)

# TO SIZE
import torchvision.transforms.v2.functional
def fCropToSize(img: torch.Tensor, h, w):
    """
    First crops the image to center to match aspect ratio, then resizes to the specified size
    """
    additional_dim = False
    if img.dim() == 2:
        img = img.unsqueeze(0)
        additional_dim = True
    h_scale = h / img.shape[-2]
    w_scale = w / img.shape[-1]
    scale = max(h_scale, w_scale)
    img = torchvision.transforms.v2.functional.resize(img, [int(img.shape[-2]*scale), int(img.shape[-1]*scale)], antialias=True)
    img = torchvision.transforms.v2.functional.center_crop(img, [h, w])
    return img[0] if additional_dim else img

class CropToSize(torch.nn.Module):
    def __init__(self, h, w):
        """
        First crops the image to center to match aspect ratio, then resizes to the specified size
        """
        super().__init__()
        if not isinstance(h, int): raise TypeError(f"num_channels must be an integer but it is {type(h)}")
        if not isinstance(w, int): raise TypeError(f"num_channels must be an integer but it is {type(w)}")
        self.h = h
        self.w = w

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fCropToSize(img, self.h, self.w)


# TO DIMS
def fToDims(img: torch.Tensor, dims):
    """
    Makes all tensors have specified numner of dimensions. Missing dimensions will be unsqueezed, extra dimensions flattened.
    """
    img = img.detach().clone()
    while img.dim()>dims: img = torch.flatten(img, 0, 1)
    while img.dim()<dims: img = torch.unsqueeze(img, 0)
    return img

class ToDims(torch.nn.Module):
    def __init__(self, dims):
        """
        Makes all tensors have specified numner of dimensions. Missing dimensions will be unsqueezed, extra dimensions flattened.
        """
        super().__init__()
        self.dims = dims

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fToDims(img, self.dims)

def fToRange(img:torch.Tensor, min, max):
    """
    Tensor will be rescaled so that lowest value will be `min`, highest will be `max`.
    """
    if isinstance(img, torch.Tensor): img = img.detach().clone().float()
    elif isinstance(img, np.ndarray): img = img.copy()
    img -= img.min()
    if img.max() > 0: img /= img.max()
    img *= (max-min)
    img += min
    return img

class ToRange(torch.nn.Module):
    def __init__(self, min = -1, max = 1):
        """
        Tensor will be rescaled so that lowest value will be `min`, highest will be `max`.
        """
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fToRange(img, self.min, self.max)

def fUnNormalize(tensor, mean, std):
    """Undoes v2.Normalize"""
    inverse_mean = [-mean[i]/std[i] for i in range(len(mean))]
    inverse_std = [1/std[i] for i in range(len(mean))]
    transform = torchvision.transforms.v2.Normalize(inverse_mean, inverse_std)
    return transform(tensor)

class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        """
        Tensor will be rescaled so that lowest value will be `min`, highest will be `max`.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fUnNormalize(img, self.mean, self.std)


import random
def fFillRandomSquare(image: torch.Tensor, hrange = (0.25,0.75), wrange = (0.25,0.75), color = (0,0,0)):
    """
    Fills a randomly positioned square of random size `hrange`%, `wrange`% with `color`
    """
    image = image.detach().clone()
    h = image.shape[1]
    w = image.shape[2]
    sizeh = random.randrange(int(h*hrange[0]),int(h*hrange[1]))
    sizew = random.randrange(int(w*wrange[0]),int(w*wrange[1]))
    starth = random.randrange(0,h-sizeh)
    startw = random.randrange(0,w-sizew)
    for i, ch in enumerate(image):
        ch[starth:starth+sizeh,startw:startw+sizeh] = color[i]
    return image

class FillRandomSquare(torch.nn.Module):
    def __init__(self, hrange = (0.25,0.75), wrange = (0.25,0.75), color = (0,0,0)):
        """
        Fills a randomly positioned square of random size with `color`. `hrange` and `wrange` control the range of the width and the height of the square relative to image size.
        """
        super().__init__()
        self.hrange = hrange
        self.wrange = wrange
        self.color = color

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fFillRandomSquare(img, self.hrange, self.wrange, self.color)

def fFillSide(image: torch.Tensor, side = 'bottom', size = 0.5, color = (0,0,0)):
    """
    Fills `size` of the image `side` with `color`
    """
    image = image.detach().clone()
    side = side.lower()
    for i, ch in enumerate(image):
        if side in ('bottom', 'top'):
            split = int(image.size(1)*size)
            if side == 'bottom': ch[-split:] = color[i]
            elif side == 'top': ch[:split] = color[i]
        if side in ('left', 'right'):
            split = int(image.size(2)*size)
            if side == 'right': ch[:,-split:] = color[i]
            elif side == 'left': ch[:,:split] = color[i]
    return image

class FillSide(torch.nn.Module):
    def __init__(self, side = 'bottom', size = 0.5, color = (0,0,0)):
        """
        Fills a randomly positioned square of random size with `color`. `hrange` and `wrange` control the range of the width and the height of the square relative to image size.
        """
        super().__init__()
        self.side = side
        self.size = size
        self.color = color

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return fFillSide(img, self.side, self.size, self.color)


def fToChannelsFirst(image: torch.Tensor | np.ndarray):
    """
    Converts to channels first, if your image has less pixels along 1st dimension than channels than it won't work because how is it meant to know which dimension is the channel one.
    """

    if image.ndim == 3 and image.shape[0] <= image.shape[2]: return image
    if isinstance(image, torch.Tensor):
        if image.ndim == 2: return torch.unsqueeze(image, 2)
        return image.permute(2,0,1) # h w c -> c h w
    if isinstance(image, np.ndarray):
        if image.ndim == 2: return np.expand_dims(image, 2)
        return np.transpose(image, (2,0,1)) # h w c -> c h w

class ToChannelsFirst(torch.nn.Module):
    def __init__(self):
        """
        This won't work if your images are smaller than 5x5, or have more than 4 channels!
        """
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor | np.ndarray:
        return fToChannelsFirst(img)

def fToChannelsLast(image: torch.Tensor | np.ndarray):
    """
    This won't work if your images are smaller than 5x5, or have more than 4 channels!
    """
    if image.ndim == 3 and image.shape[0] >= image.shape[2]: return image
    if isinstance(image, torch.Tensor):
        if image.ndim == 2: return torch.unsqueeze(image, 2)
        return image.permute(1,2,0) # c h w -> h w c
    if isinstance(image, np.ndarray):
        if image.ndim == 2: return np.expand_dims(image, 2)
        return np.transpose(image, (1,2,0)) # c h w -> h w c

class ToChannelsLast(torch.nn.Module):
    def __init__(self):
        """
        This won't work if your images are smaller than 5x5, or have more than 4 channels!
        """
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor | np.ndarray:
        return fToChannelsLast(img)


def fMaxSize(img: torch.Tensor, max_size):
    """
    If any of img dimensions is bigger than max_size, image will be resized by an integer factor by slicing (so it is nearest interpolation)
    """
    if isinstance(img, torch.Tensor): img = img.detach().clone()
    elif isinstance(img, np.ndarray): img = img.copy()
    if img.ndim == 3 and img.shape[0] < img.shape[2]: ch_first = True
    else: ch_first = False
    for i,v in enumerate(img.shape):
        if v > max_size:
            img = img[:, ::int(v//max_size), ::int(v//max_size)] if ch_first else img[::int(v//max_size), ::int(v//max_size)]
    return img

def fToSquareInt(img):
    """
    Repeats image pixels along smaller dimension to make it closer to a square. Useful for extremely thin images.
    """
    if isinstance(img, torch.Tensor): img = img.detach().clone()
    elif isinstance(img, np.ndarray): img = img.copy()
    if img.ndim == 3 and img.shape[0] < img.shape[2]: h, w = 1, 2
    else: h, w = 0, 1
    if img.shape[h] / img.shape[w] > 1:
        if isinstance(img, torch.Tensor): img = img.repeat_interleave(repeats = int(img.shape[h] / img.shape[w]), dim = w, )
        else: img = np.repeat(img, repeats = int(img.shape[h] / img.shape[w]), axis = w)
    elif img.shape[w] / img.shape[h] > 1:
        if isinstance(img, torch.Tensor): img = img.repeat_interleave(repeats = int(img.shape[w] / img.shape[h]), dim = h, )
        else: img = np.repeat(img, repeats = int(img.shape[w] / img.shape[h]), axis = h)
    return img

import cv2
def fResize(img, h, w): # TODO
    if isinstance(img, torch.Tensor):
        img = np.array(img.cpu().detach(), copy=False)
        was_t = True
    else: was_t = False
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = fToChannelsLast(img)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA) # type:ignore # pylint:disable=E1101
        return fToChannelsFirst(torch.as_tensor(img) if was_t else img)
    else:
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)# pylint:disable=E1101
        return torch.as_tensor(img) if was_t else img

def z_normalize(x:torch.Tensor | np.ndarray):
    if x.std() != 0: return (x - x.mean()) / x.std()
    return x - x.mean()

def z_normalize_channels(img:np.ndarray):
    if img.std() != 0: return (img - img.mean(axis = list(range(1, img.ndim)), keepdims = True)) / img.std(axis = list(range(1, img.ndim)), keepdims = True)
    return img - img.mean(axis = list(range(1, img.ndim)), keepdims = True)

def norm_to01(x:torch.Tensor | np.ndarray):
    x = x - x.min()
    if x.max() != 0:
        x = x / x.max()
    return x

def norm_to11(x:torch.Tensor | np.ndarray):
    """Norm to (-1, 1)"""
    x = x - x.min()
    if x.max() != 0:
        x = x / x.max()
    return x - 0.5

def norm_torange(x:torch.Tensor | np.ndarray, min=0, max=1): #pylint:disable=W0622
    x -= x.min()
    if x.max()!=0:
        x /= x.max()
    else: return x
    x *= max - min
    x += min
    return x