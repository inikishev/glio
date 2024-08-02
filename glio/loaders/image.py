"img oladers "

import logging
import os
import torch
import torchvision.transforms.v2


import numpy as np


import torchvision.io
import matplotlib.pyplot as plt
import skimage.io
# import cv2
import PIL.Image

from ..transforms.intensity import norm

def imreadtensor_torchvision(path:str) -> torch.Tensor:
    return torchvision.io.read_image(path)

def imread_skimage(path:str) -> np.ndarray:
    return skimage.io.imread(path)

def imreadtensor_skimage(path:str) -> torch.Tensor:
    return torch.as_tensor(imread_skimage(path))

def imread_plt(path:str) -> np.ndarray:
    return plt.imread(path)

def imreadtensor_plt(path:str) -> torch.Tensor:
    return torch.as_tensor(imread_plt(path))

# def imread_cv2(path:str, bgr2rgb=True) -> np.ndarray:
#     image = cv2.imread(path)
#     if bgr2rgb and image.ndim == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image

# def imreadtensor_cv2(path:str, bgr2rgb=True) -> torch.Tensor:
#     return torch.as_tensor(imread_cv2(path, bgr2rgb))

def imread_pil(path:str) -> np.ndarray:
    return np.array(PIL.Image.open(path))

def imreadtensor_pil(path:str) -> torch.Tensor:
    return torch.as_tensor(imread_pil(path))

def imread(path:str) -> np.ndarray:
    try: return imread_plt(path)
    except Exception:
        try: return imread_skimage(path)
        except Exception:
            # try: return imread_cv2(path)
            # except Exception:
                return imread_pil(path)

def imreadtensor(path:str):
    if path.lower().endswith(('jpg', 'jpeg', 'png', 'gif')): return imreadtensor_torchvision(path)
    else: return torch.as_tensor(imread(path))

def imwrite(x:np.ndarray | torch.Tensor, outfile:str, mkdir=False, normalize=True, compression = 9, optimize=True):
    while x.ndim not in (2, 3): 
        if x.shape[0] > 1: raise ValueError('x must be 2d or 3d')
        x = x[0]
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if normalize: x = norm(x, 0, 255).astype(np.uint8) # type:ignore
    if x.ndim == 3 and x.shape[0] < x.shape[2]: x = np.transpose(x, (1, 2, 0))
    if mkdir and not os.path.exists(os.path.dirname(outfile)): os.mkdir(os.path.dirname(outfile))
    PIL.Image.fromarray(x).save(outfile, optimize=optimize, compress_level=compression) # type:ignore
