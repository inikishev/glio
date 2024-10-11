import random

import torch
from torchzero.nn.functional.pad import pad_to_shape
from torchzero.nn.functional.crop import crop_to_shape

from ....python_tools import Compose
from ....torch_tools import center_of_mass


# THIS IS FOR BINARY SEGMENTATION ONLY!
def randcrop(x: tuple[torch.Tensor, torch.Tensor], size = (96,96)):
    """Randomly crop `x` to `size`. X needs to be tuple of `(image, seg)`. `image` needs to be CHW, `seg` needs to be BINARY HW."""
    if x[0].shape[1] == size[0] and x[0].shape[2] == size[1]: return x

    if x[0].shape[1] <= size[0] or x[0].shape[2] <= size[1]:
        return (crop_to_shape(pad_to_shape(x[0], (x[0].shape[0], *size), where='center', mode = 'min'), (x[0].shape[0], *size)).to(torch.float32),
                crop_to_shape(pad_to_shape(x[1], size, where='center', value = 0), size).to(torch.float32))

    startx = random.randint(0, (x[0].shape[1] - size[0]) - 1)
    starty = random.randint(0, (x[0].shape[2] - size[1]) - 1)
    return (x[0][:, startx:startx+size[0], starty:starty+size[1]].to(torch.float32),
            x[1][startx:startx+size[0], starty:starty+size[1]].to(torch.float32), )

def com_randcrop(x: tuple[torch.Tensor, torch.Tensor], size = (96,96)):
    """Randomly crop `x` to `size`, with random distribution centered on center of mass of segmentation. X needs to be tuple of `(image, seg)`. `image` needs to be CHW, `seg` needs to be BINARY HW."""
    if x[0].shape[1] == size[0] and x[0].shape[2] == size[1]: return x

    if x[1].sum() == 0:
        return randcrop(x, size)

    if x[0].shape[1] <= size[0] or x[0].shape[2] <= size[1]:
        return (crop_to_shape(pad_to_shape(x[0], (x[0].shape[0], *size), where='center', mode = 'min'), (x[0].shape[0], *size)).to(torch.float32),
                crop_to_shape(pad_to_shape(x[1], size, where='center', value = 0), size).to(torch.float32))

    com = center_of_mass(x[1])
    xmax = (x[0].shape[1] - size[0]) - 1
    ymax = (x[0].shape[2] - size[1]) - 1
    if xmax < com[0]: com[0] = xmax
    if ymax < com[1]: com[1] = ymax

    startx = int(random.triangular(0, xmax, float(com[0])))
    starty = int(random.triangular(0, ymax, float(com[1])))

    starty = int(starty)

    return (x[0][:, startx:startx+size[0], starty:starty+size[1]].to(torch.float32),
            x[1][startx:startx+size[0], starty:starty+size[1]].to(torch.float32), )
