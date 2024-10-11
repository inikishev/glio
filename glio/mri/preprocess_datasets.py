from typing import Optional
import SimpleITK as sitk
import numpy as np
from .crop_bg import crop_bg_imgs
from .normalize import znormalize_imgs, znormalize
from ..python_tools import find_file_containing

def preprocess_images_seg(t1:str, t1ce:str, flair:str, t2w:str, seg:str, cropbg=True):
    if cropbg: t1ce_crop, t1_crop, flair_crop, t2w_crop, seg_crop = crop_bg_imgs([t1ce, t1, flair, t2w, seg])
    else: t1ce_crop, t1_crop, flair_crop, t2w_crop, seg_crop = t1ce, t1, flair, t2w, sitk.ReadImage(seg)
    t1_norm, t1ce_norm, flair_norm, t2w_norm = znormalize_imgs([t1_crop, t1ce_crop, flair_crop, t2w_crop])
    return (np.stack([sitk.GetArrayFromImage(t1_norm),
                     sitk.GetArrayFromImage(t1ce_norm),
                     sitk.GetArrayFromImage(flair_norm),
                     sitk.GetArrayFromImage(t2w_norm)]), sitk.GetArrayFromImage(seg_crop))

def preprocess_images(t1:str, t1ce:str, flair:str, t2w:str):
    t1ce_crop, t1_crop, flair_crop, t2w_crop = crop_bg_imgs([t1ce, t1, flair, t2w])
    t1_norm, t1ce_norm, flair_norm, t2w_norm = znormalize_imgs([t1_crop, t1ce_crop, flair_crop, t2w_crop])
    return np.stack([sitk.GetArrayFromImage(t1_norm),
                     sitk.GetArrayFromImage(t1ce_norm),
                     sitk.GetArrayFromImage(flair_norm),
                     sitk.GetArrayFromImage(t2w_norm)])

def preprocess_images_seg_tensor(t1:str, t1ce:str, flair:str, t2w:str, seg:str):
    import torch
    images, segm = preprocess_images_seg(t1, t1ce, flair, t2w, seg)
    return torch.from_numpy(images).to(torch.float32), torch.from_numpy(segm.astype(np.int32))

def preprocess_images_tensor(t1:str, t1ce:str, flair:str, t2w:str):
    import torch
    images = preprocess_images(t1, t1ce, flair, t2w)
    return torch.from_numpy(images).to(torch.float32)

def preprocess_rhuh(path:str):
    """Crops black background and applies znormalization to each modality.

    Returns: ndarray[t1, t1ce, flair, t2w], ndarray[seg]."""
    t1 = find_file_containing(path, "t1.")
    t1ce = find_file_containing(path, "t1ce.")
    flair = find_file_containing(path, "flair.")
    t2w = find_file_containing(path, "t2.")
    seg = find_file_containing(path, "segmentations.")

    return preprocess_images_seg(t1, t1ce, flair, t2w, seg)

def preprocess_rhuh_tensor(path:str):
    """Crops black background and applies znormalization to each modality.

    Returns: torch.Tensor[t1, t1ce, flair, t2w], torch.Tensor[seg]."""
    import torch
    images, seg = preprocess_rhuh(path)
    return torch.from_numpy(images).to(torch.float32), torch.from_numpy(seg.astype(np.int32))

def preprocess_brats2024goat(path:str, cropbg=True):
    """Crops black background and applies znormalization to each modality.

    Returns: ndarray[t1, t1ce, flair, t2w], ndarray[seg]."""
    t1 = find_file_containing(path, "t1n.")
    t1ce = find_file_containing(path, "t1c.")
    flair = find_file_containing(path, "t2f.")
    t2w = find_file_containing(path, "t2w.")
    seg = find_file_containing(path, "seg.")
    return preprocess_images_seg(t1, t1ce, flair, t2w, seg, cropbg=cropbg)

def preprocess_brats2024goat_tensor(path:str, cropbg=True):
    """Crops black background and applies znormalization to each modality.

    Returns: torch.Tensor[t1, t1ce, flair, t2w], torch.Tensor[seg]."""
    import torch
    images, seg = preprocess_brats2024goat(path, cropbg=cropbg)
    return torch.from_numpy(images).to(torch.float32), torch.from_numpy(seg.astype(np.int32))


def preprocess_brats2024gli(path:str, cropbg=True):
    """Crops black background and applies znormalization to each modality.

    Returns: ndarray[t1, t1ce, flair, t2w], ndarray[seg]."""
    t1 = find_file_containing(path, "t1n.")
    t1ce = find_file_containing(path, "t1c.")
    flair = find_file_containing(path, "t2f.")
    t2w = find_file_containing(path, "t2w.")
    seg = find_file_containing(path, "seg.")
    return preprocess_images_seg(t1, t1ce, flair, t2w, seg, cropbg=cropbg)

def preprocess_brats2024gli_tensor(path:str, cropbg=True):
    """Crops black background and applies znormalization to each modality.

    Returns: torch.Tensor[t1, t1ce, flair, t2w], torch.Tensor[seg]."""
    import torch
    images, seg = preprocess_brats2024goat(path, cropbg=cropbg)
    return torch.from_numpy(images).to(torch.float32), torch.from_numpy(seg.astype(np.int32))

def preprocess_brats2024met(path:str, cropbg=True):
    """Crops black background and applies znormalization the image.

    Returns: ndarray[t1ce], ndarray[seg]."""
    t1ce = find_file_containing(path, "t1c.")
    seg = find_file_containing(path, 'gtv.')
    if cropbg: t1ce_crop, seg_crop = crop_bg_imgs([t1ce, seg])
    else: t1ce_crop, seg_crop = t1ce, sitk.ReadImage(seg)
    t1ce_norm = znormalize(t1ce_crop)

    return (sitk.GetArrayFromImage(t1ce_norm)[np.newaxis, :, :], np.where(sitk.GetArrayFromImage(seg_crop) == 0, False, True))

def preprocess_brats2024met_tensor(path:str, cropbg=True):
    """Crops black background and applies znormalization to each modality.

    Returns: torch.Tensor[t1ce], torch.Tensor[seg]."""
    import torch
    images, seg = preprocess_brats2024met(path, cropbg=cropbg)
    return torch.from_numpy(images).to(torch.float32), torch.where(torch.from_numpy(seg) == 0, False, True)