# Автор - Никишев Иван Олегович группа 224-31

import logging
import os
import numpy as np
import torch
import torchvision.transforms.v2
import pydicom


def dcmread(path) -> np.ndarray:
    """Считывает DICOM файл и преобразует в тензор."""
    return pydicom.dcmread(path).pixel_array

def dcmreadtensor(path, dtype=None) -> torch.Tensor:
    arr = dcmread(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)


def dcmread_sorted_paths(paths) -> np.ndarray:
    """Считывает файлы DICOM и создаёт трёхмерный тензор. Пути к файлам считаются отсортированными по координате среза"""
    return np.array([pydicom.dcmread(i).pixel_array for i in paths], copy=False)

def dcmreadtensor_sorted_paths(paths, dtype=None) -> torch.Tensor:
    arr = dcmread_sorted_paths(paths)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)


def dcmread_paths(paths) -> np.ndarray:
    """Считывает файлы DICOM и создаёт трёхмерный тензор, в котором координаты каждого среза соответствуют IOD `InstanceNumber`"""
    images = sorted([pydicom.dcmread(i) for i in paths], key = lambda x: x.InstanceNumber)# pyright:ignore[reportPossiblyUnboundVariable]
    return np.array([i.pixel_array for i in images], copy=False)

def dcmreadtensor_paths(paths, dtype=None) -> torch.Tensor:
    arr = dcmread_paths(paths)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)


def dcmread_folder(path) -> np.ndarray:
    """Считывает файлы DICOM из папки и создаёт трёхмерный тензор, в котором координаты каждого среза соответствуют IOD `InstanceNumber`"""
    paths = [os.path.join(path, i) for i in os.listdir(path)]
    return dcmread_paths(paths)

def dcmreadtensor_folder(path, dtype=None) -> torch.Tensor:
    """Считывает файлы DICOM из папки и создаёт трёхмерный тензор, в котором координаты каждого среза соответствуют IOD `InstanceNumber`"""
    arr = dcmread_folder(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)

def affine2d(ds:pydicom.Dataset):
    F11, F21, F31 = ds.ImageOrientationPatient[3:]
    F12, F22, F32 = ds.ImageOrientationPatient[:3]

    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, 0, Sz],
            [0, 0, 0, 1]
        ]
    )

def dcmread_affine(path) -> np.ndarray:
    """Считывает файлы DICOM и создаёт трёхмерный тензор, в котором координаты каждого среза соответствуют IOD `InstanceNumber`"""
    ds = pydicom.dcmread(path)
    return affine2d(ds)