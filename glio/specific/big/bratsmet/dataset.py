from collections.abc import Callable

import joblib
import SimpleITK as sitk
import torch

from ....python_tools import (find_file_containing, listdir_fullpaths,
                              reduce_dim)
from ....torch_tools import MRISlicer

DATASET_ROOT = r'E:\dataset\BraTS2024-MEN-RT-TrainingData'
DATASET_FILES = rf'{DATASET_ROOT}\BraTS-MEN-RT-Train-v2'
RAW_0_400 = rf"{DATASET_ROOT}\raw 0-400.joblib"
RAW_400_500 = rf"{DATASET_ROOT}\raw 400-500.joblib"

CASE_DIRS = listdir_fullpaths(DATASET_FILES)
"""Each directory has t1c and segmentation."""

def get_t1c_seg_fpaths_by_idx(idx):
    """Returns two file paths."""
    folder = CASE_DIRS[idx]
    t1c = find_file_containing(folder, 't1c.')
    seg = find_file_containing(folder, 'gtv.')
    return t1c, seg

def load_t1c_sitk_by_idx(idx):
    """Return T1c SimpleITK Image"""
    folder = CASE_DIRS[idx]
    t1c = find_file_containing(folder, 't1c.')
    return sitk.ReadImage(t1c)

def load_seg_sitk_by_idx(idx):
    """Return segmentataion SimpleITK Image"""
    folder = CASE_DIRS[idx]
    seg = find_file_containing(folder, 'gtv.')
    return sitk.ReadImage(seg)

def load_t1c_seg_sitk_by_idx(idx):
    """Return a tuple of T1c and segmentation SimpleITK Images"""
    folder = CASE_DIRS[idx]
    t1c = find_file_containing(folder, 't1c.')
    seg = find_file_containing(folder, 'gtv.')
    return sitk.ReadImage(t1c), sitk.ReadImage(seg)


def get_ds_allsegslices(path, around=1, any_prob = 0.1) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a study that contain segmentation + `any_prob` * 100 % objects that return a random slice."""
    MRIs:list[MRISlicer] = joblib.load(path)
    for i in MRIs: i.set_settings(around = around, any_prob = any_prob)
    ds = reduce_dim([i.get_all_seg_slice_callables() for i in MRIs])
    random_slices = reduce_dim([i.get_anyp_random_slice_callables() for i in MRIs])
    ds.extend(random_slices)
    return ds

def get_ds_allslices(path, around=1) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a study"""
    MRIs:list[MRISlicer] = joblib.load(path)
    for i in MRIs: i.set_settings(around = around, any_prob=1)
    ds = reduce_dim([i.get_all_slice_callables() for i in MRIs])
    return ds