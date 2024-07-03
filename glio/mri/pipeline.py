import logging, os
import torch
import numpy as np
import SimpleITK as sitk
from .dicom_to_nifti import dicom2sitk
from .registration import register_imgs_to_SRI24, register_with
from .skullstrip import skullstrip_imgs
from .normalize import znormalize_imgs
from .crop_bg import crop_bg_imgs


def _toImage(x) -> sitk.Image:
    if isinstance(x, sitk.Image): return x
    elif os.path.isfile(x): return sitk.ReadImage(x)
    elif os.path.isdir(x): return dicom2sitk(x)
    else: raise ValueError(f"x either dir file or sitk image but its {type(x) = }")

def pipeline(t1:str|sitk.Image, t1ce:str|sitk.Image, flair:str|sitk.Image, t2w:str|sitk.Image,
             register=True, skullstrip=True, erode=1, cropbg=True) -> tuple[list[sitk.Image],list[sitk.Image]]:
    t1_orig = _toImage(t1)
    t1ce_orig = _toImage(t1ce)
    flair_orig = _toImage(flair)
    t2w_orig = _toImage(t2w)

    logging.info("registering to SRI24")
    if register: t1_sri, t1ce_sri, flair_sri, t2w_sri = register_imgs_to_SRI24(t1_orig, (t1ce_orig, flair_orig, t2w_orig))
    else: t1_sri, t1ce_sri, flair_sri, t2w_sri = t1_orig, t1ce_orig, flair_orig, t2w_orig

    logging.info("skullstripping")
    if skullstrip: t1ce_skullstrip, t1_skullstrip, flair_skullstrip, t2w_skullstrip = skullstrip_imgs(t1ce_sri, (t1_sri, flair_sri, t2w_sri), erode=erode)
    else: t1ce_skullstrip, t1_skullstrip, flair_skullstrip, t2w_skullstrip = t1ce_sri, t1_sri, flair_sri, t2w_sri

    logging.info("normalization")
    norm = znormalize_imgs((t1_skullstrip,t1ce_skullstrip,flair_skullstrip, t2w_skullstrip))

    if cropbg: return [t1_sri, t1ce_sri, flair_sri, t2w_sri], crop_bg_imgs(norm)
    else: return [t1_sri, t1ce_sri, flair_sri, t2w_sri, t2w_skullstrip], norm


class Pipeline:
    def __init__(self, t1:str, t1ce:str, flair:str, t2w:str, register=True, skullstrip=True, erode=1, cropbg=True):
        self.register = register
        self.skullstrip = skullstrip
        self.erode = erode
        self.cropbg = cropbg

        self.t1_native, self.t1ce_native, self.flair_native, self.t2w_native = [_toImage(i) for i in (t1,t1ce,flair,t2w)]

    def preprocess(self) -> torch.Tensor:
        (self.t1_sri, self.t1ce_sri, self.flair_sri, self.t2w_sri), \
        (self.t1_final, self.t1ce_final, self.flair_final, self.t2w_final) = \
            pipeline(self.t1_native, self.t1ce_native, self.flair_native, self.t2w_native, self.register, self.skullstrip, self.erode, self.cropbg)

        return torch.from_numpy(np.stack(
            [sitk.GetArrayFromImage(i) for i in (self.t1_final, self.t1ce_final, self.flair_final, self.t2w_final)]
            ).astype(np.float32))

    def save(self, path, mkdirs=True):
        if mkdirs: os.makedirs(path, exist_ok=True)

        sitk.WriteImage(self.t1_native, os.path.join(path, 't1_native.nii.gz'))
        sitk.WriteImage(self.t1ce_native, os.path.join(path, 't1ce_native.nii.gz'))
        sitk.WriteImage(self.flair_native, os.path.join(path, 'flair_native.nii.gz'))
        sitk.WriteImage(self.t2w_native, os.path.join(path, 't2w_native.nii.gz'))

        sitk.WriteImage(self.t1_sri, os.path.join(path, 't1_sri.nii.gz'))
        sitk.WriteImage(self.t1ce_sri, os.path.join(path, 't1ce_sri.nii.gz'))
        sitk.WriteImage(self.flair_sri, os.path.join(path, 'flair_sri.nii.gz'))
        sitk.WriteImage(self.t2w_sri, os.path.join(path, 't2w_sri.nii.gz'))

        sitk.WriteImage(self.t1_final, os.path.join(path, 't1_final.nii.gz'))
        sitk.WriteImage(self.t1ce_final, os.path.join(path, 't1ce_final.nii.gz'))
        sitk.WriteImage(self.flair_final, os.path.join(path, 'flair_final.nii.gz'))
        sitk.WriteImage(self.t2w_final, os.path.join(path, 't2w_final.nii.gz'))

        if hasattr(self, 'seg'):
            sitk.WriteImage(self.seg, os.path.join(path, 'seg.nii.gz'))
            sitk.WriteImage(self.seg_native, os.path.join(path, 'seg_native.nii.gz'))

    def unregister_seg(self, seg:str|sitk.Image|np.ndarray|torch.Tensor, to = 't1') -> sitk.Image:
        if isinstance(seg, torch.Tensor): seg = seg.detach().cpu().numpy()
        if isinstance(seg, np.ndarray): seg = sitk.GetImageFromArray(seg)
        else: seg = _toImage(seg)

        self.seg = seg

        to = to.lower()

        if to == 't1': _, self.seg_native = register_with(self.t1_sri, seg, self.t1_native)
        elif to == 't1ce': _, self.seg_native = register_with(self.t1ce_sri, seg, self.t1ce_native)
        elif to == 'flair': _, self.seg_native = register_with(self.flair_sri, seg, self.flair_native)
        elif to == 't2w': _, self.seg_native = register_with(self.t2w_sri, seg, self.t2w_native)

        return self.seg_native