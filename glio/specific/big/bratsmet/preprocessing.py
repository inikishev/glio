from collections.abc import Mapping
from typing import Any, overload

import monai.transforms as mtf
import numpy as np
import SimpleITK as sitk


def crop_bg(image: sitk.Image) -> sitk.Image:
    image = sitk.RescaleIntensity(image, 0, 255)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(sitk.OtsuThreshold(image, 0, 255))
    bbox = filt.GetBoundingBox(255)
    return sitk.RegionOfInterest( image, bbox[int(len(bbox) / 2) :],  bbox[0 : int(len(bbox) / 2)],)


def crop_bg_D[T: Mapping[Any, sitk.Image]](images: T, key: Any) -> T: # type:ignore #pylint:disable = E0602
    """Finds the bounding box of `images[key]` and crops all images in `images` to that bounding box."""
    image = images[key]
    image = sitk.RescaleIntensity(image, 0, 255)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(sitk.OtsuThreshold(image, 0, 255))
    bbox = filt.GetBoundingBox(255)
    res = {k: sitk.RegionOfInterest(v, bbox[int(len(bbox) / 2) :],  bbox[0 : int(len(bbox) / 2)]) for k,v in images.items()}
    return type(images)(**res)


# normal znormalize
# image = sitk.Normalize(image)

def foreground_znormalize(image: sitk.Image) -> sitk.Image:
    """Performs z-normalization but only considers foreground values."""
    # get C* array
    arr = sitk.GetArrayFromImage(image)[np.newaxis, ...]
    # normalize to 0-1
    arr = arr - arr.min()
    arr:np.ndarray = arr / arr.max()

    # find foreground
    tfm = mtf.ForegroundMask() # type:ignore
    foreground_mask = np.asarray(tfm(arr)).astype(bool)

    # get only foreground from array
    mean = arr[foreground_mask].mean()
    std = arr[foreground_mask].std()

    arr = (arr - mean) / std

    zimage = sitk.GetImageFromArray(arr[0])
    zimage.CopyInformation(image)
    return zimage