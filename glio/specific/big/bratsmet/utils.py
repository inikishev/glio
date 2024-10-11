
import SimpleITK as sitk
import numpy as np
import torch

Loadable = np.ndarray | sitk.Image | torch.Tensor | str

def tositk(x: Loadable) -> sitk.Image:
    """Load an image into an itk.Image object.
    `image` can be a numpy array, an sitk.Image, a torch.Tensor or a string (path to an image file)."""
    if isinstance(x ,np.ndarray): return sitk.GetImageFromArray(x)
    elif isinstance(x, sitk.Image): return x
    elif isinstance(x, str): return sitk.ReadImage(x)
    if isinstance(x, torch.Tensor): return sitk.GetImageFromArray(x.numpy())
    else: raise TypeError(f"Unsupported type {type(x)}")

def tonumpy(x: Loadable) -> np.ndarray:
    if isinstance(x ,np.ndarray): return x
    elif isinstance(x, sitk.Image): return sitk.GetArrayFromImage(x)
    elif isinstance(x, str): return sitk.GetArrayFromImage(sitk.ReadImage(x))
    if isinstance(x, torch.Tensor): return x.numpy()
    else: raise TypeError(f"Unsupported type {type(x)}")
    
def totensor(x: Loadable) -> torch.Tensor:
    if isinstance(x ,np.ndarray): return torch.from_numpy(x)
    elif isinstance(x, sitk.Image): return torch.from_numpy(sitk.GetArrayFromImage(x))
    elif isinstance(x, str): return  torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(x)))
    if isinstance(x, torch.Tensor): return x
    else: raise TypeError(f"Unsupported type {type(x)}")