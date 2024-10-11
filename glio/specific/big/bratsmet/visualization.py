import SimpleITK as sitk

from ....jupyter_tools import show_slices, show_slices_arr
from .utils import tositk, tonumpy, totensor, Loadable

def auto_show_slices(img: Loadable):
    show_slices_arr(tonumpy(img))

def auto_show_slices_arr(img: Loadable):
    show_slices_arr(tonumpy(img))