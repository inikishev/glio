from typing import Any
from collections.abc import Sequence
import os, subprocess, shutil
import tempfile
import SimpleITK as sitk

def get_brain_mask(inputs:str | sitk.Image | Sequence[str | sitk.Image], mni = False) -> Sequence[sitk.Image]:
    """Runs skullstripping using HD BET (https://github.com/MIC-DKFZ/HD-BET). Requires it to be installed.

    Returns `outpath` which is path to brain mask for convenience.

    Args:
        inputs (str | sitk.Image): Path to a nifti file or a sitk.Image of the image to generate brain mask from, all inputs must be in MNI152 space.
        outpaths (str): Path to output file, must include `.nii.gz`.
        mkdirs (bool, optional): Whether to create `outfolder` if it doesn't exist, otherwise throws an error. Defaults to True.
    """
    from HD_BET.run import run_hd_bet
    if not isinstance(inputs, Sequence): inputs = [inputs]
    
    # register to MNI152 space
    if mni:
        from ..mri.registration import register_t1_to_MNI152, register_imgs_to_MNI152
        if len(inputs) == 1: inputs = [register_t1_to_MNI152(inputs[0])]
        else: inputs = register_imgs_to_MNI152(inputs[0], inputs[1:])

    with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as temp2:

        for i,input in enumerate(list(inputs)):
            if isinstance(input, sitk.Image): sitk.WriteImage(input, os.path.join(temp, f'{i}.nii.gz'))
            else: shutil.copyfile(input, os.path.join(temp, f'{i}.nii.gz'))

        # run skullstripping
        run_hd_bet([os.path.join(temp, i) for i in os.listdir(temp)], [os.path.join(temp2, i) for i in os.listdir(temp)])


        return [sitk.ReadImage(os.path.join(temp2, i)) for i in sorted(os.listdir(temp2))]
