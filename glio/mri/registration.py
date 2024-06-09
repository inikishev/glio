from collections.abc import Sequence
import SimpleITK as sitk

MNI152 = r"F:\Stuff\Programming\AI\glio_diff\glio\mri\mni_icbm152_t1_tal_nlin_asym_09a.nii"
SRI24 = r"F:\Stuff\Programming\AI\glio_diff\glio\mri\SRI24.nii"


def resample_to(input:str | sitk.Image, reference: str | sitk.Image, interpolation=sitk.sitkNearestNeighbor) -> sitk.Image:
    """Resample `input` to `reference`, both can be either a `sitk.Image` or a path to a nifti file that will be loaded.

    Resampling uses spatial information embedded in nifti file / sitk.Image - size, origin, spacing and direction.

    `input` is transformed in such a way that those attributes will match `reference`.
    That doesn't guarantee perfect allginment.
    """
    # load inputs
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(reference, str): reference = sitk.ReadImage(reference)

    return sitk.Resample(
            input,
            reference,
            sitk.Transform(),
            interpolation
        )

def default_pmap():
    pmap = sitk.VectorOfParameterMap()
    pmap.append(sitk.GetDefaultParameterMap("translation"))
    pmap.append(sitk.GetDefaultParameterMap("rigid"))
    pmap.append(sitk.GetDefaultParameterMap("affine"))
    return pmap

def register_to(input:str | sitk.Image, reference: str | sitk.Image, pmap = default_pmap()) -> sitk.Image:
    """Register `input` to `reference`, both can be either a `sitk.Image` or a path to a nifti file that will be loaded. Returns `input` registered to `reference`.

    Registering means input image is transformed using affine transforms or some deformations
    (whatever elastix is using) to match the reference, it will have the same size, orientation, etc, and the should be perfectly alligned."""
    # load inputs
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(reference, str): reference = sitk.ReadImage(reference)

    # create elastix filter
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(reference)
    elastix.SetMovingImage(input)

    # set it to elastix filter and execute
    if pmap is not None: elastix.SetParameterMap(pmap)
    elastix.Execute()
    return elastix.GetResultImage()


def register_with(input:str | sitk.Image, other: str | sitk.Image, reference: str | sitk.Image, pmap = default_pmap(), label=True) -> tuple[sitk.Image,sitk.Image]:
    """Register `input` to reference, then use that transformation to also register `other`, which is usually segmentation."""
    # load inputs
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(other, str): other = sitk.ReadImage(other)
    if isinstance(reference, str): reference = sitk.ReadImage(reference)

    # create elastix filter
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(reference)
    elastix.SetMovingImage(input)

    # set it to elastix filter and execute
    if pmap is not None: elastix.SetParameterMap(pmap)
    input_reg = elastix.Execute()

    transform = sitk.TransformixImageFilter()
    tmap = elastix.GetTransformParameterMap()
    if label: 
        tmap[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        tmap[1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        tmap[2]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transform.SetTransformParameterMap(tmap)
    transform.SetMovingImage(other)
 
    return input_reg, transform.Execute()



def register_t1_to_MNI152(input:str|sitk.Image, reference=MNI152) -> sitk.Image:
    """Register `input` to MNI. `input` must be path/sitk.Image of a T1 scan. Returns `input` registered to MNI."""
    return register_to(input, reference)

def register_t1_to_SRI24(input:str|sitk.Image, reference=SRI24) -> sitk.Image:
    """Register `input` to SRI24. `input` must be path/sitk.Image of a T1 scan. Returns `input` registered to SRI24."""
    return register_to(input, reference)

def register_imgs_to(template_input: str | sitk.Image, other_inputs: str|sitk.Image | Sequence[str|sitk.Image], reference: str | sitk.Image) -> list[sitk.Image]:
    """Register `template_input` to `reference`, then register `other_inputs` to registered `template_input`.

    Returns registered `[template_input, *other_inputs]`."""
    if not isinstance(other_inputs, Sequence): other_inputs = [other_inputs]

    # Register `template_input` to `reference`
    registered_template_input = register_to(template_input, reference)

    # register `other_inputs` to registered `template_input`.
    registered_other_inputs = [register_to(i, registered_template_input) for i in other_inputs]

    return [registered_template_input, *registered_other_inputs]

def register_imgs_to_MNI152(t1:str|sitk.Image, other: str|sitk.Image | Sequence[str|sitk.Image], reference=MNI152) -> list[sitk.Image]:
    return register_imgs_to(t1, other, reference)

def register_imgs_to_SRI24(t1:str|sitk.Image, other: str|sitk.Image | Sequence[str|sitk.Image], reference=SRI24) -> list[sitk.Image]:
    return register_imgs_to(t1, other, reference)