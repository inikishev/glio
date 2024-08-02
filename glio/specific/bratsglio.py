from typing import Literal
from collections.abc import Sequence, Callable
from functools import partial
import os
import torch, numpy as np
import SimpleITK as sitk
from monai.inferers import SlidingWindowInferer # type:ignore
from monai.losses import DiceFocalLoss # type:ignore
from torch.utils.data import DataLoader
from ..plot import *
from ..train import *
from ..data import *
from ..datasets.mri_preloaded2 import *
from ..torch_tools import one_hot_mask, raw_preds_to_one_hot
from ..python_tools import get0, get1, perf_counter_context, find_file_containing
from ..jupyter_tools import clean_mem
from ..transforms.intensity import norm
from ..mri.preprocess_datasets import preprocess_brats2024gli_tensor, znormalize_imgs, crop_bg_imgs
from ..transforms.intensity import RandScale, RandShift
from ..transforms.spatial import RandFlipt, RandRot90t


__all__ = [
    'visualize_predictions',
    'get_brats_gli_preprocessed_by_idx',
    'visualize_brats_reference',
    'visualize_brats_reference_all',
    'SaveReferenceVisualizationsAfterEachEpochCB',
    "RandScale", "RandShift", "RandFlipt", "RandRot90t",
    "get_ds",
    "ds_performance",
    "run_train",
    "continue_train",
    "get_pretrained_model",
    "dim_inference_raw",
    "inference_raw_tta_separate",
    "inference_raw_tta_mean",
    "inference_argmax_tta",
    "inference_onehot_tta",
    "inference_argmax_tta_mode",
    "inference_softmax_argmax_tta",
    "predict_cropped",
    "predict_full",
    "predict_dataset",
    "get_mri_from_folder",
    "get_pretrained_model2",

]

# All the imaging datasets have been annotated manually, by one to four raters,
# following the same annotation protocol, and their annotations were approved by experienced neuroradiologists.
# Annotations comprise the enhancing tissue (ET — label 3),
# the surrounding non-enhancing FLAIR hyperintensity (SNFH) — label 2),
# the non-enhancing tumor core (NETC — label 1),
# and the resection cavity (RC - label 4)
# as described in the latest BraTS summarizing paper,
# except that the resection cavity has been incorporated subsequent to the paper's release.

COLORS = ('red', 'green', 'blue', 'yellow') # core, flair hyperintensity, enchancing, resection
COLOR_LEGEND = "red: NETC (non-enhancing tumor core);\ngreen: SNFH (surrounding non-enhancing FLAIR hyperintensity);\nblue: ET (enhancing tissue);\nyellow: RC (resection cavity)"

def visualize_predictions(inferer, sample:tuple[torch.Tensor, torch.Tensor], around=1, expand_channels = None, save=False, path=None, title=None):
    """Sample is (CHW, CHW) tuple"""
    fig = Figure()
    inputs:torch.Tensor = sample[0].unsqueeze(0)
    targets_raw:torch.Tensor = sample[1]
    targets:torch.Tensor = targets_raw.argmax(0)
    if expand_channels is None: preds_raw:torch.Tensor = inferer(inputs)[0]
    else:
        expanded = torch.cat((inputs, torch.zeros((1, expand_channels - inputs.shape[1], *inputs.shape[2:]))), dim=1)
        preds_raw:torch.Tensor = inferer(expanded)[0]
    preds:torch.Tensor = preds_raw.argmax(0)

    inputs[0] = torch.stack([norm(i) for i in inputs[0]]) # type:ignore
    fig.add().imshow_batch(inputs[0, 1::around*2+1], scale_each=True).style_img('inputs:\nT1, T1ce, FLAIR, T2w')
    fig.add().imshow_batch(preds_raw, scale_each=True).style_img('raw outputs')
    fig.add().imshow(torch.zeros_like(targets)).seg_overlay(targets, colors=COLORS, alpha=1.).style_img('targets')
    fig.add().imshow(torch.zeros_like(preds)).seg_overlay(preds, colors=COLORS, alpha=1.).style_img('prediction')
    fig.add().imshow(preds_raw[1:-1]).style_img('raw outputs RGB')
    #fig.add().imshow_batch((preds_raw - targets_raw).abs(), scale_each=True).style_img('raw error')
    fig.add().imshow(preds != targets,  cmap='gray').style_img('error')

    fig.add().imshow(inputs[0][0]).seg_overlay(targets, colors=COLORS).style_img('targets')
    fig.add().imshow(inputs[0][0]).seg_overlay(preds, colors=COLORS).style_img('predictions')

    if title is None: title = ''
    fig.create(2, figsize=(16,12), title=f'{title}\n{COLOR_LEGEND}')
    if save:
        fig.savefig(path)
        fig.close()

def get_brats_gli_preprocessed_by_idx(index, cropbg=True):
    """Returns a 5*h*w tensor: `t1c, t1n, t2f, t2w, seg`"""
    dspath = 'E:/dataset/BraTS-GLI v2/train'
    path = os.path.join(dspath, os.listdir(dspath)[index])
    images, seg = preprocess_brats2024gli_tensor(path, cropbg=cropbg)
    return torch.cat((images, seg.unsqueeze(0)))


brats_gli_references = (
    (partial(get_brats_gli_preprocessed_by_idx, 1349), 74),
    (partial(get_brats_gli_preprocessed_by_idx, 1349), 93),
    (partial(get_brats_gli_preprocessed_by_idx, 1348), 74),
    (partial(get_brats_gli_preprocessed_by_idx, 1347), 81),
    (partial(get_brats_gli_preprocessed_by_idx, 1346), 70),
    (partial(get_brats_gli_preprocessed_by_idx, 1344), 70),
    (partial(get_brats_gli_preprocessed_by_idx, 1342), 99),
    (partial(get_brats_gli_preprocessed_by_idx, 1340), 47),
)

def visualize_brats_reference(idx, inferer, around=1, overlap=0.75, expand=None, save=False, folder='reference preds', prefix='', mkdir = True):
    """0,1,2,3. Model must accept sequential around inputs, i.e. 3 T1c, 3 T1n, etc..."""
    slice_idx = brats_gli_references[idx][1]
    image3d = brats_gli_references[idx][0]()#.permute(0,3,2,1) # C D H W
    inputs3d, seg3d = image3d[:-1], image3d[-1]
    if around: inputs_around = inputs3d[:, slice_idx-around:slice_idx+around+1].flatten(0,1)
    else: inputs_around = inputs3d[:, slice_idx]
    seg = seg3d[slice_idx]

    if expand: inputs_around = torch.cat((inputs_around, torch.zeros((expand - inputs_around.shape[0], *inputs_around.shape[1:]))), dim=0)
    sliding = SlidingWindowInferer((96,96), 32, overlap=overlap, mode='gaussian')
    if mkdir and not os.path.exists(folder): os.mkdir(folder)
    visualize_predictions(
        partial(sliding.__call__, network=inferer),
        (inputs_around, one_hot_mask(seg, 5)),
        save=save,
        path=f"{folder}/{prefix}BRATS-GLI {idx}.jpg",
        title=f'BRATS-GLI {idx}'
    )

def visualize_brats_reference_all(inferer, around=1, overlap=0.75, expand=None, save=False, folder='reference preds', prefix=''):
    for i in range(len(brats_gli_references)):
        visualize_brats_reference(i, inferer=inferer, around=around, overlap=overlap, expand=expand, save=save, folder=folder,prefix=prefix)

class SaveReferenceVisualizationsAfterEachEpochCB(MethodCallback):
    order = 1
    def __init__(self, folder='runs', brats=tuple(range(len(brats_gli_references))), around=1):
        self.folder=folder
        if not os.path.exists(folder): os.mkdir(folder)
        self.brats = brats
        self.around = around

    def after_test_epoch(self, learner:Learner):
        folder = os.path.join(learner.get_workdir(self.folder), 'reference preds')
        if not os.path.exists(folder): os.mkdir(folder)
        prefix=f'{learner.total_epoch} {float(learner.logger.last("test loss")):.4f} '
        for i in self.brats: visualize_brats_reference(i, learner.inference, save=True, folder=folder, prefix=prefix, around=self.around)



def get_ds(
    loader_train = None,
    loader_test = None,
    tfm_init_train = (randcrop, RandRot90t(p=2/3), RandFlipt(p=2/3)),
    tfm_init_test = randcrop,
    tfm_input_train = (get0, RandScale(p=0.5), RandShift(p=0.5)),
    tfm_input_test = get0,
    tfm_target_train = get1,
    tfm_target_test = get1,
    traindir=BRATSGLI_0_1000,
    testdir: Optional[str]=BRATSGLI_1000_1350,
    around=1,
    any_prob=0.1,
    test_eq_train = False,
    nelem = None,
    ):
    bratsglitrain = get_ds_allsegslices(traindir, around=around, any_prob=any_prob)
    if test_eq_train: bratsglitest = bratsglitrain
    else: bratsglitest = get_ds_allslices(testdir, around=around)

    if nelem is not None: bratsglitrain, bratsglitest = bratsglitrain[:nelem], bratsglitest[:nelem]

    dstrain = DSToTarget(0)
    dstest = DSToTarget(0)

    dstrain.add_samples(bratsglitrain, loader = loader_train, transform_init=tfm_init_train, transform_sample=tfm_input_train, transform_target=tfm_target_train)
    dstest.add_samples(bratsglitest, loader = loader_test, transform_init=tfm_init_test, transform_sample=tfm_input_test, transform_target=tfm_target_test)
    return dstrain, dstest

def ds_performance(ds, num = 100):
    with perf_counter_context():
        for i in range(50):
            ds[i] # pylint:disable = W0104

PBAR_METRICS = (
    'train loss', 'test loss',
    'train dice - NETC', 'train dice - SNFH', 'train dice - ET', 'train dice - RC',
    'test dice - NETC', 'test dice - SNFH', 'test dice - ET', 'test dice - RC',
    "test dice mean", "test softdice mean",
)
SMOOTH = (None, None, 16, 16, 16, 16, None, None, None, None, None, None)
def run_train(
    title:str,
    dltrain,
    dltest,
    model,
    opt,
    sched,
    ref_sample:tuple[torch.Tensor, torch.Tensor],
    lossfn = DiceFocalLoss(softmax=True),
    n_epochs = 5,
    extra_cbs = (),
    test_first = True,
    around = 1,
    ):

    clean_mem()
    #MODEL = gnn.LSUV(MODEL, dltrain, max_iter=4)

    cbs = [
        LogLossCB(),
        LogTimeCB(),
        LogLRCB(),
        MONAIConfusionMatrixMetricsCB(['NETC', 'SNFH', 'ET', 'RC'], include_bg=False),
        TorchzeroDiceCB(['BG', 'NETC', 'SNFH', 'ET', 'RC'], step=8),
        TorchzeroSoftdiceCB(['BG', 'NETC', 'SNFH', 'ET', 'RC'], step=8),
        TorchzeroIoUCB(['BG', 'NETC', 'SNFH', 'ET', 'RC'], step=8),
        PerformanceTweaksCB(True),
        AccelerateCB("no"),
        SaveBestCB(),
        SaveLastCB(),
        FastProgressBarCB(step_batch=128, plot=True, metrics=PBAR_METRICS, maxv=1, smooth = SMOOTH),
        DisplayLoggerTableCB(),
        SaveReferenceVisualizationsAfterEachEpochCB(around=around),
        SaveForwardChannelImagesCB(ref_sample[0].unsqueeze(0)),
        SaveBackwardChannelImagesCB(*ref_sample, unsqueeze=True),
        SaveUpdateChannelImagesCB(*ref_sample, unsqueeze=True),
        #Metric_PredsTargetsFn(LOSS_FN, name='dice loss'),
        #CallTrainAndEvalOnOptimizer(),
        #AddLossReturnedByModelToLossInBackward(),
    ]

    learner = Learner(
        model, f"{title}",
        cbs = cbs + list(extra_cbs),
        loss_fn = lossfn,
        optimizer = opt,
        scheduler = sched,
    )

    try: learner.fit(n_epochs, dltrain, dltest, test_first=test_first, test_on_interrupt=False)
    except Exception as e: print(e)
    finally: return learner


def continue_train(
    path:str,
    title:str,
    dltrain,
    dltest,
    model,
    opt,
    sched,
    ref_sample:tuple[torch.Tensor, torch.Tensor],
    lossfn = DiceFocalLoss(softmax=True),
    n_epochs = 5,
    extra_cbs = (),
    test_first = True,
    around = 1,
    ):

    clean_mem()
    #MODEL = gnn.LSUV(MODEL, dltrain, max_iter=4)

    cbs = [
        LogLossCB(),
        LogTimeCB(),
        LogLRCB(),
        MONAIConfusionMatrixMetricsCB(['NETC', 'SNFH', 'ET', 'RC'], include_bg=False),
        TorchzeroDiceCB(['BG', 'NETC', 'SNFH', 'ET', 'RC'], step=8),
        TorchzeroSoftdiceCB(['BG', 'NETC', 'SNFH', 'ET', 'RC'], step=8),
        TorchzeroIoUCB(['BG', 'NETC', 'SNFH', 'ET', 'RC'], step=8),
        PerformanceTweaksCB(True),
        AccelerateCB("no"),
        SaveBestCB(),
        SaveLastCB(),
        FastProgressBarCB(step_batch=128, plot=True, metrics=PBAR_METRICS, maxv=1, smooth = SMOOTH),
        DisplayLoggerTableCB(),
        SaveReferenceVisualizationsAfterEachEpochCB(around=around),
        SaveForwardChannelImagesCB(ref_sample[0].unsqueeze(0)),
        SaveBackwardChannelImagesCB(*ref_sample, unsqueeze=True),
        SaveUpdateChannelImagesCB(*ref_sample, unsqueeze=True),
        #Metric_PredsTargetsFn(LOSS_FN, name='dice loss'),
        #CallTrainAndEvalOnOptimizer(),
        #AddLossReturnedByModelToLossInBackward(),
    ]

    learner = Learner.from_checkpoint(
        dir = path,
        model = model,
        cbs = cbs + list(extra_cbs),
        loss_fn = lossfn,
        optimizer = opt,
        scheduler = sched,
    )


    try: learner.fit(n_epochs, dltrain, dltest, test_first=test_first, test_on_interrupt=False, start_epoch=learner.total_epoch)
    except Exception as e: print(e)
    finally: return learner


def get_pretrained_model():
    from torchzero.nn.nets.unet import SegResNet
    model = SegResNet(44, 5, 2,).to(torch.device('cuda'))
    weights = r"F:\Stuff\Programming\experiments\brats2024-gli\training\0-1000 1000-1350\runs\came around5 50epochs 3 - 2024.7.18 21-8\checkpoints\e44 b334742 test-loss_0.18626\model.pt"
    model.load_state_dict(torch.load(weights))
    return model

def get_pretrained_model2():
    from torchzero.nn.nets.unet import SegResNet
    model = SegResNet(20, 5, 2,).to(torch.device('cuda'))
    weights = r"F:\Stuff\Programming\experiments\brats2024-gli\training\0-1000 1000-1350\runs\came around2 30epochs fast warmup 1e-2 0.5anyp gradclipnorm4 1 - 2024.7.25 11-26\checkpoints\e48 b643027 test-loss_0.09860\model.pt"
    model.load_state_dict(torch.load(weights))
    return model

@torch.no_grad
def dim_inference_raw(
    model,
    sliding_inferer,
    inputs: torch.Tensor,
    around,
    dim: Literal[0, 1, 2],
    ):
    from torchzero.nn.layers.pad import pad_to_shape
    if inputs.ndim != 4: raise ValueError(f"mritensor must be 4D, but got shape {inputs.shape}")
    # padding to match input size
    if dim == 1: inputs = inputs.swapaxes(1, 2)
    elif dim == 2: inputs = inputs.swapaxes(1, 3)
    padded_shape = list(inputs.shape)
    padded_shape[1] += around*2
    inputs = pad_to_shape(inputs, shape=tuple(padded_shape), where='center', mode='constant', value=inputs.min())
    mri_slicer = MRISlicer(inputs, torch.zeros_like(inputs)[0], 1, around=around, any_prob=0, warn_empty=False)
    slices = torch.stack([i[0] for i in mri_slicer.get_all_dim_slices(0)], 0)
    preds = sliding_inferer(inputs = slices, network = model).swapaxes(0,1)
    if dim == 1: preds = preds.swapaxes(1, 2)
    elif dim == 2: preds = preds.swapaxes(1, 3)
    return preds

FLIP_DIMS = (None, 1, 2, 3, (1, 2), (1, 3), (2, 3), (1, 2, 3))

@torch.no_grad
def inference_raw_tta_separate(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = True,
    device=torch.device("cuda"),
):
    from glio.progress_bar import PBar # type:ignore pylint:disable=W0621
    results = []
    model = model.to(device)
    inputs = inputs.to(device)
    inferer = SlidingWindowInferer(
        roi_size=(96, 96),
        sw_batch_size=batch_size,
        overlap=overlap,
        mode="gaussian",
        progress=False,
    )
    #for swapaxes in PBar(SWAPAXES_DIMS) if pbar else SWAPAXES_DIMS:
    for dim in PBar((0,1,2)) if pbar else (0,1,2):
        for flipdims in PBar(FLIP_DIMS) if pbar else FLIP_DIMS:
            inputs_flipped = inputs if flipdims is None else inputs.flip(flipdims)
            preds = dim_inference_raw(
                model = model,
                sliding_inferer = inferer,
                inputs = inputs_flipped,
                around = around,
                dim = dim,)
            if flipdims is not None: preds = preds.flip(flipdims)
            results.append(preds.cpu())
            if printp: print('|', end = '')
    if printp: print()
    return results

@torch.no_grad
def inference_raw_tta_mean(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = False,
    device=torch.device("cuda"),
):
    results = inference_raw_tta_separate(
        model=model,
        inputs=inputs,
        around=around,
        overlap=overlap,
        batch_size=batch_size,
        pbar=pbar,
        printp = printp,
        device=device,
    )
    return torch.stack(results, dim=0).mean(0).cpu()

@torch.no_grad
def inference_argmax_tta(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = False,
    device=torch.device("cuda"),
):
    results = inference_raw_tta_separate(
        model=model,
        inputs=inputs,
        around=around,
        overlap=overlap,
        batch_size=batch_size,
        pbar=pbar,
        printp = printp,
        device=device,
    )
    return torch.stack(results, dim=0).mean(0).argmax(0).cpu()

@torch.no_grad
def inference_onehot_tta(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = False,
    device=torch.device("cuda"),
):
    results = inference_raw_tta_separate(
        model=model,
        inputs=inputs,
        around=around,
        overlap=overlap,
        batch_size=batch_size,
        pbar=pbar,
        printp = printp,
        device=device,
    )
    return raw_preds_to_one_hot(torch.stack(results, dim=0).mean(0)).cpu()

@torch.no_grad
def inference_onehot_tta_mode(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = False,
    device=torch.device("cuda"),
):
    from monai.transforms import VoteEnsemble # type:ignore
    results = inference_raw_tta_separate(
        model=model,
        inputs=inputs,
        around=around,
        overlap=overlap,
        batch_size=batch_size,
        pbar=pbar,
        printp = printp,
        device=device,
    )
    one_hot_results =  [raw_preds_to_one_hot(i) for i in results]
    return VoteEnsemble()(one_hot_results).cpu() # type:ignore

@torch.no_grad
def inference_argmax_tta_mode(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = False,
    device=torch.device("cuda"),
):
    from monai.transforms import VoteEnsemble # type:ignore
    results = inference_raw_tta_separate(
        model=model,
        inputs=inputs,
        around=around,
        overlap=overlap,
        batch_size=batch_size,
        pbar=pbar,
        printp = printp,
        device=device,
    )
    one_hot_results =  [raw_preds_to_one_hot(i) for i in results]
    return VoteEnsemble()(one_hot_results).argmax(0).cpu() # type:ignore

@torch.no_grad
def inference_softmax_argmax_tta(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    around,
    overlap=0.75,
    batch_size=128,
    pbar=False,
    printp = False,
    device=torch.device("cuda"),
):
    results = inference_raw_tta_separate(
        model=model,
        inputs=inputs,
        around=around,
        overlap=overlap,
        batch_size=batch_size,
        pbar=pbar,
        printp = printp,
        device=device,
    )
    return torch.stack([i.softmax(0) for i in results], dim=0).mean(0).argmax(0).cpu()

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

def get_mri_from_folder(path, cropbg=True):
    """Returns 4* tensor with T1n, T1c, T2f, T2w"""
    t1c = find_file_containing(path, 't1c.')
    t1n = find_file_containing(path, 't1n.')
    t2f = find_file_containing(path, 't2f.')
    t2w = find_file_containing(path, 't2w.')
    if cropbg: t1c_crop, t1n_crop, t2f_crop, t2w_crop = crop_bg_imgs([t1c, t1n, t2f, t2w])
    else: t1c_crop, t1n_crop, t2f_crop, t2w_crop = t1c, t1n, t2f, t2w
    t1c_norm, t1n_norm, t2f_norm, t2w_norm = znormalize_imgs([t1c_crop, t1n_crop, t2f_crop, t2w_crop])
    return torch.from_numpy(np.stack([sitk.GetArrayFromImage(t1n_norm),
                     sitk.GetArrayFromImage(t1c_norm),
                     sitk.GetArrayFromImage(t2f_norm),
                     sitk.GetArrayFromImage(t2w_norm)])).to(torch.float32)

def predict_cropped(model, path, outfile, around, printp=False):
    t1c = find_file_containing(path, 't1c.')
    t1n = find_file_containing(path, 't1n.')
    t2f = find_file_containing(path, 't2f.')
    t2w = find_file_containing(path, 't2w.')
    t1c_crop, t1n_crop, t2f_crop, t2w_crop = crop_bg_imgs([t1c, t1n, t2f, t2w])
    t1c_norm, t1n_norm, t2f_norm, t2w_norm = znormalize_imgs([t1c_crop, t1n_crop, t2f_crop, t2w_crop])
    stacked_tensor = torch.from_numpy(np.stack([sitk.GetArrayFromImage(t1n_norm),
                     sitk.GetArrayFromImage(t1c_norm),
                     sitk.GetArrayFromImage(t2f_norm),
                     sitk.GetArrayFromImage(t2w_norm)])).to(torch.float32)
    preds = inference_softmax_argmax_tta(model, stacked_tensor, around = around, pbar=False, printp = printp, overlap=0.75)
    sitk_preds = sitk.GetImageFromArray(preds.to(torch.uint8).numpy())
    sitk_preds.CopyInformation(t1c_norm)
    resampled_sitk_preds = resample_to(sitk_preds, t1c)
    resampled_sitk_preds.CopyInformation(sitk.ReadImage(t1c))
    resampled_sitk_preds.SetOrigin((-90., 126., -72.))
    sitk.WriteImage(resampled_sitk_preds, outfile)
    return resampled_sitk_preds

def predict_full(model, path, outfile, around, printp=False):
    t1c = find_file_containing(path, 't1c.')
    t1n = find_file_containing(path, 't1n.')
    t2f = find_file_containing(path, 't2f.')
    t2w = find_file_containing(path, 't2w.')
    #t1c_crop, t1n_crop, t2f_crop, t2w_crop = crop_bg_imgs([t1c, t1n, t2f, t2w])
    t1c_norm, t1n_norm, t2f_norm, t2w_norm = znormalize_imgs([t1c, t1n, t2f, t2w])
    stacked_tensor = torch.from_numpy(np.stack([sitk.GetArrayFromImage(t1n_norm),
                     sitk.GetArrayFromImage(t1c_norm),
                     sitk.GetArrayFromImage(t2f_norm),
                     sitk.GetArrayFromImage(t2w_norm)])).to(torch.float32)
    preds = inference_softmax_argmax_tta(model, stacked_tensor, around = around, pbar=False, printp = printp, overlap=0.75)
    sitk_preds = sitk.GetImageFromArray(preds.to(torch.uint8).numpy())
    sitk_preds.CopyInformation(sitk.ReadImage(t1c))
    sitk_preds.SetOrigin((-90., 126., -72.))
    sitk.WriteImage(sitk_preds, outfile)
    return sitk_preds


def predict_dataset(model, path, outdir, around, ):
    from glio.progress_bar import PBar # type:ignore pylint:disable=W0621
    for dir in PBar(os.listdir(path), step=1):
        full_dir = os.path.join(path, dir)
        predict_cropped(model, full_dir, os.path.join(outdir, 'seg-' + dir + '.nii.gz'), around = around, printp=False)