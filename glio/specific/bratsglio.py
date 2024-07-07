from functools import partial
import os
import torch
from monai.inferers import SlidingWindowInferer # type:ignore
from ..plot import *
from ..train import *
from ..torch_tools import one_hot_mask
from ..transforms.intensity import norm
from ..mri.preprocess_datasets import preprocess_brats2024gli_tensor

__all__ = [
    'visualize_predictions',
    'get_brats_gli_preprocessed_by_idx',
    'visualize_brats_reference',
    'visualize_brats_reference_all',
    'SaveReferenceVisualizationsAfterEachEpochCB',

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

def get_brats_gli_preprocessed_by_idx(index):
    """Returns a 5*h*w tensor: `t1c, t1n, t2f, t2w, seg`"""
    dspath = 'E:/dataset/BraTS-GLI v2/train'
    path = os.path.join(dspath, os.listdir(dspath)[index])
    images, seg = preprocess_brats2024gli_tensor(path)
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

class SaveReferenceVisualizationsAfterEachEpochCB(CBMethod):
    order = 1
    def __init__(self, folder='runs', brats=tuple(range(len(brats_gli_references)))):
        self.folder=folder
        if not os.path.exists(folder): os.mkdir(folder)
        self.brats = brats

    def after_test_epoch(self, learner:Learner):
        folder = os.path.join(learner.get_workdir(self.folder), 'reference preds')
        if not os.path.exists(folder): os.mkdir(folder)
        prefix=f'{learner.total_epoch} {float(learner.logger.last("test loss")):.4f} '
        for i in self.brats: visualize_brats_reference(i, learner.inference, save=True, folder=folder, prefix=prefix)
