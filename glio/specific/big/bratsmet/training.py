import os
from functools import partial

import torch
from monai.inferers import SlidingWindowInferer # type:ignore

from ....mri.preprocess_datasets import (preprocess_brats2024met,
                                         preprocess_brats2024met_tensor)
from ....plot import Figure
from ....train import Learner, MethodCallback
from ....transforms.intensity import norm
from .dataset import CASE_DIRS



def visualize_predictions(inferer, sample:tuple[torch.Tensor, torch.Tensor], around=1, expand_channels = None, save=False, path=None, title=None):
    """Sample is (CHW, HW) tuple (image, binary seg)"""
    fig = Figure()
    inputs:torch.Tensor = sample[0].unsqueeze(0)
    targets:torch.Tensor = sample[1]
    if expand_channels is None: preds_raw:torch.Tensor = inferer(inputs)[0][0]
    else:
        expanded = torch.cat((inputs, torch.zeros((1, expand_channels - inputs.shape[1], *inputs.shape[2:]))), dim=1)
        preds_raw:torch.Tensor = inferer(expanded)[0][0]
    preds = torch.where(preds_raw > 0, 1, 0)

    inputs[0] = torch.stack([norm(i) for i in inputs[0]]) # type:ignore
    fig.add().imshow_batch(inputs[0, 1::around*2+1], scale_each=True).style_img('inputs:\nT1, T1ce, FLAIR, T2w')
    fig.add().imshow_batch(preds_raw, scale_each=True).style_img('raw outputs')
    fig.add().imshow(torch.zeros_like(targets)).seg_overlay(targets, alpha=1.).style_img('targets')
    fig.add().imshow(torch.zeros_like(preds)).seg_overlay(preds, alpha=1.).style_img('prediction')
    fig.add().imshow(preds_raw[1:-1]).style_img('raw outputs RGB')
    #fig.add().imshow_batch((preds_raw - targets_raw).abs(), scale_each=True).style_img('raw error')
    fig.add().imshow(preds != targets,  cmap='gray').style_img('error')

    fig.add().imshow(inputs[0][0]).seg_overlay(targets).style_img('targets')
    fig.add().imshow(inputs[0][0]).seg_overlay(preds).style_img('predictions')

    if title is None: title = ''
    fig.create(2, figsize=(16,12), title=f'{title}')
    if save:
        fig.savefig(path)
        fig.close()

REFERENCES = [
    (0, 95),
    (1, 137),
    (2, 48),
    (3, 160),
    (6, 34),
]

def visualize_reference(idx, inferer, around=1, overlap=0.75, expand=None, save=False, folder='reference preds', prefix='', mkdir = True):
    """0,1,2,3. Model must accept sequential around inputs, i.e. 3 T1c, 3 T1n, etc..."""
    inputs3d, seg3d = preprocess_brats2024met_tensor(CASE_DIRS[REFERENCES[idx][0]])
    slice_idx = REFERENCES[idx][1]

    if around: inputs_around = inputs3d[:, slice_idx-around:slice_idx+around+1].flatten(0,1)
    else: inputs_around = inputs3d[:, slice_idx]
    seg = seg3d[slice_idx]

    if expand: inputs_around = torch.cat((inputs_around, torch.zeros((expand - inputs_around.shape[0], *inputs_around.shape[1:]))), dim=0)
    sliding = SlidingWindowInferer((96,96), 32, overlap=overlap, mode='gaussian')
    if mkdir and not os.path.exists(folder): os.mkdir(folder)
    visualize_predictions(
        partial(sliding, network=inferer),
        (inputs_around, seg),
        save=save,
        path=f"{folder}/{prefix}BRATS-GLI {idx}.jpg",
        title=f'BRATS-GLI {idx}'
    )

class SaveReferenceVisualizationsAfterEachEpochCB(MethodCallback):
    order = 1
    def __init__(self, folder='runs', brats=tuple(range(len(REFERENCES))), around=1):
        self.folder=folder
        if not os.path.exists(folder): os.mkdir(folder)
        self.brats = brats
        self.around = around

    def after_test_epoch(self, learner:Learner):
        folder = os.path.join(learner.get_workdir(self.folder), 'reference preds')
        if not os.path.exists(folder): os.mkdir(folder)
        prefix=f'{learner.total_epoch} {float(learner.logger.last("test loss")):.4f} '
        for i in self.brats: visualize_reference(i, learner.inference, save=True, folder=folder, prefix=prefix, around=self.around)

