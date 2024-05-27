from functools import partial
import os
import torch
from matplotlib.colors import ListedColormap
from monai.inferers import SlidingWindowInferer # type:ignore
from ..plot import *
from ..train2 import *
from ..python_tools import find_file_containing
from ..torch_tools import one_hot_mask
from ..transforms import norm_to01
from ..datasets.preprocessor import Preprocessor, _load_if_needed
from ..progress_bar import PBar
from ..loaders import niiread_affine

brgb = ListedColormap(['black','red', 'green', 'blue'])
brgb_legend = "\nчерный:нет;\nсиний:отёк;\nзелёный:некротическое ядро,\nкрасный:усиливающая опухоль"

def visualize_3_segm_classes(inputs:torch.Tensor, segm:torch.Tensor):
    """
    inputs: HW
    segm: 4HW or HW, where 0th class is background"""
    preview = torch.stack([inputs,inputs,inputs], dim=0)
    if segm.ndim == 3: segm = segm.argmax(0)
    preview[0] = torch.where(segm == 1, preview[0]*2, preview[0]).clip(0,1)
    preview[1] = torch.where(segm == 2, preview[1]*2, preview[2]).clip(0,1)
    preview[2] = torch.where(segm == 3, preview[2]*2, preview[2]).clip(0,1)
    return preview

def visualize_predictions(inferer, sample:tuple[torch.Tensor, torch.Tensor], around=1, expand_channels = None, save=False, path=None):
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

    inputs[0] = torch.stack([norm_to01(i) for i in inputs[0]]) # type:ignore
    fig.add().imshow_batch(inputs[0, 1::around*2+1], scale_each=True).style_img('вход:\nT1c, T1, FLAIR, T2')
    fig.add().imshow_batch(preds_raw, scale_each=True).style_img(f'сырой выход\n{brgb_legend}')
    fig.add().imshow(targets, cmap=brgb).style_img(f'реальная карта\n{brgb_legend}')
    fig.add().imshow(preds, cmap=brgb).style_img(f'предсказанная карта\n{brgb_legend}')
    fig.add().imshow_batch((preds_raw - targets_raw).abs(), scale_each=True).style_img('сырая ошибка')
    fig.add().imshow(preds != targets,  cmap='gray').style_img('ошибка')

    fig.add().imshow(visualize_3_segm_classes(inputs[0][0], targets)).style_img(f'реальная карта\n{brgb_legend}')
    fig.add().imshow(visualize_3_segm_classes(inputs[0][0], preds)).style_img(f'предсказанная карта\n{brgb_legend}')

    fig.create(2, figsize=(16,16))
    if save:
        fig.savefig(path)
        fig.close()

def visualize_our_predictions(inferer, sample: torch.Tensor, around=1, expand_channels = None, save=False, path=None):
    """Sample is (CHW, CHW) tuple"""
    fig = Figure()
    inputs:torch.Tensor = sample.unsqueeze(0)
    if expand_channels is None: preds_raw:torch.Tensor = inferer(inputs)[0]
    else:
        expanded = torch.cat((inputs, torch.zeros((1, expand_channels - inputs.shape[1], *inputs.shape[2:]))), dim=1)
        preds_raw:torch.Tensor = inferer(expanded)[0]
    preds:torch.Tensor = preds_raw.argmax(0)

    inputs[0] = torch.stack([norm_to01(i) for i in inputs[0]]) # type:ignore
    fig.add().imshow_batch(inputs[0, 1::around*2+1], scale_each=True).style_img('вход:\nT1c, T1, FLAIR, T2')
    fig.add().imshow_batch(preds_raw, scale_each=True).style_img(f'сырой выход\n{brgb_legend}')
    fig.add().imshow(preds, cmap=brgb).style_img(f'предсказанная карта\n{brgb_legend}')
    fig.add().imshow(visualize_3_segm_classes(inputs[0][0], preds)).style_img(f'предсказанная карта\n{brgb_legend}')

    fig.create(1, figsize=(16,16))

    if save:
        fig.savefig(path)
        fig.close()

def get_brats_preprocessed_by_idx(index, hist=False):
    """Returns a 5*h*w tensor: `t1c, t1n, t2f, t2w, seg`"""
    path = rf'E:\dataset\BRaTS2024-GoAT\ISBI2024-BraTS-GoAT-TrainingData/BraTS-GoAT-{str(index).rjust(5, "0")}'
    t1c = find_file_containing(path, 't1c.')
    t1n = find_file_containing(path, 't1n.')
    t2f = find_file_containing(path, 't2f.')
    t2w = find_file_containing(path, 't2w.')
    seg = find_file_containing(path, 'seg')
    p = Preprocessor()
    images_nohist, images, seg = p(t1c, t1n, t2f, t2w, seg=seg, return_nohist=True)
    if hist: return torch.cat((images, seg.unsqueeze(0)))
    else: return torch.cat((images_nohist, seg.unsqueeze(0)))

def get_rhuh_preprocessed_by_idx(index, hist=False):
    """Returns a 6*h*w tensor: `t1c, t1n, t2f, t2w, adc, seg`"""
    index1, index2 = int(index // 3)+1, int(index % 3)
    path = rf'E:\dataset\RHUH-GBM\RHUH-GBM_nii_v1/RHUH-{str(index1).rjust(4, "0")}/{index2}'
    t1c = find_file_containing(path, 't1ce.')
    t1n = find_file_containing(path, 't1.')
    t2f = find_file_containing(path, 'flair.')
    t2w = find_file_containing(path, 't2.')
    adc = find_file_containing(path, 'adc.')
    seg = find_file_containing(path, 'seg')
    p = Preprocessor()
    images_nohist, images, seg = p(t1c, t1n, t2f, t2w, adc, seg=seg, return_nohist=True)
    if hist: return torch.cat((images, seg.unsqueeze(0)))
    else: return torch.cat((images_nohist, seg.unsqueeze(0)))


def _our_rotate(x):
    return x.flip(2)

def get_our_preprocessed_by_idx_old(index, hist=False, names = ("T1C", "T1N", "FLAIR", "T2W")):
    """Returns a 4*h*w tensor: `t1c, t1n, t2f, t2w`"""
    index1, index2 = int(index // 2)+1, int(index % 2)
    path = rf'D:\vkr\new\nii_preprocessed/{index1}'
    path = path + '/' + os.listdir(path)[index2]
    t1c = find_file_containing(path, names[0])
    t1n = find_file_containing(path, names[1])
    t2f = find_file_containing(path, names[2])
    t2w = find_file_containing(path, names[3])
    p = Preprocessor(rotate=_our_rotate)
    images_nohist, images = p(t1c, t1n, t2f, t2w, return_nohist=True)
    if hist: return images
    return images_nohist


def find_t1(folder):
    x = find_file_containing(folder, 'T1W_3D_TFE_ ', error=False)
    if x is None: x = find_file_containing(folder, 'T1 SPGR S ', error=False)
    if x is None: x = find_file_containing(folder, 'T1 FSE F ', error=True)
    return x

def find_t1ce(folder):
    x = find_file_containing(folder, 'T1W_3D_TFE_+C', error=False)
    if x is None: x = find_file_containing(folder, 'T1 SPGR S+C', error=False)
    if x is None: x = find_file_containing(folder, 'T1 FSE F+C', error=True)
    return x

def find_t2(folder):
    return find_file_containing(folder, 'T2')

def find_flair(folder):
    x = find_file_containing(folder, 'FLAIR', error=False)
    if x is None: x = find_file_containing(folder, 'Flair', error=False)
    if x is None: x = find_file_containing(folder, 'flair', error=True)
    return x

def get_our_preprocessed_by_idxs(idx1, idx2, hist=False, return_affine = False):
    """Returns a 4*h*w tensor: `t1c, t1n, t2f, t2w`"""
    path = rf'D:\vkr\new\nii\{idx1}\{idx2}\preop\wdir'
    path = path + '\\' + fr'{os.listdir(path)[0]}\nii_final'
    t1c = find_t1ce(path)
    t1n = find_t1(path)
    t2f = find_flair(path)
    t2w = find_t2(path)
    p = Preprocessor(rotate=_our_rotate)
    images_nohist, images = p(t1c, t1n, t2f, t2w, return_nohist=True)
    if return_affine:
        affine = niiread_affine(t1c) # type:ignore
        if hist: return images, affine
        return images_nohist, affine
    else:
        if hist: return images
        return images_nohist


def sliding_inference_around_3d(inputs:torch.Tensor, inferer, size, step, around, nlabels, expand = None):
    """Input must be a 4D C* or 5D BC* tensor"""
    if inputs.ndim == 4: inputs = inputs.unsqueeze(0)
    results = torch.zeros((inputs.shape[0], nlabels, *inputs.shape[2:]), device=inputs.device,)
    counts = torch.zeros_like(results)
    for x in range(around, inputs.shape[2]-around, 1):
        for y in range(0, inputs.shape[3]-size[0], step):
            for z in range(0, inputs.shape[4]-size[1], step):
                print(x, y, z, end='    \r')
                inputs_slice = inputs[:, :, x-1:x+around+1, y:y+size[0], z:z+size[1]].flatten(1,2)
                if expand: inputs_slice = torch.cat((inputs_slice, torch.zeros((1, expand - inputs_slice.shape[1], *inputs_slice.shape[2:]))), dim=1)
                preds = inferer(inputs_slice)
                results[:, :, x, y:y+size[0], z:z+size[1]] += preds
                counts[:, :, x, y:y+size[0], z:z+size[1]] += 1

    results /= counts
    return results.nan_to_num(0)



def sliding_inference_around_3d_batched(inputs:torch.Tensor, inferer, size, step, nlabels, expand = None):
    """Input must be a 4D C* tensor, around is 1"""
    inputs = inputs.swapaxes(0,1) # B, C, H, W
    inputs = torch.stack(( inputs[:-2], inputs[1:-1],inputs[2:]), 2).flatten(1,2)
    results = torch.zeros((inputs.shape[0], nlabels, *inputs.shape[2:]), device=inputs.device,)
    counts = torch.zeros_like(results)
    for x in range(0, inputs.shape[2]-size[0], step):
        for y in range(0, inputs.shape[3]-size[1], step):
            print(f'{x}/{inputs.shape[2]-size[0]}, {y}/{inputs.shape[3]-size[1]}', end='    \r')
            inputs_slice = inputs[:, :, x:x+size[0], y:y+size[1]]
            if expand: inputs_slice = torch.cat((inputs_slice, torch.zeros((inputs_slice.shape[0], expand - inputs_slice.shape[1], *inputs_slice.shape[2:]))), dim=1)
            preds = inferer(inputs_slice)
            results[:, :, x:x+size[0], y:y+size[1]] += preds
            counts[:, :, x:x+size[0], y:y+size[1]] += 1

    results /= counts

    padding = torch.zeros((1, *results.shape[1:],))
    results = torch.cat((padding, results, padding))
    return results.nan_to_num(0).swapaxes(0,1)

def sliding_inference_around_3d_monai(inputs:torch.Tensor, inferer, size, overlap=0.5, expand = None):
    """Input must be a 4D C* tensor, around is 1"""
    inputs = inputs.swapaxes(0,1) # B, C, H, W
    inputs = torch.stack(( inputs[:-2], inputs[1:-1],inputs[2:]), 2).flatten(1,2)
    if expand: inputs = torch.cat((inputs, torch.zeros((inputs.shape[0], expand-inputs.shape[1], *inputs.shape[2:]))), 1)
    sliding = SlidingWindowInferer(size, 32, overlap, mode='gaussian', progress=True)

    results = sliding(inputs, inferer)
    padding = torch.zeros((1, *results.shape[1:],)) # type:ignore
    results = torch.cat((padding, results, padding)) # type:ignore
    return results.swapaxes(0,1) # type:ignore


def get_checkpoint_preds_on_our_around(cpath, model, idx1, idx2, hist=False, expand=None, inferer_fn = sliding_inference_around_3d_batched, **inferer_kwargs):
    """Tests a checkpoint on our images, around=1"""
    input = get_our_preprocessed_by_idxs(idx1, idx2, hist=hist)

    learner = Learner.from_checkpoint(cpath, model=model, cbs=())
    preds = inferer_fn(input, learner.inference, size=(96,96), expand=expand, **inferer_kwargs)

    return preds

brats_references = [
    (partial(get_brats_preprocessed_by_idx, 0), 43),
    (partial(get_brats_preprocessed_by_idx, 1), 102),
    (partial(get_brats_preprocessed_by_idx, 3), 39),
    (partial(get_brats_preprocessed_by_idx, 5), 79),
]

rhuh_references = [
    (partial(get_rhuh_preprocessed_by_idx, 2), 99),
    (partial(get_rhuh_preprocessed_by_idx, 7), 23),
    (partial(get_rhuh_preprocessed_by_idx, 8), 33),
    (partial(get_rhuh_preprocessed_by_idx, 16), 62),
]

our_references = [
    # (partial(get_our_preprocessed_by_idx, 0), 91),
    # (partial(get_our_preprocessed_by_idx, 1), 106),
    # (partial(get_our_preprocessed_by_idx, 2), 90),
    # (partial(get_our_preprocessed_by_idx, 3), 96),
    # (partial(get_our_preprocessed_by_idx, 4), 53),
    # (partial(get_our_preprocessed_by_idx, 5), 54),
    # (partial(get_our_preprocessed_by_idx, 6), 99),
    # (partial(get_our_preprocessed_by_idx, 8), 39),
    # (partial(get_our_preprocessed_by_idx, 9), 36),
    # (partial(get_our_preprocessed_by_idx, 10), 97),
    # (partial(get_our_preprocessed_by_idx, 11), 105),
    #(partial(get_our_preprocessed_by_idxs, 0), 91),
    (partial(get_our_preprocessed_by_idxs, 1, 1), 106),
    (partial(get_our_preprocessed_by_idxs, 2, 0), 90),
    (partial(get_our_preprocessed_by_idxs, 2, 1), 96),
    (partial(get_our_preprocessed_by_idxs, 3, 0), 53),
    (partial(get_our_preprocessed_by_idxs, 3, 1), 54),
    #(partial(get_our_preprocessed_by_idxs, 4, 0), 99),
    (partial(get_our_preprocessed_by_idxs, 5, 0), 39),
    (partial(get_our_preprocessed_by_idxs, 5, 1), 36),
    #(partial(get_our_preprocessed_by_idxs, 6, 0), 97),
    (partial(get_our_preprocessed_by_idxs, 6, 1), 105),
]


def visualize_brats_reference(idx, inferer, around=1, hist=False, overlap=0.75, expand=None, save=False, folder=None, prefix=''):
    """0,1,2,3. Model must accept sequential around inputs, i.e. 3 T1c, 3 T1n, etc..."""
    slice_idx = brats_references[idx][1]
    image3d = brats_references[idx][0](hist=hist) # C D H W
    inputs3d, seg3d = image3d[:-1], image3d[-1]
    if around: inputs_around = inputs3d[:, slice_idx-around:slice_idx+around+1].flatten(0,1)
    else: inputs_around = inputs3d[:, slice_idx]
    seg = seg3d[slice_idx]

    if expand: inputs_around = torch.cat((inputs_around, torch.zeros((expand - inputs_around.shape[0], *inputs_around.shape[1:]))), dim=0)
    sliding = SlidingWindowInferer((96,96), 32, overlap=overlap, mode='gaussian')
    visualize_predictions(partial(sliding.__call__, network=inferer), (inputs_around, one_hot_mask(seg, 4)), save=save, path=f'{folder}/{prefix}BRATS {idx}.jpg')

def visualize_brats_reference_all(inferer, around=1, hist=False, overlap=0.75, expand=None, save=False, folder=None, prefix=''):
    for i in range(4): visualize_brats_reference(i, inferer=inferer, around=around, hist=hist, overlap=overlap, expand=expand, save=save, folder=folder,prefix=prefix)

def visualize_rhuh_reference(idx, inferer, around=1, hist=False, overlap=0.75, expand=None, pass_adc = False, save=False, folder=None, prefix=''):
    """0,1,2,3. Model must accept sequential around inputs, i.e. 3 T1c, 3 T1n, etc..."""
    slice_idx = rhuh_references[idx][1]
    image3d = rhuh_references[idx][0](hist=hist) # C D H W
    inputs3d, seg3d = image3d[:-1], image3d[-1]
    if not pass_adc: inputs3d = inputs3d[:4]
    if around: inputs_around = inputs3d[:, slice_idx-around:slice_idx+around+1].flatten(0,1)
    else: inputs_around = inputs3d[:, slice_idx]
    seg = seg3d[slice_idx]

    if expand: inputs_around = torch.cat((inputs_around, torch.zeros((expand - inputs_around.shape[0], *inputs_around.shape[1:]))), dim=0)
    sliding = SlidingWindowInferer((96,96), 32, overlap=overlap, mode='gaussian')
    visualize_predictions(partial(sliding.__call__, network=inferer), (inputs_around, one_hot_mask(seg, 4)), save=save, path=f'{folder}/{prefix}RHUH {idx}.jpg')

def visualize_rhuh_reference_all(inferer, around=1, hist=False, overlap=0.75, expand=None, pass_adc = False, save=False, folder=None, prefix=''):
    for i in range(4): visualize_rhuh_reference(i, inferer=inferer, around=around, hist=hist, overlap=overlap, expand=expand, pass_adc=pass_adc, save=save, folder=folder,prefix=prefix)

def visualize_our_reference(idx, inferer, around=1, hist=False, overlap=0.75, expand=None, save=False, folder=None, prefix=''):
    """0-10. Model must accept sequential around inputs, i.e. 3 T1c, 3 T1n, etc..."""
    slice_idx = our_references[idx][1]
    inputs3d = our_references[idx][0](hist=hist) # C D H W
    if around: inputs_around = inputs3d[:, slice_idx-around:slice_idx+around+1].flatten(0,1)
    else: inputs_around = inputs3d[:, slice_idx]

    if expand: inputs_around = torch.cat((inputs_around, torch.zeros((expand - inputs_around.shape[0], *inputs_around.shape[1:]))), dim=0)
    sliding = SlidingWindowInferer((96,96), 32, overlap=overlap, mode='gaussian')
    visualize_our_predictions(partial(sliding.__call__, network=inferer), inputs_around, save=save, path=f'{folder}/{prefix}OUR {idx}.jpg')


def visualize_our_reference_all(inferer, around=1, hist=False, overlap=0.75, expand=None, save=False, folder=None, prefix=''):
    for i in range(11): visualize_our_reference(i, inferer=inferer, around=around, hist=hist, overlap=overlap, expand=expand, save=save, folder=folder, prefix=prefix)


def visualize_all_references(inferer, around=1, hist=False, overlap=0.75, expand=None, pass_adc=False, save=False, folder=None,prefix=''):
    visualize_brats_reference_all(inferer, around=around, hist=hist, overlap=overlap, expand=expand, save=save, folder=folder,prefix=prefix)
    visualize_rhuh_reference_all(inferer, around=around, hist=hist, overlap=overlap, expand=expand, pass_adc=pass_adc, save=save, folder=folder,prefix=prefix)
    visualize_our_reference_all(inferer, around=around, hist=hist, overlap=overlap, expand=expand, save=save, folder=folder,prefix=prefix)


class SaveReferenceVisualizationsAfterEachEpoch(CBMethod):
    order = 1
    def __init__(self, folder, brats=(0,), rhuh=(2,), our=(0,8,9)):
        self.folder=folder
        if not os.path.exists(folder): os.mkdir(folder)
        self.brats = brats
        self.rhuh = rhuh
        self.our = our

    def after_test_epoch(self, learner:Learner):
        if not os.path.exists(f'{self.folder}/{learner.cp_number}. {learner.name}'): os.mkdir(f'{self.folder}/{learner.cp_number}. {learner.name}')
        folder = f'{self.folder}/{learner.cp_number}. {learner.name}'
        prefix=f'{learner.total_epoch} {float(learner.logger.last("test loss")):.4f} '
        for i in self.brats: visualize_brats_reference(i, learner.inference, save=True, folder=folder, prefix=prefix)
        for i in self.rhuh: visualize_rhuh_reference(i, learner.inference, save=True, folder=folder, prefix=prefix)
        #for i in self.our: visualize_our_reference(i, learner.inference, save=True, folder=folder, prefix=prefix)


def show_slices_with_seg_from_files(*imgs, seg):
    from ..jupyter_tools import show_slices
    p = Preprocessor(hist_correction=None, rotate=_our_rotate)
    image, seg = p(*imgs, seg=seg)
    preview = norm_to01(torch.stack((image,image,image)))
    print(preview.shape)
    preview2 = preview.clone() # type:ignore
    preview[0] = torch.where(seg == 1, preview[0]*2, preview[0]).clip(0,1)
    preview[1] = torch.where(seg == 2, preview[1]*2, preview[2]).clip(0,1)
    preview[2] = torch.where(seg == 3, preview[2]*2, preview[2]).clip(0,1)
    #cat = torch.cat((image, torch.stack([seg for _ in range(5)])), 3)
    show_slices(torch.cat((preview, preview2), 2).permute(1,4,2,3,0)) # type:ignore

def show_rhuh_slices_with_seg(idx, seg=None):
    from ..jupyter_tools import show_slices
    arr = get_rhuh_preprocessed_by_idx(idx).permute(0,3,2,1)
    image, _seg = arr[:-1], arr[-1]
    if seg is None: seg = _seg
    else: seg = _load_if_needed(seg)
    preview = norm_to01(torch.stack((image,image,image)))
    preview2 = preview.clone() # type:ignore
    preview[0] = torch.where(seg == 1, preview[0]*2, preview[0]).clip(0,1)
    preview[1] = torch.where(seg == 2, preview[1]*2, preview[2]).clip(0,1)
    preview[2] = torch.where(seg == 3, preview[2]*2, preview[2]).clip(0,1)
    #cat = torch.cat((image, torch.stack([seg for _ in range(5)])), 3)
    show_slices(torch.cat((preview, preview2), 4).permute(1,2,3,4,0)) # type:ignore

def show_our_slices_with_seg(idx1, idx2, seg, rotate=True):
    from ..jupyter_tools import show_slices
    seg = _load_if_needed(seg)
    image = get_our_preprocessed_by_idxs(idx1, idx2)

    if rotate:
        image = image.permute(0,3,2,1)
        seg = seg.permute(2,1,0)
    #image, seg = arr[:-1], arr[-1]
    preview = norm_to01(torch.stack((image,image,image)))
    preview2 = preview.clone() # type:ignore
    preview[0] = torch.where(seg == 1, preview[0]*2, preview[0]).clip(0,1)
    preview[1] = torch.where(seg == 2, preview[1]*2, preview[2]).clip(0,1)
    preview[2] = torch.where(seg == 3, preview[2]*2, preview[2]).clip(0,1)
    #cat = torch.cat((image, torch.stack([seg for _ in range(5)])), 3)
    show_slices(torch.cat((preview, preview2), 4).permute(1,2,3,4,0)) # type:ignore
    
    
def sliding_inference_around_tta(inputs:torch.Tensor, inferer, size, expand = None, inferer_fn = sliding_inference_around_3d_batched, **inferer_kwargs):
    """Input must be a 4D C* or 5D BC* tensor"""
    result1 = inferer_fn(inputs, inferer, size=size,  expand=expand, **inferer_kwargs)
    result2 = inferer_fn(inputs.flip(-1), inferer, size=size,  expand=expand, **inferer_kwargs).flip(-1)
    result3 = inferer_fn(inputs.flip(-2), inferer, size=size,  expand=expand, **inferer_kwargs).flip(-2)
    result4 = inferer_fn(inputs.flip((-1, -2)), inferer, size=size,  expand=expand, **inferer_kwargs).flip((-1, -2))

    inputs2 = inputs.swapaxes(-1, -2)
    result5 = inferer_fn(inputs2, inferer, size=size, expand=expand, **inferer_kwargs).swapaxes(-1, -2)
    result6 = inferer_fn(inputs2.flip(-1), inferer, size=size, expand=expand, **inferer_kwargs).flip(-1).swapaxes(-1, -2)
    result7 = inferer_fn(inputs2.flip(-2), inferer, size=size,  expand=expand, **inferer_kwargs).flip(-2).swapaxes(-1, -2)
    result8 = inferer_fn(inputs2.flip((-1, -2)), inferer, size=size,  expand=expand, **inferer_kwargs).flip((-1, -2)).swapaxes(-1, -2)

    return torch.stack((result1, result2, result3, result4, result5, result6, result7, result8)).sum(0)