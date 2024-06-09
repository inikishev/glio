from typing import Optional
import torch
from torchvision.transforms import v2
import  monai.transforms
from ..train2 import *
from ..torch_tools import to_binary

def get_first_model() -> Learner:
    from monai.networks.nets import VNet # type:ignore
    from schedulefree import AdamWScheduleFree
    MODEL = VNet(2, 4,4)
    OPT = AdamWScheduleFree(MODEL.parameters(), lr=1e-2, eps=1e-6)
    path = r"F:\Stuff\Programming\AI\glio_diff\glio\models\glio postop segm\refining tandem v1\1. v1 VNet"
    l = Learner.from_checkpoint(path, MODEL, [Accelerate("no"),], optimizer=OPT)
    l.fit(0, None, None)
    return l

def get_second_model() -> Learner:
    from monai.networks.nets import SegResNetDS # type:ignore
    from schedulefree import AdamWScheduleFree
    MODEL = SegResNetDS(2, in_channels=12, out_channels=4, init_filters=24)
    OPT = AdamWScheduleFree(MODEL.parameters(), lr=1e-3, eps=1e-6)
    path = r"F:\Stuff\Programming\AI\glio_diff\glio\models\glio postop segm\refining tandem v1\2. v1 SegResNetDS"
    l = Learner.from_checkpoint(path, MODEL, [Accelerate("no"),], optimizer=OPT)
    l.fit(0, None, None)
    return l

def inference_96(
    whole: Optional[torch.Tensor] = None,
    t1c: Optional[torch.Tensor] = None,
    t1n: Optional[torch.Tensor] = None,
    t2f: Optional[torch.Tensor] = None,
    t2w: Optional[torch.Tensor] = None,
):
    """Вход: 4x96x96 (t1c, t1n, t2f, t2w) или 4 соотвествующих изображения 96x96
    Выход: 4x96x96 (фон, отёк, некротическое ядро, усиливающая опухоль)"""
    model1 = get_first_model()
    model2 = get_second_model()
    if whole is None:
        if any([i is None for i in (t1c, t1n, t2f, t2w)]): raise ValueError("Не все модальности.")
        for i in [t1c, t1n, t2f, t2w]:
            if tuple(i.shape) != (96,96): raise ValueError(f"Все тензоры должны быть 96x96, получены: {[i.shape for i in [t1c, t1n, t2f, t2w]]}") # type:ignore
        inputs = torch.stack([t1c, t1n, t2f, t2w], dim=0) # type:ignore
    else:
        if tuple(whole.shape) != (4, 96, 96): raise ValueError(f"Полный тензор должен быть 4x96x96, получено: {whole.shape}")
        inputs = whole
    inputs = inputs.unsqueeze(0)
    preds1 = model1.inference(inputs)
    preds2 = model2.inference(torch.cat([inputs, preds1], 1))

    return to_binary(torch.nn.functional.softmax(preds2[0], 0), 0.5)

def inference_sliding_2d(
    whole: Optional[torch.Tensor] = None,
    t1c: Optional[torch.Tensor] = None,
    t1n: Optional[torch.Tensor] = None,
    t2f: Optional[torch.Tensor] = None,
    t2w: Optional[torch.Tensor] = None,):
    model1 = get_first_model()
    model2 = get_second_model()
    from monai.inferers import SlidingWindowInfererAdapt # type:ignore

    if whole is None:
        if any([i is None for i in (t1c, t1n, t2f, t2w)]): raise ValueError("Не все модальности.")
        inputs = torch.stack([t1c, t1n, t2f, t2w], dim=0) # type:ignore
    else:
        inputs = whole
    inputs = inputs.unsqueeze(0)
    inferer = SlidingWindowInfererAdapt(roi_size=(96, 96), sw_batch_size=1, overlap=0.25)
    preds1 = inferer(inputs, model1.inference)
    preds2 = inferer(torch.cat([inputs, preds1], 1), model2.inference) # type:ignore
    return to_binary(torch.nn.functional.softmax(preds2[0], 0), 0.5)

def inference_sliding_3d(
    whole: Optional[torch.Tensor] = None,
    t1c: Optional[torch.Tensor] = None,
    t1n: Optional[torch.Tensor] = None,
    t2f: Optional[torch.Tensor] = None,
    t2w: Optional[torch.Tensor] = None,):
    model1 = get_first_model()
    model2 = get_second_model()
    from monai.inferers import SlidingWindowInfererAdapt # type:ignore

    if whole is None:
        if any([i is None for i in (t1c, t1n, t2f, t2w)]): raise ValueError("Не все модальности.")
        if t1c.ndim != 3: raise ValueError("Все модальности должны быть объёмными тензорами")  #type:ignore
        inputs = torch.stack([t1c, t1n, t2f, t2w], dim=1) # type:ignore
    else:
        if whole.ndim != 4: raise ValueError("Полный тензор должен быть 4x96x96x4")
        inputs = whole.swapaxes(0, 1)
    inferer = SlidingWindowInfererAdapt(roi_size=(96, 96), sw_batch_size=32, overlap=0.25)
    preds1 = inferer(inputs, model1.inference)
    preds2 = inferer(torch.cat([inputs, preds1], 1), model2.inference) # type:ignore
    return to_binary(torch.nn.functional.softmax(preds2, 0), 0.5).swapaxes(0,1) # type:ignore

# 155, 240, 240

def inference_whole_brain(whole: Optional[torch.Tensor] = None,
    t1c: Optional[torch.Tensor] = None,
    t1n: Optional[torch.Tensor] = None,
    t2f: Optional[torch.Tensor] = None,
    t2w: Optional[torch.Tensor] = None,):
    """Вывод только на полных изображениях!"""
    model1 = get_first_model()
    model2 = get_second_model()
    from monai.inferers import SlidingWindowInfererAdapt # type:ignore
    if whole is None:
        if any([i is None for i in (t1c, t1n, t2f, t2w)]): raise ValueError("Не все модальности.")
        if t1c.ndim == 2: # type:ignore
            resizer = v2.Resize((240, 240))
            inputs = torch.stack([resizer(t1c.unsqueeze(0)), resizer(t1n.unsqueeze(0)), resizer(t2f.unsqueeze(0)), resizer(t2w.unsqueeze(0))], dim=1) # type:ignore
        elif t1c.ndim == 3:  # type:ignore
            fist_dim = max([i.size(0) for i in [t1c, t1n, t2f, t2w]]) # type:ignore
            resizer = monai.transforms.Resize((fist_dim, 240,240)) # type:ignore
            inputs = torch.stack([resizer(t1c), resizer(t1n), resizer(t2f), resizer(t2w)], dim=1) # type:ignore
        else: raise ValueError(f"Модальности должны быть 3D или 4D, получены {[i.size() for i in [t1c, t1n, t2f, t2w]]}") # type:ignore
    else:
        if whole.ndim == 4:
            resizer = monai.transforms.Resize((whole.size(1), 240,240)) # type:ignore
            inputs = resizer(whole).swapaxes(0, 1)
        elif whole.ndim == 3:
            resizer = v2.Resize((240, 240))
            inputs = resizer(whole).unsqueeze(0)
        else: raise ValueError(f"Полный тензор должен быть 3D или 4D, получен {whole.size()}")
    inferer = SlidingWindowInfererAdapt(roi_size=(96, 96), sw_batch_size=32, overlap=0.25)
    preds1 = inferer(inputs, model1.inference)
    preds2 = inferer(torch.cat([inputs, preds1], 1), model2.inference) # type:ignore
    return to_binary(torch.nn.functional.softmax(preds2, 0), 0.5).swapaxes(0,1) # type:ignore