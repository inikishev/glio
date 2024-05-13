import torch.nn

def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    batch_norm=False,
    dropout=0.,
    act=None,
    pool = None,
    ndim=2,
):
    """Свёрточный блок с пакетной нормализацией и исключением"""
    if ndim == 1:
        Convnd = torch.nn.Conv1d
        BatchNormnd = torch.nn.BatchNorm1d
        Dropoutnd = torch.nn.Dropout1d
    elif ndim == 2:
        Convnd = torch.nn.Conv2d
        BatchNormnd = torch.nn.BatchNorm2d
        Dropoutnd = torch.nn.Dropout2d
    elif ndim == 3:
        Convnd = torch.nn.Conv3d
        BatchNormnd = torch.nn.BatchNorm3d
        Dropoutnd = torch.nn.Dropout3d
    else: raise NotImplementedError

    # Список модулей со 3D свёрточным модулем
    modules:list[torch.nn.Module] = [Convnd(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]

    # Функция активации
    if act is not None: modules.append(act)

    # Функция пулинга
    if pool is not None: modules.append(pool)

    # Пакетная нормализация
    if batch_norm is True: modules.append(BatchNormnd(out_channels))

    # Исключение
    if dropout is not None and dropout != 0: modules.append(Dropoutnd(dropout))

    # Возвращается Sequential c распакованным списком модулей
    return torch.nn.Sequential(*modules)

def linear(in_features, out_features, bias = True, batch_norm = False, dropout = 0, act = None):
    """Линейный блок с пакетной нормализацией и исключением"""
    # Список модулей со 3D свёрточным модулем
    modules:list[torch.nn.Module]  = [torch.nn.Linear(in_features, out_features, bias)]

    # Функция активации
    if act is not None: modules.append(act)

    # Пакетная нормализация
    if batch_norm is True: modules.append(torch.nn.BatchNorm1d(out_features))

    # Исключение
    if dropout is not None and dropout != 0: modules.append(torch.nn.Dropout(dropout))

    # Возвращается Sequential c распакованным списком модулей
    return torch.nn.Sequential(*modules)


def accelerate(model: torch.nn.Module, *args):
    """Ускорение обучения с помощью `Hugging Face Accelerate`

    Принимает `model`, `optimizer`, `scheduler`, `dl_train`, `dl_test`.

    Возвращает `acc`, `model`, `*args`"""
    from accelerate import Accelerator
    acc = Accelerator()
    model = model.to(acc.device)
    (model, *args) = acc.prepare(model, *args)
    return acc, model, *args

def to_device(*args):
    """Перемещает данные на GPU"""
    return [x.to("cuda") for x in args if hasattr(x, "to")]

import numpy as np
def metrics_for_plotting(logger, metrics, max_length = 8192, smooth: list[float] = None):
    metrics = [logger[metric] for metric in metrics if metric in logger]
    metrics = [i for i in metrics if len(i) > 0]
    if len(metrics) > 0:
        reduction = [max(int(len(metric) / max_length), 1) for metric in metrics]
        metrics = [([list(m.keys())[::reduction[i]], list(m.values())[::reduction[i]]] if reduction[i]>1 else [list(m.keys()), list(m.values())]) for i, m in enumerate(metrics)]
        if smooth:
            for i in range(len(metrics)):
                if smooth[i] is not None and smooth[i] != 1 and smooth[i] != 0 and len(metrics[i][1]) > smooth[i]:
                    metrics[i][1] = np.convolve(metrics[i][1], np.ones(smooth[i])/smooth[i], 'same') # pyright:ignore
    return metrics

from ...python_tools import identity_kwargs, Wrapper
def train_loop(model: torch.nn.Module, func, dl_train, dl_test, epochs: int, bar = False,
            logger = None, step_batch = 16, step_epoch = None, metrics = ['train loss', 'test loss'],
            max_length = 8192, smooth: list[float] = None, test_first = True):
    """Цикл обучения. """
    # если модели - список или кортеж, для возможности хранения аттрибутов используется wrapper
    if isinstance(model, (list, tuple, dict)): model = Wrapper(model) # pyright:ignore

    if bar: from fastprogress.fastprogress import master_bar, progress_bar
    else: master_bar, progress_bar = identity_kwargs, identity_kwargs

    model.total_batch = 0 # pyright:ignore
    mb = master_bar(range(epochs))

    # train for n_epochs
    for model.cur_epoch in mb: # pyright:ignore
        # test first
        if test_first and model.cur_epoch == 0:
            model.eval()
            with torch.no_grad():
                for model.cur_batch, batch in enumerate(progress_bar(dl_test, leave=False, parent=mb)): # pyright:ignore
                    batch = to_device(*batch)
                    func(model = model, batch = batch, train = False)
        # train
        model.train()
        for model.cur_batch, batch in enumerate(progress_bar(dl_train, leave=False, parent=mb)): # pyright:ignore
            model.total_batch += 1
            batch = to_device(*batch)
            func(model = model, batch = batch, train = True)
            if logger is not None:
                if (step_batch is not None and model.total_batch % step_batch == 0) or (step_epoch is not None and model.total_batch % step_epoch == 0):
                    mb.update_graph(metrics_for_plotting(logger, metrics, max_length, smooth)) # pyright:ignore
        # test
        model.eval()
        with torch.no_grad():
            for model.cur_batch, batch in enumerate(progress_bar(dl_test, leave=False, parent=mb)): # pyright:ignore
                batch = to_device(*batch)
                func(model = model, batch = batch, train = False)

