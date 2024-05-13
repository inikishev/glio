from . import DS
from typing import Callable, Optional
from ..python_tools import call_if_callable, identity_if_none

import torch

class SampleBasic(DS.SampleBasic):
    pass

class DSBasic(DS.DSBasic, torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()

    def get_mean_std(self, n_samples, batch_size, num_workers = 0, shuffle=True) -> tuple[tuple[float], tuple[float]]:
        """
        Calculates per channel mean and std for `torchvision.transforms.Normalize`, pass it as Normalize(*result). Returns `Tensor(R.mean, G.mean, B.mean), Tensor(R.std, G.std. B.std)` Increasing `batch_size` MIGHT lead to faster processing, but that will only work if all images have the same size.
        """
        # copy self to avoid shuffling self if shuffle is True
        if shuffle:
            ds = (
                self.copy()
                if ((not n_samples) or n_samples >= len(self))
                else self.split(n_samples)[0].copy()
            )
            ds.shuffle()
        else:
            ds = (
                self
                if ((not n_samples) or n_samples >= len(self))
                else self.split(n_samples, shuffle=False)[0]
            )

        # test dimensions and class
        # create a test dataloader
        import torch.utils.data

        test_dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        # get a test batch
        batch = next(iter(test_dataloader))
        # check whether a batch is a collated list/tuple/dict
        has_class = True if isinstance(batch, (list, tuple, dict)) else False
        # if batch has class, batch is [samples, classes], and we need first sample which is samples[0], so we do batch[0][0]
        # else batch is [samples], so we do batch[0]
        # number of channels is the first dimension in the sample
        channels = batch[0][0].shape[0] if has_class else batch[0].shape[0]

        # create a dataloader for calculating mean and std
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        # create tensors with zero per channel
        mean = torch.zeros(channels)
        std = torch.zeros(channels)
        # iterate over dataloader
        for item in dataloader:
            # get sample
            sample = item[0] if has_class else item
            # get number of dimensions
            ndim = sample.ndim
            # get list of shape indexes
            channels = list(range(ndim))  # 0 1 2 3
            # remove the channel dimension from shape indexes
            channels.remove(1)  # 0 2 3
            # calculate mean and std over all dimensions except channel
            mean += sample.mean(dim=tuple(channels))
            std += sample.std(dim=tuple(channels))

        # calculate mean and std over all samples
        mean /= len(dataloader)
        std /= len(dataloader)
        return mean, std

class SampleClassification(DS.SampleClassification):
    pass

class DSClassification(DS.DSClassification, torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()

class SampleToTarget(DS.SampleToTarget):
    pass

class DSToTarget(DS.DSToTarget, torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()

class SampleRegression(DS.SampleRegression):
    def __init__(
        self,
        data,
        target: Callable | float | int,
        loader: Optional[Callable],
        transform: Optional[Callable],
        target_dtype=torch.float32,
    ) -> None:
        self.data = data
        self.target_dtype = target_dtype

        # assign label, call on data if callable
        self.set_target(target)

        self.loader = identity_if_none(loader)
        self.transform = identity_if_none(transform)

        self.preloaded = None

    def set_target(self, target: Callable | str | int) -> None:
        if callable(target):
            target = target(call_if_callable(self.data))
        self.target = torch.tensor(target, dtype=self.target_dtype)

class DSRegression(DS.DSRegression, torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()
    def targets(self) -> torch.Tensor: return torch.as_tensor([s.target for s in self.samples], dtype = torch.float32)
    def get_targets_min(self):
        """Returns min of all targets"""
        return self.targets().min()

    def get_targets_max(self):
        """Returns max of all targets"""
        return self.targets().max()

    def get_targets_mean(self):
        """Returns mean of all targets"""
        return self.targets().mean()

    def get_targets_std(self):
        """Returns stdev of all targets"""
        return self.targets().std()

    def normalize_targets(self, mode = 'z-norm', min = 0, max = 1):
        """Normalizes labels using z normalization if `mode` is `z`, or by fitting all labels to min-max range"""
        targets_mean = self.get_targets_mean()
        targets_std = self.get_targets_std()
        targets_min = self.get_targets_min()
        targets_max = self.get_targets_max()

        if mode.lower().startswith('z'):
            for sample in self.samples:
                sample.set_target((sample.target - targets_mean) / targets_std)
        else:
            for sample in self.samples:
                target = sample.target - targets_min
                target = target / targets_max
                sample.set_target(target * (max - min) + min)
