"""datasets"""
try: from typing import Callable, Optional, Iterable, Sequence, Any, Self, final
except ImportError: from typing_extensions import Callable, Optional, Iterable, Sequence, Any, Self, final

import random
import statistics
from ...python_tools import (
    identity_if_none,
    call_if_callable,
    get_all_files,
    ItemUnderIndex,
    ItemUnder2Indexes,
    ExhaustingIterator,
    Compose,
    identity,
    identity_kwargs_if_none,
    get1,
)


class SampleBasic:
    def __init__(self, data, loader: Optional[Callable],transform: Optional[Callable]) -> None:
        self.data = data
        self.loader = identity_if_none(loader)
        self.transform = identity_if_none(transform)

        self.preloaded = None

    def __call__(self) -> tuple:
        """Returns loaded and transformed sample:
        ```
        self.transform(self.loader(self.sample))
        ```"""
        if self.preloaded is not None:
            return self.transform(self.preloaded)
        return self.transform(self.loader(call_if_callable(self.data)))

    def preload(self) -> None:
        self.preloaded = self.loader(call_if_callable(self.data))

    def unload(self) -> None:
        del self.preloaded
        self.preloaded = None

    def copy(self):
        sample = SampleBasic(data=self.data, loader=self.loader, transform=self.transform)
        sample.preloaded = self.preloaded
        return sample

    def set_loader(self, loader: Optional[Callable]):
        self.loader = identity_if_none(loader)

    def add_loader(self, loader: Callable):
        if self.loader is identity:
            self.loader = loader
        else:
            # try to add new loader to existing loader container such as torch.nn.Sequential
            try:
                self.loader = self.loader + loader # type: ignore
            # if it doesn't work, use Compose
            except (TypeError, ValueError):
                self.loader = Compose(self.loader, loader)

    def set_transform(self, transform: Optional[Callable]):
        self.transform = identity_if_none(transform)

    def add_transform(self, transform: Callable):
        if self.transform is identity:
            self.transform = transform
        else:
            # try to add new transform to existing transform container such as torch.nn.Sequential
            try:
                self.transform = self.transform + transform  # type: ignore
            # if it doesn't work, use Compose
            except (TypeError, ValueError, NotImplementedError):
                self.transform = Compose(self.transform, transform)


class DSBasic:
    def __init__(self):
        self.samples: list[SampleBasic] = []
        self.iter_cursor = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_cursor < len(self.samples):
            self.iter_cursor += 1
            return self[self.iter_cursor - 1]
        self.iter_cursor = 0
        raise StopIteration

    def copy(self, copy_samples=True) -> "Self":
        ds = type(self)()
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()
        return ds

    def shuffle(self):
        random.shuffle(self.samples)

    def add_sample(self,data,loader: Optional[Callable] = None, transform: Optional[Callable] = None):
        self.samples.append(SampleBasic(data=data, loader=loader, transform=transform))

    def add_samples(self,data: Iterable,loader: Optional[Callable] = None,transform: Optional[Callable] = None):
        self.samples.extend([SampleBasic(data=d, loader=loader, transform=transform) for d in data])

    def add_folder(
        self,
        path: str,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ):
        self.add_samples(
            data=get_all_files(path=path, recursive=recursive, extensions=extensions, path_filter=path_filt),
            loader=loader,
            transform=transform,
        )

    @staticmethod
    def from_folder(
        path: str,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ) -> "DSBasic":
        ds = DSBasic()
        ds.add_folder(
            path=path,
            loader=loader,
            transform=transform,
            recursive=recursive,
            extensions=extensions,
            path_filt=path_filt,
        )
        return ds

    def merge(self, ds: "DSBasic"):
        self.samples.extend(ds.samples)

    def add_external_dataset(self,
        dataset: Sequence,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        n_elems=None,
        auto_get_zero = True
    ):
        # check if dataset returns (sample, class_index)
        test_sample = dataset[0]
        auto_get_zero = auto_get_zero and isinstance(test_sample, (tuple, list)) and len(test_sample) == 2

        if n_elems is None:
            n_elems = len(dataset)
        else: n_elems = min(n_elems, len(dataset))

        if auto_get_zero: samples = [ItemUnder2Indexes(obj=dataset, index1=i, index2=0) for i in range(n_elems)]
        else: samples = [ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)]

        self.add_samples(
            data=samples,
            loader=loader,
            transform=transform,
        )

    def set_loader(self, loader: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_loader(loader)

    def add_loader(self,loader: Callable, sample_filter: Optional[Callable] = None, identical=False):
        # new loader variable for when loaders are identical
        new_loader = None
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                # if new loader was generated from the first sample, just assign it
                if new_loader is not None:
                    sample.set_loader(new_loader)
                # else identical is false or new_loader is not yet generated
                else:
                    sample.add_loader(loader)
                    if identical and new_loader is None:
                        new_loader = sample.loader

    def set_transform(self, transform: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform(transform)

    def add_transform(self, transform: Callable, sample_filter: Optional[Callable] = None, identical=False):
        # new loader variable for when loaders are identical
        new_transform = None
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                # if new transform was generated from the first sample, just assign it
                if new_transform is not None:
                    sample.set_transform(new_transform)
                # else identical is False or new_transform is not yet generated
                else:
                    sample.add_transform(transform)
                    if identical and new_transform is None:
                        new_transform = sample.transform

    def preload(self):
        for sample in self.samples:
            sample.preload()

    def unload(self):
        for sample in self.samples:
            sample.unload()

    def split(self, *splits, shuffle=True, copy_samples=False) -> list["DSBasic | DSClassification"]:
        # avoid shuffling self if shuffle is False
        if shuffle:
            ds = self.copy(copy_samples=False)
            ds.shuffle()
        else:
            ds = self

        dataset_length = len(ds)

        # parse args
        if len(splits) == 1:
            if isinstance(splits[0], tuple):
                splits = splits[0]
            elif isinstance(splits[0], int):
                splits = (splits[0], dataset_length - splits[0])
            elif isinstance(splits[0], float):
                splits = (splits[0], 1 - splits[0])
            else:
                raise ValueError(f"Invalid splits argument {splits}")

        # parse splits
        if isinstance(splits[0], float):
            if sum(splits) != 1:
                raise ValueError(
                    f"Sum of float splits must be 1, but it is {sum(splits)}. Your splits are `{splits}`"
                )
            splits = [int(i * dataset_length) for i in splits]
            splits[-1] += dataset_length - sum(splits)

        # check if sum of splits is equal to dataset length
        elif isinstance(splits[0], int):
            if sum(splits) != dataset_length:
                raise ValueError(
                    f"Sum of integer splits must be equal to dataset length which is {dataset_length}, but it is {sum(splits)}. Your splits are `{splits}`"
                )

        split_datasets: list[DSBasic | DSClassification] = []
        cur = 0

        # iterate over splits
        for i in splits:
            # get samples
            samples = ds.samples[cur : min(cur + i, dataset_length)]
            cur += i

            # create a mew dataset
            new_ds = ds.copy(copy_samples=False)

            # set samples
            if copy_samples:
                new_ds.samples = [s.copy() for s in samples]
            else:
                new_ds.samples = samples

            # add dataset
            split_datasets.append(new_ds)

        return split_datasets


    def length_iterator(self, length: int, shuffle: bool = True):
        """
        Returns a new dataset that always uses all elements from this dataset evenly while behaving like it has a different number of elements specified in `length`.

        This can be used to make epoch lengths unified for different size datasets.
        """
        return ExhaustingIterator(self, length=length, shuffle=shuffle)

    def get_mean_std(self, n_samples, batch_size, num_workers = 0, shuffle=True) -> tuple:
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
            ds, # type: ignore
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
            ds, # type: ignore
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


# _________________ CLASSIFICATION ________________________


def class_index(cls, class_to_int: dict[Any, int]) -> int:
    return class_to_int[cls]


def multiclass_index(classes: Sequence, class_to_int: dict[Any, int]) -> list[int]:
    return [class_to_int[cls] for cls in classes]


def auto_index(classes, class_to_int: dict[Any, int]) -> int | list[int]:
    if isinstance(classes, (list, tuple)):
        return multiclass_index(classes, class_to_int)
    return class_index(classes, class_to_int)


def one_hot(cls, class_to_int: dict[Any, int]) -> list[int]:
    zeros = [0 for _ in range(len(class_to_int))]
    zeros[class_to_int[cls]] = 1
    return zeros


def one_hot_multiclass(classes: Sequence, class_to_int: dict[Any, int]) -> list[int]:
    return [(0 if cls in classes else 1) for cls in class_to_int]


def one_hot_auto(classes, class_to_int: dict[Any, int]) -> list[int]:
    if isinstance(classes, (list, tuple)):
        return one_hot_multiclass(classes, class_to_int)
    return one_hot(classes, class_to_int)


class SampleClassification(SampleBasic):
    def __init__( # pylint: disable=W0231
        self,
        data,
        target: Callable | str | int,
        loader: Optional[Callable],
        transform: Optional[Callable],
        target_encoder: Optional[Callable],
    ) -> None:
        self.data = data

        # assign label, call on data if callable
        self.set_target(target)

        self.loader = identity_if_none(loader)
        self.transform = identity_if_none(transform)
        self.target_encoder = identity_kwargs_if_none(target_encoder)

        self.preloaded = None

    def __call__(self, target_to_idx: dict[Any, int]) -> tuple:
        """Returns a two element tuple:
        ```python
        (
            self.transform(self.loader(self.sample)),
            self.target_encoder(self.target, dataset.target_to_idx)
        )
        ```,
        where `dataset.target_to_idx` is a dictionary mapping from target to integer index, like `{'dog': 0, 'cat': 1, 'car': 2}`"""
        if self.preloaded is not None:
            return self.transform(self.preloaded), self.target_encoder(self.target, target_to_idx)
        return self.transform(self.loader(call_if_callable(self.data))), self.target_encoder(self.target, target_to_idx)

    def copy(self):
        sample = SampleClassification(
            data=self.data,
            target=self.target,
            loader=self.loader,
            transform=self.transform,
            target_encoder=self.target_encoder,
        )
        sample.preloaded = self.preloaded
        return sample

    def set_target(self, target: Callable | str | int) -> None:
        if callable(target):
            target = target(call_if_callable(self.data))
        self.target = target

    def set_target_encoder(self, target_encoder: Optional[Callable]):
        self.target_encoder = identity_kwargs_if_none(target_encoder)


class TargetFromDataset:
    pass


class DSClassification(DSBasic):
    def __init__(self): # pylint: disable=W0231
        self.samples: list[SampleClassification] = [] # type: ignore
        self.iter_cursor = 0

        self.targets: list = []
        """List of targets, e.g. `['dog', 'cat', 'car']`"""

        self.target_to_idx: dict[Any, int] = {}
        """Mapping from target to integer index, e.g. `{'dog': 0, 'cat': 1, 'car': 2}`"""

        self.idx_to_target: dict[int, Any] = {}
        """Mapping from integer index to target, e.g. `{0: 'dog', 1: 'cat', 2: 'car'}`"""

    def __getitem__(self, index: int) -> tuple:
        return self.samples[index](self.target_to_idx)

    def copy(self, copy_samples=True) -> "DSClassification":
        ds = DSClassification()
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()

        ds.targets = self.targets.copy()
        ds.target_to_idx = self.target_to_idx.copy()
        ds.idx_to_target = self.idx_to_target.copy()

        return ds

    @property
    def classes(self): return self.targets

    def set_target_encoder(self, target_encoder: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_target_encoder(target_encoder)

    def add_target(self, cls):
        if cls not in self.target_to_idx:
            self.targets.append(cls)
            self.target_to_idx[cls] = len(self.target_to_idx)
            self.idx_to_target[self.target_to_idx[cls]] = cls

    def add_targets(self, targets: list):
        for cls in targets:
            self.add_target(cls)

    def sort_targets(self, key=identity):
        """Sorts target keys"""
        self.target_to_idx = {k: i for i, k in enumerate(sorted(self.targets, key=key))}
        self.targets = list(self.target_to_idx.keys())
        self.idx_to_target = {v: k for k, v in self.target_to_idx.items()}

    def merge_targets(self, targets: list[str], new_target: str = None):
        """Merges all targets in `targets` into a single one called `new_target` or the first target in `targets`"""
        if new_target is None:
            new_target = targets[0]
        for i in targets:
            self.targets.remove(i)
        self.add_target(new_target)
        for s in self.samples:
            if s.target in targets:
                s.target = new_target

    def get_samples_per_target(self, sort=True):
        targets = [s.target for s in self.samples]
        samples_per_target = {cls: targets.count(cls) for cls in self.targets}
        if sort:
            samples_per_target = {
                k: v
                for k, v in sorted(
                    samples_per_target.items(), key=lambda item: item[1], reverse=True
                )
            }
        return samples_per_target

    def balance_targets(self, copy_samples=False, mode="copy", max_samples=None):
        """Some samples will be duplicated so that each target has as many samples as the target with the highest number of samples in the dataset. Tries to use samples evenly"""
        n_samples = max(list(self.get_samples_per_target().values()))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        for target in self.targets:
            if mode == "copy":
                while self.get_samples_per_target()[target] < n_samples:
                    samples = [i for i in self.samples if i.target == target][
                        : n_samples - self.get_samples_per_target()[target]
                    ]
                    if copy_samples:
                        samples = [i.copy() for i in samples]
                    self.samples.extend(samples)

                while self.get_samples_per_target()[target] > n_samples:
                    samples = [i for i in self.samples if i.target == target][
                        : self.get_samples_per_target()[target] - n_samples
                    ]
                    for i in samples:
                        self.samples.remove(i)

            else:
                raise NotImplementedError

    def add_sample( # pylint: disable=W0237
        self,
        data,
        target: Callable | str | int,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_encoder: Optional[Callable] = auto_index,
    ):
        sample = SampleClassification(
            data=data,
            target=target,
            loader=loader,
            transform=transform,
            target_encoder=target_encoder,
        )
        self.samples.append(sample)
        self.add_target(sample.target)

    def add_samples( # pylint: disable=W0237
        self,
        data: Iterable,
        target: Callable | str | int,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_encoder: Optional[Callable] = auto_index,
    ):
        samples = [
            SampleClassification(
                data=d,
                target=target,
                loader=loader,
                transform=transform,
                target_encoder=target_encoder,
            )
            for d in data
        ]
        self.samples.extend(samples)
        self.add_targets(list(set([s.target for s in samples])))

    def add_folder( # pylint: disable=W0237
        self,
        path: str,
        target: Callable | str | int,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_encoder: Optional[Callable] = auto_index,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ):
        self.add_samples(
            data=get_all_files(
                path=path,
                recursive=recursive,
                extensions=extensions,
                path_filter=path_filt,
            ),
            target=target,
            loader=loader,
            transform=transform,
            target_encoder=target_encoder,
        )

    @staticmethod
    def from_folder( # pylint: disable=W0237
        path: str,
        target: Callable | str | int,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_encoder: Optional[Callable] = auto_index,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ) -> "DSClassification":
        ds = DSClassification()
        ds.add_folder(
            path=path,
            target=target,
            loader=loader,
            transform=transform,
            target_encoder=target_encoder,
            recursive=recursive,
            extensions=extensions,
            path_filt=path_filt,
        )
        return ds

    def merge(self, ds: "DSClassification"): 
        self.samples.extend(ds.samples)
        self.add_targets(list(set([s.target for s in ds.samples])))

    def add_external_dataset( # pylint: disable=W0237
        self,
        dataset: Sequence,
        target: Callable | str | int | TargetFromDataset = TargetFromDataset(),
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_encoder: Optional[Callable] = auto_index,
        target_attr: str = "classes",
        n_elems=None,
        auto_get_zero: bool = True,
    ):
        # check if dataset returns (sample, target_index)
        test_sample = dataset[0]
        returns_sample_and_target = (isinstance(test_sample, (tuple, list)) and len(test_sample) == 2)

        # if dataset doesn't return (sample, target_index), no need to execute auto_get_zero
        auto_get_zero = returns_sample_and_target and auto_get_zero

        # if dataset returns (sample, target_index), get target from dataset
        if returns_sample_and_target:
            # get targets attribute of the dataset
            if target_attr is not None:
                targets = getattr(dataset, target_attr)

            # if target needs to be generated from dataset, generate target function
            if isinstance(target, TargetFromDataset):
                # if we know targets, we assume dataset returns [sample, target_index]; we get targets[target_index]
                if target_attr is not None:
                    target = lambda x: targets[x[1]]
                # else we just get the second element returned by dataset __getindex__
                else:
                    target = get1

        # if dataset doesn't return (sample, target_index), ClassFromDataset is not valid
        elif isinstance(target, TargetFromDataset):
            raise ValueError(
                f"`TargetFromDataset` requires dataset to return `(sample, target_index)`, but it returns `{type(test_sample)}`"
            )

        # get number of elements
        if n_elems is None:
            n_elems = len(dataset)
        n_elems = min(n_elems, len(dataset))

        # create samples
        if auto_get_zero: samples = [ItemUnder2Indexes(obj=dataset, index1=i, index2=0) for i in range(n_elems)]
        else: samples = [ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)]

        self.add_samples(
            data=samples,
            target=target,
            loader=loader,
            transform=transform,
            target_encoder=target_encoder,
        )


class SampleToTarget(SampleBasic):
    def __init__( # pylint: disable=W0231
        self,
        data,
        loader: Optional[Callable],
        transform_init: Optional[Callable],
        transform_sample: Optional[Callable],
        transform_target: Optional[Callable],
    ) -> None:

        self.data = data
        self.loader = identity_if_none(loader)
        self.transform_init = identity_if_none(transform_init)
        self.transform_sample = identity_if_none(transform_sample)
        self.transform_target = identity_if_none(transform_target)

        self.preloaded = None

    def __call__(self) -> tuple:
        """Returns a tuple with loaded and transformed sample:
        ```
        (
            self.transform_sample(self.transform_init(self.loader(self.sample))),
            self.transform_target(self.transform_init(self.loader(self.sample)))
        )
        ```"""
        if self.preloaded is not None:
            return (
            self.transform_sample(
                self.transform_init(self.preloaded)
            ),
            self.transform_target(
                self.transform_init(self.preloaded)
            )
        )
        return (
            self.transform_sample(
                self.transform_init(self.loader(call_if_callable(self.data)))
            ),
            self.transform_target(
                self.transform_init(self.loader(call_if_callable(self.data)))
            ),
        )

    def copy(self):
        sample = SampleToTarget(
            data = self.data,
            loader = self.loader,
            transform_init = self.transform_init,
            transform_sample = self.transform_sample,
            transform_target = self.transform_target,
        )
        sample.preloaded = self.preloaded
        return sample

    def set_transform_init(self, transform_init: Optional[Callable]):
        self.transform_init = identity_if_none(transform_init)

    def add_transform_init(self, transform_init: Callable):
        if self.transform_init is identity:
            self.transform_init = transform_init
        else:
            # try to add new transform to existing transform container such as torch.nn.Sequential
            try:
                self.transform_init = self.transform_init + transform_init # type: ignore
            # if it doesn't work, use Compose
            except (TypeError, ValueError, NotImplementedError):
                self.transform_init = Compose(self.transform_init, transform_init)

    def set_transform_sample(self, transform_sample: Optional[Callable]):
        self.transform_sample = identity_if_none(transform_sample)

    def add_transform_sample(self, transform_sample: Callable):
        if self.transform_sample is identity:
            self.transform_sample = transform_sample
        else:
            # try to add new transform to existing transform container such as torch.nn.Sequential
            try:
                self.transform_sample = self.transform_sample + transform_sample # type: ignore
            # if it doesn't work, use Compose
            except (TypeError, ValueError, NotImplementedError):
                self.transform_sample = Compose(self.transform_sample, transform_sample)

    def set_transform_target(self, transform_target: Optional[Callable]):
        self.transform_target = identity_if_none(transform_target)

    def add_transform_target(self, transform_target: Callable):
        if self.transform_target is identity:
            self.transform_target = transform_target
        else:
            # try to add new transform to existing transform container such as torch.nn.Sequential
            try:
                self.transform_target = self.transform_target + transform_target # type: ignore
            # if it doesn't work, use Compose
            except (TypeError, ValueError, NotImplementedError):
                self.transform_target = Compose(self.transform_target, transform_target)

    def set_transform(self, *args, **kwargs): raise NotImplementedError("use `set_transform_init`, `set_transform_sample` and `set_transform_target` instead")
    def add_transform(self, *args, **kwargs): raise NotImplementedError("use `add_transform_init`, `add_transform_sample` and `add_transform_target` instead")


class DSToTarget(DSBasic):
    def __init__(self): # pylint: disable=W0231
        self.samples: list[SampleToTarget] = [] # type: ignore
        self.iter_cursor = 0

    def copy(self, copy_samples=True) -> "DSToTarget":
        ds = DSToTarget()
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()
        return ds

    def add_sample( # pylint: disable=W0237
        self,
        data,
        loader: Optional[Callable] = None,
        transform_init: Optional[Callable] = None,
        transform_sample: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
    ):
        self.samples.append(
            SampleToTarget(
                data=data,
                loader=loader,
                transform_init=transform_init,
                transform_sample=transform_sample,
                transform_target=transform_target,
            )
        )

    def add_samples( # pylint: disable=W0237
        self,
        data,
        loader: Optional[Callable] = None,
        transform_init: Optional[Callable] = None,
        transform_sample: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
    ):
        self.samples.extend(
            [
                SampleToTarget(
                    data=d,
                    loader=loader,
                    transform_init=transform_init,
                    transform_sample=transform_sample,
                    transform_target=transform_target,
                )
                for d in data
            ]
        )

    def add_folder( # pylint: disable=W0237
        self,
        path: str,
        loader: Optional[Callable] = None,
        transform_init: Optional[Callable] = None,
        transform_sample: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ):
        self.add_samples(
            data=get_all_files(
                path=path,
                recursive=recursive,
                extensions=extensions,
                path_filter=path_filt,
            ),
            loader=loader,
            transform_init=transform_init,
            transform_sample=transform_sample,
            transform_target=transform_target,
        )

    @staticmethod
    def from_folder( # pylint: disable=W0237
        path: str,
        loader: Optional[Callable] = None,
        transform_init: Optional[Callable] = None,
        transform_sample: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ) -> "DSToTarget":
        ds = DSToTarget()
        ds.add_folder(
            path=path,
            loader=loader,
            transform_init=transform_init,
            transform_sample=transform_sample,
            transform_target=transform_target,
            recursive=recursive,
            extensions=extensions,
            path_filt=path_filt,
        )
        return ds

    def add_external_dataset(self, # pylint: disable=W0237
        dataset: Sequence,
        loader: Optional[Callable] = None,
        transform_init: Optional[Callable] = None,
        transform_sample: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        n_elems=None,
        auto_get_zero = True
    ):
        # check if dataset returns (sample, class_index)
        test_sample = dataset[0]
        auto_get_zero = auto_get_zero and isinstance(test_sample, (tuple, list)) and len(test_sample) == 2

        if n_elems is None:
            n_elems = len(dataset)
        else: n_elems = min(n_elems, len(dataset))

        if auto_get_zero: samples = [ItemUnder2Indexes(obj=dataset, index1=i, index2=0) for i in range(n_elems)]
        else: samples = [ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)]

        self.add_samples(
            data=samples,
            loader=loader,
            transform_init=transform_init,
            transform_sample=transform_sample,
            transform_target=transform_target,
        )

    def set_loader(self, loader: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_loader(loader)

    def add_loader(self,loader: Callable, sample_filter: Optional[Callable] = None, identical=False):
        # new loader variable for when loaders are identical
        new_loader = None
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                # if new loader was generated from the first sample, just assign it
                if new_loader is not None:
                    sample.set_loader(new_loader)
                # else identical is false or new_loader is not yet generated
                else:
                    sample.add_loader(loader)
                    if identical and new_loader is None:
                        new_loader = sample.loader

    def set_transform(self, *args, **kwargs): raise NotImplementedError("use `set_transform_init`, `set_transform_sample` and `set_transform_target` instead")
    def add_transform(self, *args, **kwargs): raise NotImplementedError("use `add_transform_init`, `add_transform_sample` and `add_transform_target` instead")

    def set_transform_init(self, transform_init: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform_init(transform_init)

    def add_transform_init(self, transform_init: Callable, sample_filter: Optional[Callable] = None, identical=False):
        # new loader variable for when loaders are identical
        new_transform_init = None
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                # if new transform was generated from the first sample, just assign it
                if new_transform_init is not None:
                    sample.set_transform_init(new_transform_init)
                # else identical is False or new_transform is not yet generated
                else:
                    sample.add_transform_init(transform_init)
                    if identical and new_transform_init is None:
                        new_transform_init = sample.transform_init

    def set_transform_sample(self, transform_sample: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform_sample(transform_sample)

    def add_transform_sample(self, transform_sample: Callable, sample_filter: Optional[Callable] = None, identical=False):
        # new loader variable for when loaders are identical
        new_transform_sample = None
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                # if new transform was generated from the first sample, just assign it
                if new_transform_sample is not None:
                    sample.set_transform_sample(new_transform_sample)
                # else identical is False or new_transform is not yet generated
                else:
                    sample.add_transform_sample(transform_sample)
                    if identical and new_transform_sample is None:
                        new_transform_sample = sample.transform_sample

    def set_transform_target(self, transform_target: Optional[Callable], sample_filter: Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform_target(transform_target)

    def add_transform_target(self, transform_target: Callable, sample_filter: Optional[Callable] = None, identical=False):
        # new loader variable for when loaders are identical
        new_transform_target = None
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                # if new transform was generated from the first sample, just assign it
                if new_transform_target is not None:
                    sample.set_transform_target(new_transform_target)
                # else identical is False or new_transform is not yet generated
                else:
                    sample.add_transform_target(transform_target)
                    if identical and new_transform_target is None:
                        new_transform_target = sample.transform_target



# __________________________ REGRESSION _____________________________

class SampleRegression(SampleBasic):
    def __init__(self, data, target: Callable | float | int, loader: Optional[Callable], transform: Optional[Callable]) -> None: # pylint: disable=W0231
        self.data = data

        # assign label, call on data if callable
        self.set_target(target)

        self.loader = identity_if_none(loader)
        self.transform = identity_if_none(transform)

        self.preloaded = None

    def __call__(self) -> tuple:
        """Returns a two element tuple:
        ```python
        (
            self.transform(self.loader(self.sample)),
            self.target(self.sample) | self.target
        )
        ```
        """
        if self.preloaded is not None:
            return self.transform(self.preloaded), self.target
        return self.transform(self.loader(call_if_callable(self.data))), self.target

    def copy(self):
        sample = SampleRegression(
            data=self.data,
            target=self.target,
            loader=self.loader,
            transform=self.transform,
        )
        sample.preloaded = self.preloaded
        return sample

    def set_target(self, target: Callable | int | float) -> None:
        if callable(target):
            target = target(call_if_callable(self.data))
        self.target: int | float = target # type: ignore


class DSRegression(DSBasic):
    def __init__(self): # pylint: disable=W0231
        self.samples: list[SampleRegression] = [] # type: ignore
        self.iter_cursor = 0

    def copy(self, copy_samples=True) -> "DSRegression":
        ds = DSRegression()
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()
        return ds

    def add_sample(self, data, target:Callable|float|int, loader: Optional[Callable] = None, transform: Optional[Callable] = None): # pylint: disable=W0237
        self.samples.append(SampleRegression(data=data, target=target, loader=loader, transform=transform))

    def add_samples(self, data:Iterable, target:Callable|float|int, loader: Optional[Callable] = None,transform: Optional[Callable] = None): # pylint: disable=W0237
        self.samples.extend([SampleRegression(data=d, target=target,loader=loader, transform=transform) for d in data])

    def add_folder( # pylint: disable=W0237
        self,
        path: str,
        target:Callable|float|int,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ):
        self.add_samples(
            data=get_all_files(path=path, recursive=recursive, extensions=extensions, path_filter=path_filt),
            target=target,
            loader=loader,
            transform=transform,
        )

    @staticmethod
    def from_folder( # pylint: disable=W0237
        path: str,
        target:Callable|float|int,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
    ) -> "DSRegression":
        ds = DSRegression()
        ds.add_folder(
            path=path,
            target=target,
            loader=loader,
            transform=transform,
            recursive=recursive,
            extensions=extensions,
            path_filt=path_filt,
        )
        return ds

    def add_external_dataset(self, # pylint: disable=W0237
        dataset: Sequence,
        target:Callable|float|int|TargetFromDataset = TargetFromDataset(),
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        n_elems=None,
        auto_get_zero = True
    ):
        # check if dataset returns (sample, target_index)
        test_sample = dataset[0]
        ds_returns_data_target = isinstance(test_sample, (tuple, list)) and len(test_sample) == 2

        if isinstance(target, TargetFromDataset) and not ds_returns_data_target:
            raise ValueError(f"`TargetFromDataset` can only be used when dataset returns `(sample, target_index)`; but it returns {type(test_sample)}")

        if n_elems is None:
            n_elems = len(dataset)
        else: n_elems = min(n_elems, len(dataset))


        if ds_returns_data_target:
            data_target = [
                (
                    ItemUnder2Indexes(obj=dataset, index1=i, index2=0),
                    dataset[i][1] if isinstance(target, TargetFromDataset) else target,
                )
                for i in range(n_elems)
            ]
            # if auto-get-zero, it is assumed that the dataset returns (data, target), so we can get data from index 0
            # target it either index 1 if TargetFromDataset is used, or whatever is passed into `target`
            if auto_get_zero:
                for data, target_ in data_target:
                    self.add_sample(data=data, target=target_, loader=loader, transform=transform)
            # otherwise data is the entire returned value of the dataset
            else:
                for i, (data, target_) in enumerate(data_target):
                    self.add_sample(data=ItemUnderIndex(obj=dataset, index=i), target=target_, loader=loader, transform=transform)

        else:
            if isinstance(target, TargetFromDataset):
                raise ValueError(f"`TargetFromDataset` can only be used when dataset returns `(sample, target_index)`; but it returns {type(test_sample)}")

            self.add_samples(
                data=[ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)],
                target=target,
                loader=loader,
                transform=transform,
            )

    def targets(self): return [s.target for s in self.samples]

    def get_targets_min(self):
        """Returns min of all targets"""
        return min(self.targets())

    def get_targets_max(self):
        """Returns max of all targets"""
        return max(self.targets())

    def get_targets_mean(self):
        """Returns mean of all targets"""
        return sum(self.targets())/len(self.targets())

    def get_targets_std(self):
        """Returns stdev of all targets"""
        return statistics.stdev(self.targets())

    def normalize_targets(self, mode = 'z-norm', min = 0, max = 1): # pylint: disable=W0622
        """Normalizes labels using z normalization if `mode` is `z`, or by fitting all labels to min-max range"""
        targets_mean = self.get_targets_mean()
        targets_std = self.get_targets_std()
        targets_min = self.get_targets_min()
        targets_max = self.get_targets_max()

        if mode.lower().startswith('z'):
            for sample in self.samples:
                if not isinstance(sample.target, (int,float)): raise ValueError(f"Expected int or float, got {type(sample.target)}")
                sample.target = (sample.target - float(targets_mean)) / float(targets_std)
        else:
            for sample in self.samples:
                if not isinstance(sample.target, (int,float)): raise ValueError(f"Expected int or float, got {type(sample.target)}")
                sample.target = sample.target - targets_min
                sample.target = sample.target / targets_max
                sample.target = sample.target * (max - min) + min
