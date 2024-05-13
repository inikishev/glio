"""datasets"""
from typing import Callable, Optional, Iterable, Sequence, Any, no_type_check
from types import EllipsisType
import random
import concurrent.futures
import torch
# from torchvision.transforms import v2, transforms
from ..python_tools import (
    call_if_callable,
    get_all_files,
    ItemUnderIndex,
    ItemUnder2Indexes,
    identity,
    auto_compose,
    identity_kwargs_if_none,
    get1,
    get0
)
from ..visualize.visualize import Visualizer

from . import DSBase
from .DSBase import Composable

class SampleBasic(DSBase.Sample, DSBase.SampleWithTransform):
    def __init__(self, data, loader:Composable, transform:Composable) -> None:
        self.data = data
        self.loader = auto_compose(loader)
        self.transform = auto_compose(transform)

        self.preloaded = None

    def __call__(self):
        if self.preloaded is not None: return self.transform(self.preloaded)
        return self.transform(self.loader(self.data))

    def copy(self) -> "SampleBasic":
        sample = SampleBasic(self.data, self.loader, self.transform)
        sample.preloaded = self.preloaded
        return sample


class DSBasic(DSBase.DS, DSBase.DSWithTransform):
    def __init__(self, n_threads = 8):
        self.samples: list[SampleBasic] = []
        self.n_threads = n_threads

        self.iter_cursor = 0
        self.last_accessed = []

    def add_sample(self, data, loader: Composable = None, transform: Composable = None):
        self.samples.append(SampleBasic(data=data, loader = loader, transform = transform))

    def add_samples(self, data: Iterable, loader: Composable = None,transform: Composable = None):
        self.samples.extend([SampleBasic(data=d, loader = loader, transform = transform) for d in data])

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

    @classmethod
    def from_folder(
        cls,
        path: str,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
        n_threads = 8
    ) -> "DSBasic":
        ds = DSBasic(n_threads = n_threads)
        ds.add_folder(
            path=path,
            loader=loader,
            transform=transform,
            recursive=recursive,
            extensions=extensions,
            path_filt=path_filt,
        )
        return ds

    def add_external_dataset(self,
        dataset,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        n_elems=None,
        smart_get0 = True
    ):
        # check if dataset returns (sample, class_index)
        test_sample = dataset[0]
        smart_get0 = smart_get0 and isinstance(test_sample, (tuple, list)) and len(test_sample) == 2

        if n_elems is None:
            n_elems = len(dataset)
        else: n_elems = min(n_elems, len(dataset))

        if smart_get0: samples = [ItemUnder2Indexes(obj=dataset, index1=i, index2=0) for i in range(n_elems)]
        else: samples = [ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)]

        self.add_samples(
            data=samples,
            loader=loader,
            transform=transform,
        )

    @classmethod
    def from_external_dataset(cls,
        dataset,
        loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        n_elems=None,
        smart_get0 = True,
        n_threads = 8,
        ) -> "DSBasic":
        ds = DSBasic(n_threads = n_threads)
        ds.add_external_dataset(
            dataset=dataset,
            loader=loader,
            transform=transform,
            n_elems=n_elems,
            smart_get0=smart_get0,
        )
        return ds


# _________________ CLASSIFICATION ________________________


def class_index(cls, class_to_int: dict[Any, torch.Tensor]) -> torch.Tensor:
    return class_to_int[cls]


def multiclass_index(classes: Sequence, class_to_int: dict[Any, torch.Tensor]) -> list[torch.Tensor]:
    return [class_to_int[cls] for cls in classes]


def auto_index(classes, class_to_int: dict[Any, torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(classes, (list, tuple)):
        return multiclass_index(classes, class_to_int)
    return class_index(classes, class_to_int)


def one_hot(cls, class_to_int: dict[Any, torch.Tensor]) -> torch.Tensor:
    zeros = torch.zeros((len(class_to_int)), dtype=torch.int64)
    zeros[class_to_int[cls]] = 1
    return zeros


def one_hot_multiclass(classes: Sequence, class_to_int: dict[Any, torch.Tensor]) -> list[torch.Tensor]:
    return [(torch.tensor(0, dtype=torch.float32) if cls in classes else torch.tensor(1, dtype=torch.int64)) for cls in class_to_int]


def one_hot_auto(classes, class_to_int: dict[Any, torch.Tensor]) -> list[torch.Tensor] | torch.Tensor:
    if isinstance(classes, (list, tuple)):
        return one_hot_multiclass(classes, class_to_int)
    return one_hot(classes, class_to_int)


class SampleClassification(DSBase.Sample, DSBase.SampleWithTransform, DSBase.SampleWithTarget, DSBase.SampleWithTargetEncoder):
    def __init__(
        self,
        data,
        loader: Composable,
        transform: Composable,
        target: Callable | str | int,
        target_encoder: Optional[Callable],
    ) -> None:
        self.data = data

        # assign label, call on data if callable
        self.set_target(target)

        self.loader = auto_compose(loader)
        self.transform = auto_compose(transform)
        self.target_encoder = identity_kwargs_if_none(target_encoder)

        self.preloaded = None

    @no_type_check
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
            loader=self.loader,
            transform=self.transform,
            target=self.target,
            target_encoder=self.target_encoder,
        )
        sample.preloaded = self.preloaded
        return sample

    def set_target_encoder(self, target_encoder: Optional[Callable]):
        self.target_encoder = identity_kwargs_if_none(target_encoder)


class DSClassification(DSBase.DS, DSBase.DSWithTransform, DSBase.DSWithTargets, DSBase.DSWithTargetEncoder):
    def __init__(self, n_threads = 8, target_dtype = torch.int64):
        self.samples: list[SampleClassification] = []
        self.n_threads = n_threads

        self.last_accessed = []
        self.iter_cursor = 0

        self.targets: list = []
        """List of targets, e.g. `['dog', 'cat', 'car']`"""

        self.target_to_idx: dict[Any, torch.Tensor] = {}
        """Mapping from target to integer index, e.g. `{'dog': 0, 'cat': 1, 'car': 2}`"""

        self.idx_to_target: dict[torch.Tensor, Any] = {}
        """Mapping from integer index to target, e.g. `{0: 'dog', 1: 'cat', 2: 'car'}`"""

        self.target_dtype = target_dtype

    def __getitem__(self, index: int):
        if isinstance(index, int):
            self.last_accessed.append(index)
            return self.samples[index](self.target_to_idx)

    def __getitems__(self, indexes: Iterable[int]) -> list:
        self.last_accessed.extend(indexes)
        if self.n_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                return list(executor.map(lambda x: self.samples[x](self.target_to_idx), indexes))
        return [self.samples[i](self.target_to_idx) for i in indexes]

    def copy(self, copy_samples=True) -> "DSClassification":
        ds = DSClassification(n_threads=self.n_threads, target_dtype=self.target_dtype)
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()

        ds.targets = self.targets.copy()
        ds.target_to_idx = self.target_to_idx.copy()
        ds.idx_to_target = self.idx_to_target.copy()

        return ds

    def add_sample(
        self,
        data,
        loader: Composable = None,
        transform: Composable = None,
        target: Callable | Any = ...,
        target_encoder: Optional[Callable] = auto_index,
    ):
        if target is ...: raise ValueError("target must be specified")
        sample = SampleClassification(
            data=data,
            loader=loader,
            transform=transform,
            target=target,
            target_encoder=target_encoder,
        )
        self.samples.append(sample)
        self.update_target(sample.target)

    def add_samples(
        self,
        data: Iterable,
        loader: Composable = None,
        transform: Composable = None,
        target: Callable | Any = ...,
        target_encoder: Optional[Callable] = auto_index,
    ):
        samples = [
            SampleClassification(
                data=d,
                loader=loader,
                transform=transform,
                target=target,
                target_encoder=target_encoder,
            )
            for d in data
        ]
        self.samples.extend(samples)
        self.update_targets(list(set([s.target for s in samples])))

    def add_folder(
        self,
        path: str,
        loader: Composable = None,
        transform: Composable = None,
        target: Callable | Any = ...,
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

    @classmethod
    def from_folder(
        cls,
        path: str,
        loader: Composable = None,
        transform: Composable = None,
        target: Callable | Any = ...,
        target_encoder: Optional[Callable] = auto_index,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
        n_threads = 8,
        target_dtype = torch.int64,
    ) -> "DSClassification":
        ds = DSClassification(n_threads = n_threads, target_dtype=target_dtype)
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
        self.update_targets(list(set([s.target for s in ds.samples])))

    def add_external_dataset(
        self,
        dataset,
        loader: Composable = None,
        transform: Composable = None,
        target: Callable | Any | DSBase.AutoTargetFromDataset = DSBase.AutoTargetFromDataset(),
        target_encoder: Optional[Callable] = auto_index,
        target_attr: str = "classes",
        n_elems=None,
        smart_get0: bool = True,
    ):
        # check if dataset returns (sample, target_index)
        test_sample = dataset[0]
        returns_sample_and_target = (isinstance(test_sample, (tuple, list)) and len(test_sample) == 2)

        # if dataset doesn't return (sample, target_index), no need to execute auto_get_zero
        smart_get0 = returns_sample_and_target and smart_get0

        # if dataset returns (sample, target_index), get target from dataset
        if returns_sample_and_target:
            # get targets attribute of the dataset
            if target_attr is not None:
                targets = getattr(dataset, target_attr)

            # if target needs to be generated from dataset, generate target function
            if isinstance(target, DSBase.AutoTargetFromDataset):
                # if we know targets, we assume dataset returns [sample, target_index]; we get targets[target_index]
                if target_attr is not None:
                    target = lambda x: targets[x[1]]
                # else we just get the second element returned by dataset __getindex__
                else:
                    target = get1

        # if dataset doesn't return (sample, target_index), ClassFromDataset is not valid
        elif isinstance(target, DSBase.AutoTargetFromDataset):
            raise ValueError(
                f"`TargetFromDataset` requires dataset to return `(sample, target_index)`, but it returns `{type(test_sample)}`"
            )

        # get number of elements
        if n_elems is None:
            n_elems = len(dataset)
        n_elems = min(n_elems, len(dataset))

        # create samples
        if smart_get0:
            loader = DSBase.smart_compose(get0, auto_compose(loader))
        samples = [ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)]

        self.add_samples(
            data=samples,
            target=target,
            loader=loader,
            transform=transform,
            target_encoder=target_encoder,
        )

    @classmethod
    def from_external_dataset(cls,
        dataset,
        loader: Composable = None,
        transform: Composable = None,
        target: Callable | Any | DSBase.AutoTargetFromDataset = DSBase.AutoTargetFromDataset(),
        target_encoder: Optional[Callable] = auto_index,
        target_attr: str = "classes",
        n_elems=None,
        smart_get0: bool = True,
        n_threads = 8,
        target_dtype = torch.int64,
        ) -> "DSClassification":
        ds = DSClassification(n_threads=n_threads, target_dtype=target_dtype)
        ds.add_external_dataset(
            dataset=dataset,
            loader=loader,
            transform=transform,
            target = target,
            target_encoder=target_encoder,
            target_attr = target_attr,
            n_elems=n_elems,
            smart_get0=smart_get0,
        )
        return ds

    @property
    def classes(self): return self.targets

    def update_target(self, target):
        if target not in self.target_to_idx:
            self.targets.append(target)
            self.target_to_idx[target] = torch.tensor(len(self.target_to_idx), dtype=self.target_dtype)
            self.idx_to_target[self.target_to_idx[target]] = target

    def update_targets(self, targets: Sequence):
        for cls in targets:
            self.update_target(cls)

    def sort_targets(self, key=identity):
        """Sorts target keys"""
        self.target_to_idx = {k: torch.tensor(i, dtype=self.target_dtype) for i, k in enumerate(sorted(self.targets, key=key))}
        self.targets = list(self.target_to_idx.keys())
        self.idx_to_target = {v: k for k, v in self.target_to_idx.items()}

    def merge_targets(self, targets: list, new_target: Optional[Any] = None):
        """Merges all targets in `targets` into a single one called `new_target` or the first target in `targets`"""
        if new_target is None:
            new_target = targets[0]
        for i in targets:
            self.targets.remove(i)
        self.update_target(new_target)
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
        """Some samples will be evenly duplicated so that each target has as many samples as the target with the highest number of samples in the dataset.

        Note that this just duplicates first x samples of each class, where `x = max_samples - number of samples of that class`.
        So some samples will be twice as common as others in some cases."""
        samples_per_target = self.get_samples_per_target()
        n_samples = max(list(samples_per_target.values()))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        for target in self.targets:
            if mode == "copy":
                while samples_per_target[target] < n_samples:
                    samples = [i for i in self.samples if i.target == target][
                        : n_samples - samples_per_target[target]
                    ]
                    if copy_samples:
                        samples = [i.copy() for i in samples]
                    self.samples.extend(samples)
                    samples_per_target[target] += len(samples)

                while samples_per_target[target] > n_samples:
                    samples = [i for i in self.samples if i.target == target][
                        : self.get_samples_per_target()[target] - n_samples
                    ]
                    for i in samples:
                        self.samples.remove(i)
                    samples_per_target[target] -= len(samples)

            else:
                raise NotImplementedError

    def subsample(self, n_samples, per_class = False, shuffle = False) -> "DSClassification":
        if not per_class: n_samples = int(n_samples / len(self.targets))
        subsample_ds = DSClassification(self.n_threads, self.target_dtype)
        subsample_ds.update_targets(self.targets)
        for target in self.targets:
            samples = [i for i in self.samples if i.target == target]
            if shuffle: random.shuffle(samples)
            samples = samples[:n_samples]
            if len(samples) < n_samples: print(f"WARNING: there are {len(samples)} samples with target=`{target}`, which is less then {n_samples}.")
            subsample_ds.samples.extend(samples)
        return subsample_ds

    def preview(self, n:int=4):
        v=Visualizer()
        for i in range(n):
            data, label = self[i]
            v.imshow(data, label = str(self.idx_to_target[label]))
        v.show()

    def preview_targets(self, n:int=1):
        v=Visualizer()
        for _ in range(n):
            subset = self.subsample(1, per_class=True, shuffle=True)
            for sample in subset:
                data, label = sample
                v.imshow(data, label = str(subset.idx_to_target[label]))
        v.show()

    def refresh_targets(self):
        self.targets = []
        self.target_to_idx = {}
        self.idx_to_target = {}
        targets = list(set([sample.target for sample in self.samples]))
        self.update_targets(targets)

class SampleToTarget(DSBase.Sample):
    def __init__(
        self,
        data,
        loader: Composable,
        transform_init: Composable,
        transform_sample: Composable,
        transform_target: Composable,
    ) -> None:

        self.data = data
        self.loader = auto_compose(loader)
        self.transform_init = auto_compose(transform_init)
        self.transform_sample = auto_compose(transform_sample)
        self.transform_target = auto_compose(transform_target)

        self.preloaded = None

    def __call__(self) -> tuple:
        """Returns a tuple with loaded and transformed sample:
        ```
        (
            self.transform_sample(self.transform_init(self.loader(self.sample))),
            self.transform_target(self.transform_init(self.loader(self.sample)))
        )
        ```"""
        loaded = self.loader(call_if_callable(self.data)) if self.preloaded is None else self.preloaded
        init_tfmed = self.transform_init(loaded)
        return (
            self.transform_sample(init_tfmed),
            self.transform_target(init_tfmed)
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

    def set_transform_init(self, transform_init: Composable):
        self.transform_init = auto_compose(transform_init)

    def add_transform_init(self, transform_init: Composable):
        self.transform_init = DSBase.smart_compose(self.transform_init, auto_compose(transform_init))

    def set_transform_sample(self, transform_sample: Composable):
        self.transform_sample = auto_compose(transform_sample)

    def add_transform_sample(self, transform_sample: Composable):
        self.transform_sample = DSBase.smart_compose(self.transform_sample, auto_compose(transform_sample))

    def set_transform_target(self, transform_target: Composable):
        self.transform_target = auto_compose(transform_target)

    def add_transform_target(self, transform_target: Composable):
        self.transform_target = DSBase.smart_compose(self.transform_target, auto_compose(transform_target))

class DSToTarget(DSBase.DS):
    def __init__(self, n_threads = 8):
        self.samples: list[SampleToTarget] = []
        self.n_threads = n_threads

        self.last_accessed = []
        self.iter_cursor = 0

    def copy(self, copy_samples=True) -> "DSToTarget":
        ds = DSToTarget(n_threads=self.n_threads)
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()
        return ds

    def add_sample(
        self,
        data,
        loader: Composable = None,
        transform_init: Composable = None,
        transform_sample: Composable = None,
        transform_target: Composable = None,
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

    def add_samples(
        self,
        data,
        loader: Composable = None,
        transform_init: Composable = None,
        transform_sample: Composable = None,
        transform_target: Composable = None,
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

    def add_folder(
        self,
        path: str,
        loader: Composable = None,
        transform_init: Composable = None,
        transform_sample: Composable = None,
        transform_target: Composable = None,
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

    @classmethod
    def from_folder(
        cls,
        path: str,
        loader: Composable = None,
        transform_init: Composable = None,
        transform_sample: Composable = None,
        transform_target: Composable = None,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
        n_threads = 8,
    ) -> "DSToTarget":
        ds = DSToTarget(n_threads = n_threads)
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

    def add_external_dataset(self,
        dataset,
        loader: Composable = None,
        transform_init: Composable = None,
        transform_sample: Composable = None,
        transform_target: Composable = None,
        n_elems=None,
        smart_get0 = True
    ):
        # check if dataset returns (sample, class_index)
        test_sample = dataset[0]
        smart_get0 = smart_get0 and isinstance(test_sample, (tuple, list)) and len(test_sample) == 2

        if n_elems is None:
            n_elems = len(dataset)
        else: n_elems = min(n_elems, len(dataset))

        if smart_get0: samples = [ItemUnder2Indexes(obj=dataset, index1=i, index2=0) for i in range(n_elems)]
        else: samples = [ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)]

        self.add_samples(
            data=samples,
            loader=loader,
            transform_init=transform_init,
            transform_sample=transform_sample,
            transform_target=transform_target,
        )


    @classmethod
    def from_external_dataset(cls,
        dataset,
        loader: Composable = None,
        transform_init: Composable = None,
        transform_sample: Composable = None,
        transform_target: Composable = None,
        n_elems=None,
        smart_get0: bool = True,
        n_threads = 8,
        ) -> "DSToTarget":
        ds = DSToTarget(n_threads=n_threads)
        ds.add_external_dataset(
            dataset=dataset,
            loader=loader,
            transform_init=transform_init,
            transform_sample=transform_sample,
            transform_target=transform_target,
            n_elems=n_elems,
            smart_get0=smart_get0,
        )
        return ds

    def set_transform_init(self, transform_init: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform_init(transform_init)

    def add_transform_init(self, transform_init: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.add_transform_init(transform_init)

    def set_transform_sample(self, transform_sample: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform_sample(transform_sample)

    def add_transform_sample(self, transform_sample: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.add_transform_sample(transform_sample)

    def set_transform_target(self, transform_target: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform_target(transform_target)

    def add_transform_target(self, transform_target: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.add_transform_target(transform_target)


# __________________________ REGRESSION _____________________________

class SampleRegression(DSBase.Sample, DSBase.SampleWithNumericTarget, DSBase.SampleWithTransform):
    def __init__(self, data, loader: Composable, transform: Composable, target: Callable | float | int| torch.Tensor, target_dtype = torch.float32) -> None:
        self.data = data

        # assign label, call on data if callable
        self.set_target(torch.tensor(target, dtype=target_dtype))

        self.loader = auto_compose(loader)
        self.transform = auto_compose(transform)

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
            loader=self.loader,
            transform=self.transform,
            target=self.target,
        )
        sample.preloaded = self.preloaded
        return sample


class DSRegression(DSBase.DS, DSBase.DSWithTargets, DSBase.DSWithTransform):
    def __init__(self, n_threads = 8, target_dtype = torch.float32):
        self.samples: list[SampleRegression] = []
        self.n_threads = n_threads
        self.target_dtype = target_dtype

        self.last_accessed = []
        self.iter_cursor = 0

    def copy(self, copy_samples=True) -> "DSRegression":
        ds = DSRegression()
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()
        return ds

    def add_sample(self, data, loader: Composable = None, transform: Composable = None, target:Callable|float|int|EllipsisType = ...):
        if target is ...: raise ValueError("Target must be specified")
        self.samples.append(SampleRegression(data=data, target=target, loader=loader, transform=transform, target_dtype=self.target_dtype))

    def add_samples(self, data:Iterable, loader: Composable = None,transform: Composable = None, target:Callable|float|int|EllipsisType = ...):
        if target is ...: raise ValueError("Target must be specified")
        self.samples.extend([SampleRegression(data=d, loader=loader, transform=transform, target = target, target_dtype=self.target_dtype) for d in data])

    def add_folder(
        self,
        path: str,
        loader: Composable = None,
        transform: Composable = None,
        target:Callable|float|int|EllipsisType = ...,
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

    @classmethod
    def from_folder(
        cls,
        path: str,
        loader: Composable = None,
        transform: Composable = None,
        target:Callable|float|int|EllipsisType = ...,
        recursive: bool = True,
        extensions: Optional[list[str]] = None,
        path_filt: Optional[Callable] = None,
        n_threads = 8,
        target_dtype = torch.float32,
    ) -> "DSRegression":
        ds = DSRegression(n_threads=n_threads, target_dtype = target_dtype)
        ds.add_folder(
            path=path,
            loader=loader,
            transform=transform,
            target=target,
            recursive=recursive,
            extensions=extensions,
            path_filt=path_filt,
        )
        return ds

    def add_external_dataset(self,
        dataset,
        loader: Composable = None,
        transform: Composable = None,
        target:Callable|float|int|DSBase.AutoTargetFromDataset = DSBase.AutoTargetFromDataset(),
        n_elems=None,
        smart_get0 = True
    ):
        # check if dataset returns (sample, target_index)
        test_sample = dataset[0]
        ds_returns_data_target = isinstance(test_sample, (tuple, list)) and len(test_sample) == 2

        if isinstance(target, DSBase.AutoTargetFromDataset) and not ds_returns_data_target:
            raise ValueError(f"`TargetFromDataset` can only be used when dataset returns `(sample, target_index)`; but it returns {type(test_sample)}")

        if n_elems is None:
            n_elems = len(dataset)
        else: n_elems = min(n_elems, len(dataset))


        if ds_returns_data_target:
            data_target = [
                (
                    ItemUnder2Indexes(obj=dataset, index1=i, index2=0),
                    dataset[i][1] if isinstance(target, DSBase.AutoTargetFromDataset) else target,
                )
                for i in range(n_elems)
            ]
            # if auto-get-zero, it is assumed that the dataset returns (data, target), so we can get data from index 0
            # target it either index 1 if TargetFromDataset is used, or whatever is passed into `target`
            if smart_get0:
                for data, target_ in data_target:
                    self.add_sample(data=data, target=target_, loader=loader, transform=transform)
            # otherwise data is the entire returned value of the dataset
            else:
                for i, (data, target_) in enumerate(data_target):
                    self.add_sample(data=ItemUnderIndex(obj=dataset, index=i), target=target_, loader=loader, transform=transform)

        else:
            if isinstance(target, DSBase.AutoTargetFromDataset):
                raise ValueError(f"`TargetFromDataset` can only be used when dataset returns `(sample, target_index)`; but it returns {type(test_sample)}")

            self.add_samples(
                data=[ItemUnderIndex(obj=dataset, index=i) for i in range(n_elems)],
                loader=loader,
                transform=transform,
                target=target,
            )

    @classmethod
    def from_external_dataset(cls,
        dataset,
        loader: Composable = None,
        transform: Composable = None,
        target:Callable|float|int|DSBase.AutoTargetFromDataset = DSBase.AutoTargetFromDataset(),
        n_elems=None,
        smart_get0: bool = True,
        n_threads = 8,
        target_dtype = torch.float32,
        ) -> "DSRegression":
        ds = DSRegression(n_threads=n_threads, target_dtype=target_dtype)
        ds.add_external_dataset(
            dataset=dataset,
            loader=loader,
            transform=transform,
            target=target,
            n_elems=n_elems,
            smart_get0=smart_get0,
        )
        return ds

    def targets(self) -> torch.Tensor: return torch.as_tensor([s.target for s in self.samples], dtype = self.target_dtype)

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

    def normalize_targets(self, mode = 'z-norm', min = 0, max = 1): # pylint: disable = W0622
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
