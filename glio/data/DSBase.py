"""datasets"""

try: from typing import Callable, Optional, Iterable, Sequence, Any, Self, final
except ImportError: from typing_extensions import Callable, Optional, Iterable, Sequence, Any, Self, final
import concurrent.futures
import os, pickle

from abc import ABC, abstractmethod
import random
import torch, torch.utils.data
from torchvision.transforms import v2
from ..python_tools import (
    ExhaustingIterator,
    Compose,
    identity,
    auto_compose,
    call_if_callable,
    identity_kwargs_if_none,
)
from ..plot import Figure
Composable = Optional[Callable | Sequence[Callable]]

class ExhaustingIteratorDataset(ExhaustingIterator, torch.utils.data.IterableDataset): pass # pylint: disable=W0223

class CacheRepeatIteratorDataset(torch.utils.data.IterableDataset): # pylint: disable=W0223
    def __init__(self, ds: "DS", times: int, elems:int, shuffle=True):
        self.ds = ds.copy(copy_samples=False)
        self.times = times
        self.elems = elems
        self.shuffle = shuffle

    def __iter__(self):
        samples:list[Sample] = []
        indexes:list[int] = []
        cur = 0
        if self.shuffle: self.ds.shuffle()
        for i, sample in enumerate(self.ds.samples):
            samples.append(sample)
            indexes.append(i)
            cur += 1
            if cur%self.elems == 0:
                self.cur = 0
                was_preloaded = [(sample.preloaded is not None) for sample in samples]
                for s, wasp in zip(samples, was_preloaded):
                    if not wasp: s.preload()
                for _ in range(self.times): yield self.ds[random.choice(indexes)]
                for s, wasp in zip(samples, was_preloaded):
                    if not wasp: s.unload()
                samples = []
                indexes = []
        return
    def __len__(self):
        return int((len(self.ds.samples) * self.times) / self.elems)

def smart_compose(old:Callable, new:Callable):
    if old is identity: return new
    if new is identity: return old
    if isinstance(old, torch.nn.Sequential) and isinstance(new, torch.nn.Sequential): return old + new
    if isinstance(old, torch.nn.Module) and isinstance(new, torch.nn.Module): return torch.nn.Sequential(old, new)
    # composes
    if isinstance(old, (v2.Compose, Compose)): tfms = list(old.transforms)
    else: tfms = [old]
    if isinstance(new, (v2.Compose, Compose)): new_tfms = new.transforms
    else: new_tfms = [new]
    tfms.extend(new_tfms)
    return v2.Compose(tfms)

class Sample(ABC):
    @abstractmethod
    def __init__(self, data, loader: Composable) -> None:
        self.data = data
        self.loader: Callable = auto_compose(loader)

    @abstractmethod
    def __call__(self) -> Any | tuple: ...

    @abstractmethod
    def copy(self) -> "Self": ...

    @final
    def preload(self) -> None:
        self.preloaded = self.loader(call_if_callable(self.data))

    @final
    def unload(self) -> None:
        del self.preloaded
        self.preloaded = None

    @final
    def set_loader(self, loader: Composable):
        self.loader = auto_compose(loader)

    @final
    def add_loader(self, loader: Composable):
        self.loader = smart_compose(self.loader, auto_compose(loader))

class DS(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def __init__(self, n_threads = 0):
        self.samples: list | list[Sample] = []
        self.n_threads = n_threads
        self.iter_cursor = 0
        self.last_accessed = []

    @final
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if isinstance(index, int):
            self.last_accessed.append(index)
            return self.samples[index]()

    def __getitems__(self, indexes: Iterable[int]) -> list:
        self.last_accessed.extend(indexes)
        if self.n_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                return list(executor.map(lambda x: self.samples[x](), indexes))
        return [self.samples[i]() for i in indexes]

    @final
    def __iter__(self):
        return self

    @final
    def __next__(self):
        if self.iter_cursor < len(self.samples):
            self.iter_cursor += 1
            return self[self.iter_cursor - 1]
        self.iter_cursor = 0
        raise StopIteration

    @final
    def shuffle(self):
        random.shuffle(self.samples)

    @final
    def set_loader(self, loader: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_loader(loader)

    @final
    def add_loader(self, loader: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.add_loader(loader)

    @final
    def preload(self, amount:Optional[int | float] = None, log = False, nthreads=8):
        if amount is None: amount = len(self.samples)
        elif isinstance(amount, float): amount = int(amount * len(self.samples))
        samples_not_preloaded = [i for i in self.samples if i.preloaded is None]
        len_samples_not_preloaded = len(samples_not_preloaded)
        if log: log_interval = max(1, int(len_samples_not_preloaded/1000))
        else: log_interval = None

        def preload(i,x):
            x.preload()
            if log and i % log_interval == 0: print(f'\r{i}/{len_samples_not_preloaded}', end='')

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
                list(executor.map(preload, list(range(len_samples_not_preloaded)), samples_not_preloaded[:amount]))
        except KeyboardInterrupt: pass
        if log: print()

    def get_preloaded_num(self):
        return sum([1 for i in self.samples if i.preloaded is not None])

    def get_preloaded_percent(self):
        return self.get_preloaded_num() / len(self.samples)

    @final
    def unload(self, amount:Optional[int | float] = None):
        samples_preloaded = [i for i in self.samples if i.preloaded is not None]
        if amount is None: amount = len(samples_preloaded)
        elif isinstance(amount, float): amount = int(amount * len(samples_preloaded))
        for sample in samples_preloaded[:amount]:
            sample.unload()

    @final
    def dump(self, path, splits = 10, lib:Any = pickle):
        split_vals = [1/splits for _ in range(splits)]
        datasets = self.split(split_vals, False)
        if not os.path.exists(path): os.mkdir(path)
        for i, ds in enumerate(datasets):
            with open(f'{path}/ds_{i}.pkl', 'wb') as f:
                lib.dump(ds, f)

    @final
    def joblib_save(self, path, splits = 10):
        import joblib
        split_vals = [1/splits for _ in range(splits)]
        datasets = self.split(split_vals, False)
        if not os.path.exists(path): os.mkdir(path)
        for i, ds in enumerate(datasets):
            joblib.dump(ds, f'{path}/ds_{i}.joblib')

    def pickle_save(self, path, splits=10): self.dump(path, splits, pickle)
    def dill_save(self, path, splits=10):
        import dill
        self.dump(path, splits, dill)

    def load(self, path, pkl_lib:Any = pickle):
        if os.path.isdir(path):
            datasets = []
            for file in os.listdir(path):
                if file.endswith(('.joblib', ".pkl")):
                    ds = self.copy(False)
                    ds.load(os.path.join(path, file))
                    datasets.append(ds)
            for ds in datasets:
                self.add_DS(ds)
        elif os.path.isfile(path):
            if path.endswith(".joblib"):
                import joblib
                ds = joblib.load(path)
                self.add_DS(ds)
            elif path.endswith(".pkl"):
                ds = pkl_lib.load(path)
                self.add_DS(ds)
            else: raise NotImplementedError(f"File type `{path}` is not supported")
        else: raise FileNotFoundError(f"`{path}` doesn't exist")

    def preview(self, n:int=4):
        v=Figure()
        for i in range(n):
            sample = self[i]
            arrays = []
            strings = []
            for d in sample:
                if isinstance(d, torch.Tensor): arrays.append(d.float())
                elif isinstance(d, (str, int, float)): strings.append(str(d))
            for d in arrays: v.add().imshow(d, label = f'{i}: {"; ".join(strings)}')
        v.show(nrow=n)

    @final
    def split(self, *splits, shuffle=True, copy_samples=False) -> list["Self"]:
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

        split_datasets: list["Self"] = []
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

    @final
    def length_iterator(self, length: int, shuffle: bool = True):
        """
        Returns a new dataset that always uses all elements from this dataset evenly while behaving like it has a different number of elements specified in `length`.

        This can be used to make epoch lengths unified for different size datasets.
        """
        return ExhaustingIteratorDataset(self, length=length, shuffle=shuffle)

    @final
    def cache_repeat_iterator(self, times: int, elems:int, shuffle: bool = True):
        """
        Returns a an iterator that preloads each sample and returns `n` times, still applying random transforms each time if those are defined.

        This can be used to speed up dataloading.
        """
        return CacheRepeatIteratorDataset(self, times=times, elems = elems, shuffle=shuffle)

    @final
    def get_mean_std(self, n_samples, batch_size, num_workers = 0, shuffle=True, progress=True) -> tuple:
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
        try:
            for i, item in enumerate(dataloader):
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

                if progress: print(f"{i} / {len(dataloader)}: mean = {mean/(i+1)}, std = {std/(i+1)}", end = '\r')
        except KeyboardInterrupt as e:
            if std[0]==0: raise e
        # calculate mean and std over all samples
        mean /= len(dataloader)
        std /= len(dataloader)

        if progress: print()
        return mean, std

    def copy(self, copy_samples=True) -> "Self":
        ds = type(self)(self.n_threads)
        if copy_samples:
            ds.samples = [sample.copy() for sample in self.samples]
        else:
            ds.samples = self.samples.copy()
        return ds

    def merge(self, ds: "Self"):
        self.samples.extend(ds.samples)

    def add_DS(self, ds: "Self"):
        self.merge(ds)

    @abstractmethod
    def add_sample(self, data) -> None: ...
    @abstractmethod
    def add_samples(self, data: Iterable) -> None: ...
    @abstractmethod
    def add_folder(self, path: str) -> None: ...
    @classmethod
    @abstractmethod
    def from_folder(cls, path:str) -> Self: ...
    @abstractmethod
    def add_external_dataset(self, dataset): ...
    @classmethod
    @abstractmethod
    def from_external_dataset(cls, dataset) -> Self: ...

class SampleWithTransform(ABC):
    @final
    def set_transform(self, transform: Composable):
        self.transform = auto_compose(transform)

    @final
    def add_transform(self, transform: Composable):
        self.transform = smart_compose(self.transform, auto_compose(transform))


class DSWithTransform(ABC):
    samples: list[SampleWithTransform] | list = []
    @final
    def set_transform(self, transform: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_transform(transform)
    @final
    def add_transform(self, transform: Composable, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.add_transform(transform)

class SampleWithTarget(ABC):
    data=...
    def set_target(self, target: Callable | Any) -> None:
        if callable(target):
            target = target(call_if_callable(self.data))
        self.target = target

class SampleWithNumericTarget(ABC):
    data=...
    def set_target(self, target: Callable | Any) -> None:
        if callable(target):
            target = target(call_if_callable(self.data))
        self.target = torch.tensor(target, dtype=torch.float32)


class SampleWithTargetEncoder(ABC):
    data = ...
    def set_target_encoder(self, target_encoder: Optional[Callable]) -> None:
        self.target_encoder = identity_kwargs_if_none(target_encoder)

class DSWithTargets(ABC):
    samples: list[SampleWithTarget] | list[SampleWithNumericTarget] | list = []
    def set_target(self, target, sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_target(target)

class DSWithTargetEncoder(ABC):
    samples: list[SampleWithTargetEncoder] | list = []
    def set_target_encoder(self, target_encoder: Optional[Callable], sample_filter:Optional[Callable] = None):
        for sample in self.samples:
            if sample_filter is None or sample_filter(sample):
                sample.set_target_encoder(target_encoder)


class AutoTargetFromDataset:
    pass

