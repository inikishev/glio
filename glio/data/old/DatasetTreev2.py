
from ..transforms import compose_if_needed
from ..python_tools import apply_tree, flatten, get_first_recursive, reduce, get_all_files
from ..visualize import info, datashow
from ..python_tools import ExhaustingIterator
import torch, torch.utils.data.dataloader
import random
from typing import Callable, Optional, Sequence

class ExternalDatasetItem:
    """A reference to a sample from another dataset, containing the index"""
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index
    def __call__(self): return self.dataset[self.index]
    
class Sample:
    def __init__(self, sample, loader: list[Callable | None] | None, returned: list[bool] | None, label_fn: Callable|str|float|None):
        
        # Assignings
        self._sample = sample
        if loader is not None and not isinstance(loader, (list, tuple)): loader = [loader]
        self.loader = loader
        
        if returned is not None: returned = flatten(returned)
        self.returned = returned
        
        # Label 
        self.set_label(label_fn=label_fn)
       
        self.preloaded = None
        self.last_seen_index: int | None = None
    
    @property
    def sample(self):
        """Needed to support callable sample"""
        if isinstance(self._sample, ExternalDatasetItem): return self._sample()
        return self._sample
        
    def tree(self, flatten: bool):
        """Returns the entire tree, ignoring `returned`"""
        if self.loader is not None:
            return apply_tree(x = self.sample, funcs = self.loader, cached = self.preloaded, flatten = flatten)
        return self.sample
    
    def __call__(self) -> list:
        """Returns a list with branches marked as `returned` + label if it exists."""
        sample = [s for i, s in enumerate(self.tree(flatten = True)) if (self.returned[i] if self.returned is not None else True)]
        if self.label is not None: sample.append(self.label)
        return sample
    
    def preload(self): 
        """Uses the first function in the loader to preload the sample into RAM."""
        loader_fn = get_first_recursive(self.loader)
        if loader_fn is not None: self.preloaded = loader_fn(self.sample)
        
    def preload_clear(self):
        self.preloaded = None
        
    def __str__(self):
        return f'{self.__class__.__name__}; loader = {self.loader}; label = {self.label}; preloaded = {self.preloaded is not None};\nReturned info:\n{info(self())}'
    
    def plot(self, n = 4):
        """Plots the sample `n` times to see the effect of random transforms"""
        # Load it once to avoid loading each time, and delete at the end
        if self.preloaded is None: 
            preloaded = True
            self.preload()
        else: preloaded = False
        
        loader_len = len(flatten(self.loader)) if self.loader else 1
        returned = flatten(self.returned) if self.returned else [True]*loader_len
        
        loader_names = [f'{"RETURNED " if returned[i] else ""}{v.__class__.__name__}' for i,v in enumerate(flatten(self.loader))] if self.loader else None
        
        images = reduce([self.tree(flatten = True) for _ in range(n)])
        if preloaded: self.preload_clear()
        
        datashow(data = images, labels = loader_names, title = self.label, nrows=n)
        
    def copy(self):
        copy = self.__class__(self.sample, self.loader, self.returned, self.label)
        copy.__dict__ = self.__dict__.copy()
        copy.loader = copy.loader.copy() if copy.loader else None
        copy.returned = copy.returned.copy() if copy.returned else None
        return copy
    
    def set_label(self, label_fn: Callable|str|float|None):
        # Label 
        if label_fn is not None: 
            
            # label_fn is label itself
            if not (callable(label_fn) or isinstance(label_fn, (list, tuple))): self.label = label_fn
            
            # label_fn is fn, compute it
            else: self.label: float | str | None = compose_if_needed(label_fn)(self.sample)
        else: self.label = None
        
        
class ExhaustingIterableDataset(ExhaustingIterator, torch.utils.data.IterableDataset): pass

class DatasetBase(torch.utils.data.Dataset):
    def __init__(self):
        self.samples: list[Sample] = []
        self.current_sample_index = 0

    def __len__(self): return len(self.samples)
    def __getitem__(self, index: int): 
        self.samples[index].last_seen_index = self.current_sample_index
        self.current_sample_index += 1
        return self.samples[index]()
    
    def _add(self, sample, loader: list[Callable | None] | None, returned: list[bool] | None, label_fn: Optional[Callable]):
        """Add a sample to the dataset"""
        sample = Sample(sample = sample, loader = loader, returned = returned, label_fn = label_fn)
        self.samples.append(sample)
    
    def _adds(self, samples, loader: list[Callable | None] | None, returned: list[bool] | None, label_fn: Optional[Callable]):
        """Faster way to add a lot of samples to the dataset"""
        samples = [Sample(sample=i, loader = loader, returned=returned, label_fn = label_fn) for i in samples]
        self.samples.extend(samples)

    def add_sample(self, sample, loader: list[Callable | None], returned: list[bool], ): 
        """Adds a sample to the dataset"""
        self._add(sample = sample, loader = loader, returned = returned, label_fn = None)
        
    def add_samples(self, samples: list, loader: list[Callable | None], returned: list[bool], ):
        """Faster way to add a lot of samples to the dataset"""
        self._adds(samples = samples, loader = loader, returned = returned, label_fn = None)
        
    def add_sample_loaded(self, sample, loader: list[Callable | None], returned: list[bool]):
        """Adds a loaded sample to the dataset, first fn in the `loader` will be skipped"""
        self._add(sample = None, loader = loader, returned = returned, label_fn = None)
        self.samples[-1].preloaded = sample
    
    def add_folder(self, path, loader: list[Callable | None], returned: list[bool], recursive = True, extensions: list[str] = None, path_filter = None):
        self._adds(samples = get_all_files(path, recursive = recursive, extensions = extensions, path_filter = path_filter), loader = loader, returned = returned, label_fn = None)
            
    def add_dataset(self, dataset: "DatasetBase", n_elems: int = None):
        self.samples.extend(dataset.samples[:n_elems] if n_elems is not None else dataset.samples)
        
    def add_external_dataset(self, dataset, loader: list[Callable | None], returned: list[bool], n_elems: int = None):        
        if n_elems is None: n_elems = len(dataset)
        samples=[ExternalDatasetItem(dataset, i) for i in range(n_elems)]
        self._adds(samples, loader = loader, returned = returned, label_fn = None)
    
    def set_loader(self, loader: list[Callable | None] | None, returned: list[bool] | None, sample_filter:Callable | None):
        """Sets loader to all samples. `sample_filter` is used to filter samples. Set `loader` to False to remove loader from samples"""
        samples = [i for i in self.samples if sample_filter(i)] if sample_filter is not None else self.samples
        for sample in samples:
            if loader is not None: sample.loader = loader
            if loader is False: sample.loader = None
            
            if returned is not None: sample.returned = returned
            if returned is False: sample.returned = None
        
    def length_iterator(self, length: int, shuffle = True):
        """
        Returns a new dataset that always uses all elements from this dataset evenly while behaving like it has a different number of elements specified in `length`.
        
        This can be used to make epoch lengths unified for different size datasets.
        """
        return ExhaustingIterableDataset(self, length = length, shuffle = shuffle)
    
    def last(self, batch_size: int):
        """Returns last batch of samples"""
        return sorted(self.samples, key = lambda x: x.last_seen_index)[-batch_size:] # pyright: ignore[reportCallIssue, reportArgumentType]
    
    def copy(self, copy_samples = True):
        copy = self.__class__()
        copy.__dict__ = self.__dict__.copy()
        if copy_samples: copy.samples = [i.copy() for i in copy.samples]
        # for k,v in copy.__dict__.items():
        #     if hasattr(v, 'copy') and callable(v.copy):
        #         copy.__dict__[k] = v.copy()
        copy.__dict__ = {k: (v.copy() if (hasattr(v, 'copy') and callable(v.copy)) else v) for k, v in copy.__dict__.items()}
        return copy
          
    def shuffle(self): random.shuffle(self.samples)
    
    def split(self, *args, shuffle=True, copy_samples = False) -> list:
        # Check if only one argument is provided
        if len(args) == 1:
            # Check if the argument is a float, convert the float argument into a tuple with the desired split ratio
            if isinstance(args[0], float): args = (args[0], 1 - args[0])
            # Else convert the integer argument into a tuple with the desired split count
            else: args = (args[0], len(self.samples) - args[0])
        # Shuffle the samples if the shuffle flag is set to True
        if shuffle: self.shuffle()
        # Create a list to store the split datasets
        split_datasets: list["DatasetBase | Dataset_Classification"] = []
        # Initialize a cursor to keep track of the current position in the samples list
        cursor = 0
        # Iterate over the arguments
        for i in range(len(args)):
            # Create a copy of the current object
            split = self.copy(copy_samples = copy_samples)
            # Calculate the length of the split based on the current argument
            length = int(len(split.samples) * args[i] if isinstance(args[i], float) else args[i])
            # Extract the samples for the current split using the cursor and length
            split.samples = split.samples[cursor : cursor + length]
            # Move the cursor to the next position
            cursor += length
            # Add the split dataset to the list
            split_datasets.append(split)
        # Return the list of split datasets
        return split_datasets
    
    def preview(self, index = None):
        if index is None: index = random.randrange(0, len(self.samples))
        self.samples[index].plot()
        
    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.samples)} samples; first sample info:\n{str(self.samples[0])}"
    
    def mean_std(self, batch_size = 1, num_workers = 0, samples =100) -> tuple[tuple[float], tuple[float]]:
        """
        Calculates per channel mean and std for `torchvision.transforms.Normalize`, pass it as Normalize(*result). Returns `Tensor(R.mean, G.mean, B.mean), Tensor(R.std, G.std. B.std)` Increasing `batch_size` MIGHT lead to faster processing, but that will only work if all images have the same size.
        """
        dataset = self if ((not samples) or samples>=len(self)) else self.split(samples)[0]
        test_dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = False)
        sample = next(iter(test_dataloader))
        has_label = True if isinstance(sample, (list, tuple, dict)) else False
        chs = sample[0][0].shape[0] if has_label else sample[0].shape[0]
        
        dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = False)
        mean = torch.zeros(chs)
        std = torch.zeros(chs)
        for item in dataloader:
            sample = item[0] if has_label else item
            ndim = sample.ndim
            channels = list(range(ndim)) # 0 1 2 3
            channels.remove(1) # 0 2 3
            mean += sample.mean(dim = tuple(channels))
            std += sample.std(dim = tuple(channels))
        mean /= len(dataloader)
        std /= len(dataloader)
        return mean, std # pyright:ignore[reportReturnType]
    
def _tree_fn_simple(loader: Callable | None, transform: Callable|None):
    return [loader, transform], [False, True]


class Dataset_Simple(DatasetBase):
    def __init__(self):
        super().__init__()
        
    def add_sample(self, sample, loader: Callable = None, transform: Callable = None): #pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a sample to the dataset"""
        loader_tree, returned = _tree_fn_simple(loader = loader, transform = transform)
        self._add(sample = sample, loader = loader_tree, returned = returned, label_fn = None)
        
    def add_samples(self, samples: list, loader: Callable = None, transform: Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Faster way to add a lot of samples to the dataset"""
        loader_tree, returned = _tree_fn_simple(loader = loader, transform = transform)
        self._adds(samples = samples, loader = loader_tree, returned = returned, label_fn = None)
        
    def add_sample_loaded(self, sample, loader: Callable = None, transform: Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a loaded sample to the dataset, first fn in the `loader` will be skipped"""
        loader_tree, returned = _tree_fn_simple(loader = loader, transform = transform)
        self._add(sample = None, loader = loader_tree, returned = returned, label_fn = None)
        self.samples[-1].preloaded = sample
    
    def add_folder(self, path, loader: Callable = None, transform: Callable = None, recursive = True, extensions:list[str] = None, path_filter:Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.add_samples(samples = get_all_files(path, recursive = recursive, extensions = extensions, path_filter = path_filter), loader = loader, transform = transform)
            
    def add_dataset(self, dataset: DatasetBase, n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.samples.extend(dataset.samples[:n_elems] if n_elems is not None else dataset.samples)

    def add_external_dataset(self, dataset, loader: Callable = None, transform: Callable = None, n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        loader_tree, returned = _tree_fn_simple(loader = loader, transform = transform)   
        if n_elems is None: n_elems = len(dataset)
        samples=[ExternalDatasetItem(dataset, i) for i in range(n_elems)]
        self._adds(samples, loader = loader_tree, returned = returned, label_fn=None)
        
    def set_loader(self, loader: Callable, transform: Callable, sample_filter: Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Sets the loader for all samples. `sample_filter` is used to filter samples. 
        
        `loader` and `transform` will always be set to the new values, even if `None` or `False`."""
        loader_tree, returned = _tree_fn_simple(loader = loader, transform = transform)
        samples = [i for i in self.samples if sample_filter(i)] if sample_filter is not None else self.samples
        
        for sample in samples:
            sample.loader = loader_tree
            sample.returned = returned

            
        
def _tree_fn_classification(loader: Callable | None, transform: Callable | None):
    return [loader, transform], [False, True]
        

class Label_FromPath:
    """Returns label inferred from path. `i = 0` means filename is the label; `i = 1` means last folder in the path is the label, etc"""
    def __init__(self, i = 1):
        self.i = i
    def __call__(self, path: str):
        path = path.replace('\\', '/').replace('//','/')
        return path.split('/')[-self.i-1]
    
class Label_FromDataset:
    def __init__(self, classes: list[str]): self.classes = classes
    def __call__(self,sample): return self.classes[sample[1]]


class Dataset_Classification(DatasetBase):
    """
    This class converts sample into target and label and returns `[target, label]` list
    1. Loads sample by passing `sample` to `loader`
    2. Loads label by passing `sample` to `label_fn`
    3. Encodes label using `label_encoding`. Possible values are `num` - for label index, `one-hot` - for one-hot, `str` for just returning the label
    4. Returns `[transform(sample), label_fn(sample)]` list, otherwise known as `[sample, label]`
    """
    def __init__(self, label_encoding: str = 'num'):
        super().__init__()
        
        self.labels: dict[str|float|int, int] = {}
        self.label_encoding = label_encoding
        
    def __getitem__(self, index): #pyright: ignore [reportIncompatibleMethodOverride]
        self.samples[index].last_seen_index = self.current_sample_index
        self.current_sample_index += 1
        sample, label = self.samples[index]()
        return sample, self.encode_label(label, label_encoding=self.label_encoding)
    
    def _add(self, sample, loader: list[Callable | None] | None, returned: list[bool] | None, label_fn: Callable|str|float): #pyright: ignore [reportIncompatibleMethodOverride]
        """Add a sample to the dataset"""
        sample = Sample(sample = sample, loader = loader, returned = returned, label_fn = label_fn)
        self.samples.append(sample)
        if sample.label is None: raise
        self._add_label(sample.label)
    
    def _adds(self, samples, loader: list[Callable | None] | None, returned: list[bool] | None, label_fn: Callable|str): #pyright: ignore [reportIncompatibleMethodOverride]
        """Faster way to add a lot of samples to the dataset"""
        samples = [Sample(sample=i, loader = loader, returned=returned, label_fn = label_fn) for i in samples]
        self.samples.extend(samples)
        self._add_label([i.label for i in samples]) # pyright: ignore [reportArgumentType]
        
    def add_sample(self, sample, loader: Callable = None, transform: Callable = None, label_fn: Callable|str = None): #pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a sample to the dataset"""
        loader_tree, returned = _tree_fn_classification(loader = loader, transform = transform)
        self._add(sample = sample, loader = loader_tree, returned = returned, label_fn = label_fn) #pyright: ignore [reportArgumentType]
        
    def add_samples(self, samples: list, loader: Callable = None, transform: Callable = None, label_fn: Callable|str = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Faster way to add a lot of samples to the dataset"""
        loader_tree, returned = _tree_fn_classification(loader = loader, transform = transform)
        self._adds(samples = samples, loader = loader_tree, returned = returned, label_fn = label_fn)#pyright: ignore [reportArgumentType]
        
    def add_sample_loaded(self, sample, loader: Callable = None, transform: Callable = None, label_fn: Callable|str = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a loaded sample to the dataset, first fn in the `loader` will be skipped"""
        loader_tree, returned = _tree_fn_classification(loader = loader, transform = transform)
        self._add(sample = None, loader = loader_tree, returned = returned, label_fn = label_fn)#pyright: ignore [reportArgumentType]
        self.samples[-1].preloaded = sample
    
    def add_folder(self, path, loader: Callable = None, transform: Callable = None, label_fn: Callable|str = None, recursive = True, extensions: Sequence[str] = None, path_filter: Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.add_samples(samples = get_all_files(path, recursive = recursive, extensions = extensions, path_filter = path_filter), loader = loader, transform = transform, label_fn = label_fn)
            
    def add_dataset(self, dataset: "Dataset_Classification", n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.samples.extend(dataset.samples[:n_elems] if n_elems is not None else dataset.samples)
        self._add_label(dataset.classes)

    def add_external_dataset(self, dataset, loader: Callable = None, transform: Callable = None, label_fn: Callable = None, label_attr = 'classes', n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        loader_tree, returned = _tree_fn_classification(loader = loader, transform = transform)   
        if label_attr and hasattr(dataset, label_attr): self._add_label(list(getattr(dataset, label_attr)))
        if n_elems is None: n_elems = len(dataset)
        samples=[ExternalDatasetItem(dataset, i) for i in range(n_elems)]
        self._adds(samples, loader = loader_tree, returned = returned, label_fn=label_fn)#pyright: ignore [reportArgumentType]

    
    def set_loader(self, loader: Callable, transform: Callable, label_fn: Callable, sample_filter: Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Sets the loader for all samples. `sample_filter` is used to filter samples. 
        
        `loader` and `transform` will always be set to the new values, even if `None` or `False`.
        
        `label_fn` won't be applied if it is `None`, and will be removed when it is `False`."""
        loader_tree, returned = _tree_fn_classification(loader = loader, transform = transform)
        samples = [i for i in self.samples if sample_filter(i)] if sample_filter is not None else self.samples
        
        for sample in samples:
            
            sample.loader = loader_tree
            sample.returned = returned
            
            if label_fn is not None: sample.set_label(label_fn=label_fn)
            if label_fn is False: sample.label = None
            
        if label_fn is not None: self._update_labels()
            
    def _add_label(self, label: list[str|float|int] | str|float|int):
        """Adds a label to the dataset if it isn't already added"""
        if isinstance(label, (list, tuple)):
            for i in label: 
                if i not in self.labels: self.labels[i] = len(self.labels)
        elif label not in self.labels: self.labels[label] = len(self.labels)
        self.sort_labels()
    
    def one_hot(self, label: str | list, dtype = torch.float):
        """Returns a one-hot encoding of a label"""
        encoding = torch.zeros(size = (len(self.labels),), dtype=dtype)
        if isinstance(label, list):
            for i in label: encoding[self.labels[i]] = 1
        else: encoding[self.labels[label]] = 1
        return encoding
    
    def encode_label(self, label, label_encoding = None):
        """Returns a label encoded as `one-hot` or `num`"""
        label_encoding = label_encoding if label_encoding is not None else self.label_encoding
        if not label_encoding: return label
        label_encoding = label_encoding.lower()
        if label_encoding.startswith('one'): return self.one_hot(label=label)
        elif label_encoding == 'num' or label_encoding == int: 
            if isinstance(label, (list, tuple)): return [self.labels[i] for i in label]
            return self.labels[label]
        elif label_encoding == 'float' or label_encoding == float: 
            if isinstance(label, (list, tuple)): return [float(i) for i in label]
            return float(label)
        elif label_encoding == 'str' or label_encoding == str: 
            if isinstance(label, (list, tuple)): return [str(i) for i in label]
            return str(label)
        raise ValueError(f'Unknown `label_encoding`: {label_encoding}')
    
    def sort_labels(self, key = lambda x: x):
        """Sorts label keys"""
        self.labels = {k:i for i, k in enumerate(sorted(self.labels.keys(), key=key))}
        
    def _update_labels(self):
        """Updates labels from samples"""
        self.labels = {k:i for i, k in enumerate(set([i.label for i in self.samples]))} # pyright: ignore [reportAttributeAccessIssue]
        
    def merge_labels(self, labels:list[str], new_label:str = None):
        """Merges all labels in `labels` into a single one called `new_label` or the first label in `labels`"""
        if new_label is None: new_label = labels[0]
        for i in labels: del self.labels[i]
        self._add_label(new_label)
        for s in self.samples:
            if s.label in labels: s.label = new_label
        self.sort_labels()

    @property
    def classes(self): 
        """Returns a list of classes in the dataset"""
        return list(self.labels.keys())
    
    def label_numbers(self):
        """Returns a dictionary in a form of `{0: 'dog', 1: 'cat', ...}`"""
        return {v:k for k,v in self.labels.items()}
    
    def samples_per_label(self):
        labels = [s.label for s in self.samples]
        return {k:labels.count(k) for k,v in self.labels.items()}
    
    def balance_labels(self, copy_samples = False, mode = 'copy', max_samples = None):
        """Some samples will be duplicated so that each label has as many samples as the label with the highest number of samples in the dataset. Tries to use samples evenly"""
        n_samples = max(list(self.samples_per_label().values()))
        if max_samples is not None: n_samples = min(n_samples, max_samples)
        
        for label in self.classes:
            if mode == 'copy':
                while self.samples_per_label()[label] < n_samples:
                    samples = [i for i in self.samples if i.label == label][:n_samples - self.samples_per_label()[label]]
                    if copy_samples: samples = [i.copy() for i in samples]
                    self.samples.extend(samples)
                
                while self.samples_per_label()[label] > n_samples:
                    samples = [i for i in self.samples if i.label == label][:self.samples_per_label()[label] - n_samples]
                    for i in samples: self.samples.remove(i)
                
            else: raise NotImplementedError
                

def _tree_fn_regression(loader: Callable | None, transform: Callable | None):
    return [loader, transform], [False, True]

def get1(x): return x[1]
class Dataset_Regression(DatasetBase):
    """
    This class converts sample into target and label and returns `[target, label]` list
    1. Loads sample by passing `sample` to `loader`
    2. Loads label by passing `sample` to `label_fn`
    4. Returns `[transform(sample), label_fn(sample)]` list, otherwise known as `[sample, label]`
    
    Labels are not encoded - as this is meant for regression with already numerical labels. You can, however, normalize labels using `normalize_labels` method.
    """
    def __init__(self):
        super().__init__()
        
    def add_sample(self, sample, loader: Callable = None, transform: Callable = None, label_fn: Callable = None): # pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a sample to the dataset"""
        loader_tree, returned = _tree_fn_regression(loader = loader, transform = transform)
        self._add(sample = sample, loader = loader_tree, returned = returned, label_fn = label_fn)
        
    def add_samples(self, samples: list, loader: Callable = None, transform: Callable = None, label_fn: Callable = None): # pyright: ignore [reportIncompatibleMethodOverride]
        """Faster way to add a lot of samples to the dataset"""
        loader_tree, returned = _tree_fn_regression(loader = loader, transform = transform)
        self._adds(samples = samples, loader = loader_tree, returned = returned, label_fn = label_fn)
        
    def add_sample_loaded(self, sample, loader: Callable = None, transform: Callable = None, label_fn: Callable = None): # pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a loaded sample to the dataset, first fn in the `loader` will be skipped"""
        loader_tree, returned = _tree_fn_regression(loader = loader, transform = transform)
        self._add(sample = None, loader = loader_tree, returned = returned, label_fn = label_fn)
        self.samples[-1].preloaded = sample
    
    def add_folder(self, path, loader: Callable = None, transform: Callable = None, label_fn: Callable = None, recursive = True, extensions:Sequence[str] = None, path_filter:Callable = None): # pyright: ignore [reportIncompatibleMethodOverride]
        self.add_samples(samples = get_all_files(path, recursive = recursive, extensions = extensions, path_filter = path_filter), loader = loader, transform = transform, label_fn = label_fn)
            
    def add_dataset(self, dataset:DatasetBase, n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.samples.extend(dataset.samples[:n_elems] if n_elems is not None else dataset.samples)
    
    def add_external_dataset(self, dataset, loader: Callable = None, transform: Callable = None, label_fn: Callable = get1, n_elems = None): # pyright: ignore [reportIncompatibleMethodOverride]
        loader_tree, returned = _tree_fn_regression(loader = loader, transform = transform)   
        if n_elems is None: n_elems = len(dataset)
        samples=[ExternalDatasetItem(dataset, i) for i in range(n_elems)]
        self._adds(samples, loader = loader_tree, returned = returned, label_fn=label_fn)
        
    def set_loader(self, loader: Callable = None, transform: Callable = None, label_fn: Callable = None, sample_filter: Callable = None): # pyright: ignore [reportIncompatibleMethodOverride]
        """Sets the loader for all samples. `sample_filter` is used to filter samples. 
        
        `loader` and `transform` will always be set to the new values, even if `None` or `False`.
        
        `label_fn` won't be applied if it is `None`, and will be removed when it is `False`."""
        loader_tree, returned = _tree_fn_regression(loader = loader, transform = transform)
        samples = [i for i in self.samples if sample_filter(i)] if sample_filter is not None else self.samples
        for sample in samples:
            
            sample.loader = loader_tree
            sample.returned = returned
            
            if label_fn is not None: sample.set_label(label_fn=label_fn)
            if label_fn is False: sample.label = None
    
    def labels(self) -> list[float]:
        """returns a list of all labels"""
        return [s.label for s in self.samples] #pyright:ignore[reportReturnType]
    
    def min(self):
        """Returns min of all labels"""
        return min(self.labels())
    
    def max(self):
        """Returns max of all labels"""
        return max(self.labels()) 
    
    def mean(self):
        """Returns mean of all labels"""
        return sum(self.labels())/len(self.labels())
    
    def std(self):
        """Returns std of all labels"""
        labels = torch.tensor(self.labels())
        return labels.std()
    
    def normalize_labels(self, mode = 'z-norm', min = 0, max = 1):
        """Normalizes labels using z normalization if `mode` is `z`, or by fitting all labels to min-max range"""
        labels = torch.tensor(self.labels())
        if mode.lower().startswith('z'):
            for sample in self.samples:
                if not isinstance(sample.label, (int,float)): raise
                sample.label = (sample.label - float(labels.mean())) / float(labels.std())
        else:
            for sample in self.samples:
                if not isinstance(sample.label, (int,float)): raise
                sample.label = sample.label - float(labels.min())
                sample.label = sample.label / float(labels.max())
                sample.label = sample.label * (max - min) + min
        
        

def _tree_fn_totarget(loader: Callable | None, transform_init: Callable | None, transform_sample: Callable | None, transform_target: Callable | None):
    return [loader, transform_init, [transform_sample], [transform_target]], [False, False, [True], [True]]


class Dataset_ToTarget(DatasetBase):
    """
    Converts sample into target and returns `[sample, target]` list
    1. Loads sample by passing `path` to `loader`
    2. Applies `transform_init` to sample
    3. Returns `[transform_sample(sample), transform_target(sample)]` list, otherwise known as `[sample, target]`
    """
    def __init__(self):
        super().__init__()
        
    def add_sample(self, sample, loader:Callable=None,transform_init:Callable=None,transform_sample:Callable=None,transform_target:Callable=None): #pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a sample to the dataset"""
        loader_tree, returned = _tree_fn_totarget(loader = loader, transform_init = transform_init, transform_sample = transform_sample, transform_target = transform_target)
        self._add(sample = sample, loader = loader_tree, returned = returned, label_fn = None)
        
    def add_samples(self, samples: list, loader:Callable=None,transform_init:Callable=None,transform_sample:Callable=None,transform_target:Callable=None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Faster way to add a lot of samples to the dataset"""
        loader_tree, returned = _tree_fn_totarget(loader = loader, transform_init = transform_init, transform_sample = transform_sample, transform_target = transform_target)
        self._adds(samples = samples, loader = loader_tree, returned = returned, label_fn = None)
        
    def add_sample_loaded(self, sample, loader:Callable=None,transform_init:Callable=None,transform_sample:Callable=None,transform_target:Callable=None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Adds a loaded sample to the dataset, first fn in the `loader` will be skipped"""
        loader_tree, returned = _tree_fn_totarget(loader = loader, transform_init = transform_init, transform_sample = transform_sample, transform_target = transform_target)
        self._add(sample = None, loader = loader_tree, returned = returned, label_fn = None)
        self.samples[-1].preloaded = sample
    
    def add_folder(self, path, loader:Callable=None,transform_init:Callable=None,transform_sample:Callable=None,transform_target:Callable=None, recursive = True, extensions:Sequence[str] = None, path_filter:Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.add_samples(samples = get_all_files(path, recursive = recursive, extensions = extensions, path_filter = path_filter), loader = loader, transform_init = transform_init, transform_sample = transform_sample, transform_target = transform_target)
            
    def add_dataset(self, dataset: DatasetBase, n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        self.samples.extend(dataset.samples[:n_elems] if n_elems is not None else dataset.samples)
    
    def add_external_dataset(self, dataset, loader:Callable=None,transform_init:Callable=None,transform_sample:Callable=None,transform_target:Callable=None, n_elems = None):#pyright: ignore [reportIncompatibleMethodOverride]
        loader_tree, returned = _tree_fn_totarget(loader = loader, transform_init = transform_init, transform_sample = transform_sample, transform_target = transform_target)
        if n_elems is None: n_elems = len(dataset)
        samples=[ExternalDatasetItem(dataset, i) for i in range(n_elems)]
        self._adds(samples, loader = loader_tree, returned = returned, label_fn = None)
        
    def set_loader(self, loader:Callable=None,transform_init:Callable=None,transform_sample:Callable=None,transform_target:Callable=None, sample_filter:Callable = None):#pyright: ignore [reportIncompatibleMethodOverride]
        """Sets the loader for all samples. `sample_filter` is used to filter samples. 
        
        `loader`, `transform_init`, `transform_sample`, `transform_target` will always be set to the new values, even if `None` or `False`."""
        loader_tree, returned = _tree_fn_totarget(loader = loader, transform_init = transform_init, transform_sample = transform_sample, transform_target = transform_target)
        samples = [i for i in self.samples if sample_filter(i)] if sample_filter is not None else self.samples
        for sample in samples:
            sample.loader = loader_tree
            sample.returned = returned
