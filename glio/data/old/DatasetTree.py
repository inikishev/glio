# Автор - Никишев Иван Олегович группа 224-31

import numpy as np
import matplotlib.pyplot as plt
from ..python_tools import flatten, reduce
from ..visualize import info, datashow
from typing import Callable

class _Dataset_Item:
    def __init__(self, dataset, index: int):
        self.dataset = dataset
        self.index = index
    def __call__(self): return self.dataset[self.index]

def _path_filter(path: str, extensions: list[str] = None, func_filter: Callable = None):
    if not extensions: return True
    if isinstance(extensions, str): extensions = [extensions]
    ext = [(i if i.startswith('.') else f'.{i}') for i in extensions]
    ext = any([path.lower().endswith(f'{ext.lower()}') for ext in extensions]) if extensions else True
    filt = func_filter(path) if func_filter else True
    return ext and filt

from ..transforms import compose_if_needed
from ..python_tools import apply_tree, get_first_recursive

class SampleTree:
    def __init__(self, sample, loader: list[Callable | None] | None = None, returned: list[bool] = None, label_fn: Callable = None, cache_saver: Callable = None, cache_loader: Callable = None):
        self._sample = sample
        
        self.loader = loader
        self.returned = returned
        
        if label_fn is not None: 
            # label_fn is label itself
            if not (callable(label_fn) or isinstance(label_fn, (list, tuple))): self.label = label_fn
            # label_fn is fn
            else: self.label = compose_if_needed(label_fn)(self.sample)
        else: self.label = None
        
        self.cache_saver = compose_if_needed(cache_saver)
        self.cache_loader = compose_if_needed(cache_loader)
        
        self.preloaded = None
        self.cached = None

    def copy(self):
        copy = self.__class__(self.sample)
        copy.__dict__ = self.__dict__.copy()
        copy.loader = copy.loader.copy() if copy.loader else None
        copy.returned = copy.returned.copy() if copy.returned else None
        return copy
    
    @property
    def sample(self): return self._sample() if isinstance(self._sample, _Dataset_Item) else self._sample
    @sample.setter
    def sample(self, x): self._sample = x
    
    def get_tree(self, flatten = False):
        """Returns the entire tree"""
        if self.loader is not None: return apply_tree(x = self.sample, funcs = self.loader, cached = self.preloaded, flatten = flatten)
        # there is no loader or there is a preloaded sample
        return [self.preloaded] if self.preloaded else [self.sample]
    
    def get(self) -> list:
        """Returns only branches marked as `returned` + label if it exists"""
        returned = flatten(self.returned) if self.returned else None
        #print(returned)
        #print(len(self.get_tree(flatten=True)))
        #print(self.get_tree(flatten=True))
        return [v for k,v in enumerate(self.get_tree(flatten = True)) if (returned[k] if returned else True)]
    
    def __call__(self): 
        """Returns only branches marked as `returned` + label if it exists. If there is only 1 returned object, this will return the object, unlike `get` which always returns a list so it will return a length 1 list"""
        sample = self.get()
        return sample[0] if len(sample) == 1 else sample
    
    def preload(self): 
        # its gonna preload the first function which will be good enough
        preloader = get_first_recursive(self.loader)
        if preloader: self.preloaded = preloader(self.sample)

    def cache(self): ...
    
    def clear(self):
        if self.sample is not None:
            self.preloaded = None
    
    # Cringe stuff
    def __str__(self):
        return f'{self.__class__.__name__}; loader = {self.loader}; preloaded = {self.preloaded is not None};\nReturned info:\n{info(self())}'

    def info_shape(self): return [x.shape for x in self.get()]
    def info_pixels(self): return [np.prod(x.shape) for x in self.get()]
    def info_parameters(self): return np.sum(self.info_pixels())
    def info_min(self): return np.min([x.min() for x in self.get()])
    def info_max(self): return np.max([x.max() for x in self.get()])
    def info_mean(self): 
        """Not 100% accurate because it returns mean of all means, and mean of a 10*10 picture has the same weight as a 1000*1000 picture"""
        return np.mean([x.mean() for x in self()])


    def plot(self, n = 4):
        # Load it once to avoid loading each time, and delete at the end
        self.preload()
        loader_len = len(flatten(self.loader)) if self.loader else 1
        returned = flatten(self.returned) if self.returned else [True]*loader_len
        loader_names = [f'{"RETURNED " if returned[i] else ""}{v.__class__.__name__}' for i,v in enumerate(flatten(self.loader))] if self.loader else None
        
        images = reduce([self.get_tree(True) for _ in range(n)])
        datashow(data = images, labels = loader_names, title = self.label, nrows=n)

import random
import os
import torch.utils.data.dataloader
import torch.utils.data
import torch.multiprocessing
from ..python_tools import ExhaustingIterator

class ExhaustingIterableDataset(ExhaustingIterator, torch.utils.data.IterableDataset): pass
class DatasetTree(torch.utils.data.Dataset):
    def __init__(self, loader: list[Callable] | Callable = None, returned: list[bool] = None, label_fn: Callable = None, cache_saver: Callable = None, cache_loader: Callable = None, label_encoding: str | None = 'num'):
        self.loader = loader
        self.returned = returned
        self.label_fn = label_fn # those 3 are composed if needed in sample
        self.cache_saver = cache_saver
        self.cache_loader = cache_loader

        self.samples: list[SampleTree] = []
        self.labels: dict[str, int] = {}
        
        self.label_encoding = label_encoding
        
        self.has_labels = False
        
        self._cur = 0

    def __len__(self): return len(self.samples)
    
    @property
    def classes(self): return list(self.labels.keys())
    
    def label_numbers(self):
        return {v:k for k,v in self.labels.items()}
    
    def samples_per_label(self):
        labels = [s.label for s in self.samples]
        return {k:labels.count(k) for k,v in self.labels.items()}   

    def __getitem__(self, index): 
        sample: SampleTree = self.samples[index]
        if sample.label is not None: return sample.get() + [self.encode_label(sample.label)]
        return sample()
            
    def get_slice(self, index):
        if self.has_labels: samples = [(i.get() + [self.encode_label(i.label)]) for i in self.samples[index]]
        else: samples = [i() for i in self.samples[index]]
        samples = tuple(zip(*samples))
        if len(samples[0]) > 1: 
            return (torch.stack(samples[0]), (torch.as_tensor(samples[1]) if isinstance(samples[1][0], (int, float)) else torch.stack(samples[1])))
        return torch.stack(samples, dim = 0)
    
    def epoch_iterator(self, epoch_size = 65536, shuffle = True):
        """
        Basically lets you use epoch size separate from the amount of samples in the dataset while ensuring all elements will be used evenly.
        
        It will exhaust every sample in the dataset, reshuffle and start again, in parallel raising StopIteration every time it returns epoch_size elements.
        """
        return ExhaustingIterableDataset(self, length = epoch_size, shuffle = shuffle)
    
    def one_hot(self, label, dtype = torch.float):
        encoding = torch.zeros(size = (len(self.labels),), dtype=dtype)
        if isinstance(label, list):
            for i in label: encoding[self.labels[i]] = 1
        else: encoding[self.labels[label]] = 1
        return encoding
    
    def add_label(self, label):
        # label is list, add each element
        if isinstance(label, list):
            for i in label: 
                if i not in self.labels: self.labels[i] = len(self.labels)
        # label is string (probably)
        elif label not in self.labels: self.labels[label] = len(self.labels)
        
    def encode_label(self, label, label_encoding = None):
        label_encoding = label_encoding if label_encoding is not None else self.label_encoding
        if not label_encoding: return label
        label_encoding = label_encoding.lower()
        if label_encoding == 'onehot': return self.one_hot(label=label)
        elif label_encoding == 'num': 
            if isinstance(label, (list, tuple)): return [self.labels[i] for i in label]
            return self.labels[label]
        elif label_encoding == 'float' or label_encoding == float: 
            if isinstance(label, (list, tuple)): return [float(i) for i in label]
            return float(label)
        raise ValueError(f'Unknown `label_encoding`: {label_encoding}')
    
    def sort_labels(self):
        self.labels = {k:i for i, k in enumerate(sorted(self.labels.keys()))}
    
    def merge_labels(self, labels:list[str], new_label:str = None):
        if new_label is None: new_label = labels[0]
        for i in labels: del self.labels[i]
        self.add_label(new_label)
        for s in self.samples:
            if s.label in labels: s.label = new_label
        self.sort_labels()
        
    def _kwargs(self, kwargs: dict, start=2, end=0):
        # Get the length of the kwargs dictionary
        lenkwargs = len(kwargs)
        
        # Iterate over the keys and values of the kwargs dictionary
        for i, k in enumerate(list(kwargs.keys())):
            # If the index is less than the start index or greater than or equal to the length of kwargs minus the end index,
            # delete the key-value pair from the kwargs dictionary
            if i < start or i >= lenkwargs - end:
                del kwargs[k]
            else:
                # If the value is a list or tuple, turn it into a callable.
                if isinstance(kwargs[k], (list, tuple)):
                    kwargs[k] = compose_if_needed(kwargs[k])
                # If the value is None, assign the value of the attribute with the same name from the self object to it
                elif kwargs[k] is None:
                    kwargs[k] = getattr(self, k)

            # for child classes processing goes there
            # it needs loader, returned, label_fn, cache_saver, cache_loader
            # Return the modified kwargs dictionary
            return kwargs
    
    def _add(self, sample, loader: list[Callable | None], returned, label_fn, cache_saver, cache_loader):
        sample = SampleTree(sample=sample, loader = loader, returned=returned, label_fn = label_fn, cache_saver = cache_saver, cache_loader = cache_loader)
        self.samples.append(sample)
        if sample.label is not None: 
            self.has_labels = True
            self.add_label(sample.label)
            
    def _adds(self, samples, loader, returned, label_fn, cache_saver, cache_loader):
        samples = [SampleTree(sample=i, loader = loader, returned=returned, label_fn = label_fn, cache_saver = cache_saver, cache_loader = cache_loader) for i in samples]
        self.samples.extend(samples)
        labels = [i.label for i in samples if i.label]
        if len(labels) > 0:
            self.has_labels = True
            [self.add_label(i) for i in labels]
            

    def _add_loaded_sample(self, sample, loader, returned, label_fn, cache_saver, cache_loader, label): 
        sample = SampleTree(sample = None, loader=loader, returned=returned, label_fn=label_fn) # sample = None; loader, returned = **kwars
        sample.preloaded = sample
        self.samples.append(sample)
        if sample.label is None: 
            sample.label = label
        if sample.label is not None: # either label_fn or label so need the twice thing
            self.has_labels = True
            self.add_label(label)
    
    def _add_dataset(self, dataset, loader, returned, label_fn, cache_saver, cache_loader, label_attr, n_elems):
        if label_attr and hasattr(dataset, label_attr): self.add_label(list(getattr(dataset, label_attr)))
        if n_elems is None: n_elems = len(dataset)
        samples=[_Dataset_Item(dataset, i) for i in range(n_elems)]
        self._adds(samples, loader = loader, returned = returned, label_fn=label_fn, cache_saver=cache_saver, cache_loader=cache_loader)
            
    def _add_file(self, path, extensions, path_filter, loader, returned, label_fn, cache_saver, cache_loader, folder_as_file):
        if _path_filter(path, extensions=extensions, func_filter=path_filter):
            if folder_as_file:
                if not os.path.isdir(path): raise FileNotFoundError(f"{path} doesn't exist or isn't a directory")
            else:
                if not os.path.isfile(path): raise FileNotFoundError(f"{path} doesn't exist or isn't a file")
            self._add(sample = path,
                            loader = loader, returned=returned, label_fn = label_fn, cache_saver = cache_saver, cache_loader = cache_loader)

    def _add_folder(self, path, recursive, extensions, path_filter, loader, returned, label_fn, cache_saver, cache_loader):
        if not os.path.isdir(path): raise FileNotFoundError(f"{path} doesn't exist or isn't a folder")
        for i in os.listdir(path):
            full_path = os.path.join(path, i)
            if os.path.isfile(full_path): self._add_file(path=full_path, extensions=extensions, path_filter=path_filter, 
                                                        loader = loader, returned = returned, label_fn = label_fn, cache_saver = cache_saver, cache_loader = cache_loader, folder_as_file=False)
            # else its a folder
            elif recursive: self._add_folder(path=full_path, recursive=recursive, extensions=extensions, path_filter=path_filter, 
                                            loader = loader, returned = returned, label_fn = label_fn, cache_saver = cache_saver, cache_loader = cache_loader)


        
    def _set_loader(self, loader, returned, label_fn, cache_saver, cache_loader, sample_filter, set_default):
        for sample in self.samples:
            if sample_filter and not sample_filter(sample): continue
            if loader is not None: 
                sample.loader = loader
                if set_default: self.loader = loader
            if returned is not None: 
                sample.returned = returned
                if set_default: self.returned = returned
            
            if label_fn is not None: 
                sample.label = compose_if_needed(label_fn)(sample.sample)
                if set_default: self.label_fn = label_fn
            
            if cache_saver is not None: 
                sample.cache_saver = compose_if_needed(cache_saver)
                if set_default: self.cache_saver = cache_saver
            if cache_loader is not None: 
                sample.cache_loader = compose_if_needed(cache_loader)
                if set_default: self.cache_loader = cache_loader


    def add_sample(self, sample, loader: list = None, 
                   returned: list = None, label_fn = None, cache_saver = None, cache_loader = None):
        self._add(sample=sample, **self._kwargs(locals())) # pyright: ignore [reportCallIssue]
        
    def add_loaded_sample(self, sample, 
                          loader = None, returned = None, label_fn = None,
                          label = None): 
        self._add_loaded_sample(sample=sample, **self._kwargs(locals(), 2, 1), label=label) # pyright: ignore [reportCallIssue]
        
    def add_dataset(self, dataset, 
                    loader: list = None, returned: list = None, label_fn = None, cache_saver = None, cache_loader = None, 
                    label_attr = None, n_elems = None):
        self._add_dataset(dataset=dataset, **self._kwargs(locals(), 2, 2), label_attr=label_attr, n_elems = n_elems) # pyright: ignore [reportCallIssue]
        
    def add_file(self, path, extensions: list = None, path_filter = None, 
                 loader: list = None, returned: list = None, label_fn = None, cache_saver = None, cache_loader = None,
                 folder_as_file = False):
        self._add_file(path=path,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 4, 1), folder_as_file = folder_as_file) # pyright: ignore [reportCallIssue]
        
    def add_folder(self, path, recursive = True, extensions = None, path_filter = None, 
                   loader: list = None, returned: list = None, label_fn = None, cache_saver = None, cache_loader = None):
        self._add_folder(path=path,recursive=recursive,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 5, 0)) # pyright: ignore [reportCallIssue]
        
    def set_loader(self, 
                   loader: list = None, returned: list = None, label_fn = None, cache_saver = None, cache_loader = None, 
                   sample_filter = None, set_default = True):
        """Arguments that are None (or not set to it defaults to None), won't do anything. If you want to set something to None, set the argument to False"""
        self._set_loader(**self._kwargs(locals(), 1, 2), sample_filter=sample_filter, set_default=set_default) # pyright: ignore [reportCallIssue]
        
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
    
    def split(self, *args, shuffle=True, copy_samples = False) -> list["DatasetTree"]:
        # Check if only one argument is provided
        if len(args) == 1:
            # Check if the argument is a float, convert the float argument into a tuple with the desired split ratio
            if isinstance(args[0], float): args = (args[0], 1 - args[0])
            # Else convert the integer argument into a tuple with the desired split count
            else: args = (args[0], len(self.samples) - args[0])
        # Shuffle the samples if the shuffle flag is set to True
        if shuffle: self.shuffle()
        # Create a list to store the split datasets
        split_datasets = []
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
    
    def random(self, n = 1):
        return [self.samples[random.randrange(0, len(self.samples))] for _ in range(n)]
    
    def plot_transformed(self, index = None):
        if index is None: index = random.randrange(0, len(self.samples))
        self.samples[index].plot()

    def plot_loaded_grid(self, rows = 4, columns = 4, figsize = (5,10)):
        indexes = list(range(0,rows*columns))
        random.shuffle(indexes)
        fig, axes = plt.subplots(rows, columns, figsize = figsize)
        axes = axes.flatten()
        for i, ax in zip(indexes, axes.flatten()):
            ax.imshow(self.samples[i].plot())
            ax.set_title(f'{i}: {str(self.samples[i].sample)[:100]}', fontsize=8)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        plt.show()

    def paths(self):
        return [(s.sample if isinstance(s.sample, str) else None) for s in self.samples ]
    
    def count_preloaded(self):
        return sum([(s.preloaded is not None) for s in self.samples])

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.samples)} samples; first sample info:\n{str(self.samples[0])}"
    
    def mean_std(self, batch_size = 1, num_workers = 0, samples =100):
        """
        Calculates per channel mean and std for `torchvision.transforms.Normalize`, pass it as Normalize(*result). Returns `Tensor(R.mean, G.mean, B.mean), Tensor(R.std, G.std. B.std)` Increasing `batch_size` MIGHT lead to faster processing, but that will only work if all images have the same size.
        """
        dataset = self if ((not samples) or samples>=len(self)) else self.split(samples)[0]
        dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = False)
        chs = self[0][0].shape[0]
        mean = torch.zeros(chs)
        std = torch.zeros(chs)
        for sample, target in dataloader:
            ndim = sample.ndim
            channels = list(range(ndim)) # 0 1 2 3
            channels.remove(1) # 0 2 3
            mean += sample.mean(dim = tuple(channels))
            std += sample.std(dim = tuple(channels))
        mean /= len(dataloader)
        std /= len(dataloader)
        return mean, std



class Dataset_ToTarget(DatasetTree):
    def __init__(self, loader: Callable | list[Callable] = None, transform_init: Callable | list[Callable] = None, transform_sample: Callable | list[Callable] = None, transform_target: Callable | list[Callable] = None, cache_saver = None, cache_loader = None):
        """
        This class converts sample into target and returns `[sample, target]` list
        1. Loads sample by passing `path` to `loader`
        2. Applies `transform_init` to sample
        3. Returns `[transform_sample(sample), transform_target(sample)]` list, otherwise known as `[sample, target]`
        """

        self.transform_init = compose_if_needed(transform_init)
        self.transform_sample = compose_if_needed(transform_sample)
        self.transform_target = compose_if_needed(transform_target)
        #tree = [self.loader, self.transform_init, [self.transform_sample],[self.transform_target]]
        super().__init__(loader = compose_if_needed(loader), returned=None, cache_saver = cache_saver, cache_loader = cache_loader)

    def _kwargs(self, kwargs:dict, start = 2, end = 0):
        lenkwargs =  len(kwargs)
        for i, k in enumerate(list(kwargs.keys())): # 0 is self, 1 is sample
            if i < start or i >= lenkwargs - end: del kwargs[k] 
            else:
                if isinstance(kwargs[k], (list, tuple)): kwargs[k] = compose_if_needed(kwargs[k])
                elif kwargs[k] is None: kwargs[k] = getattr(self, k)
        # for child classes processing goes there
        # it needs loader, returned, label_fn, cache_saver, cache_loader
        if 'loader' not in kwargs: kwargs['loader'] = None
        arguments = {}
        arguments['loader'] = [ 
                               kwargs['loader'] if 'loader' in kwargs else None, 
                               kwargs['transform_init'], 
                               [ kwargs['transform_sample'] ] , 
                               [ kwargs['transform_target'] ] 
                               ]
        arguments['returned'] = [False, False, [True], [True]]
        arguments['label_fn'] = None
        arguments['cache_saver'] = kwargs['cache_saver'] if 'cache_saver' in kwargs else None
        arguments['cache_loader'] = kwargs['cache_loader'] if 'cache_loader' in kwargs else None
        return arguments
    
    def add_sample(self, sample, # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform_init = None, transform_sample = None, transform_target = None, cache_saver = None, cache_loader = None):
        self._add(sample=sample, **self._kwargs(locals()))
        
    def add_loaded_sample(self, sample, # pyright: ignore [reportIncompatibleMethodOverride]
                          transform_init = None, transform_sample = None, transform_target = None, 
                          label = None): 
        self._add_loaded_sample(sample=sample, **self._kwargs(locals(), 2, 1), label=label)
        
    def add_dataset(self, dataset, # pyright: ignore [reportIncompatibleMethodOverride]
                    loader = None, transform_init = None, transform_sample = None, transform_target = None, cache_saver = None, cache_loader = None, 
                    n_elems = None):
        self._add_dataset(dataset=dataset, **self._kwargs(locals(), 2, 1), label_attr=None, n_elems=n_elems)
        
    def add_file(self, path, extensions: list = None, path_filter = None, # pyright: ignore [reportIncompatibleMethodOverride]
                 loader = None, transform_init = None, transform_sample = None, transform_target = None, cache_saver = None, cache_loader = None,
                 folder_as_file = False):
        self._add_file(path=path,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 4, 1), folder_as_file = folder_as_file)
        
    def add_folder(self, path, recursive = True, extensions = None, path_filter = None, # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform_init = None, transform_sample = None, transform_target = None, cache_saver = None, cache_loader = None):
        self._add_folder(path=path,recursive=recursive,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 5, 0))
        
    def set_loader(self, # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform_init = None, transform_sample = None, transform_target = None, cache_saver = None, cache_loader = None, 
                   sample_filter = None, set_default=True):
        self._set_loader(**self._kwargs(locals(), 1, 2), sample_filter=sample_filter, set_default=set_default)
        
        
    
def label_from_folder(path):
    return path.replace('\\', '/').replace('//','/').split('/')[-2]

class LabelFromDataset:
    def __init__(self, classes: list[str]): self.classes = classes
    def __call__(self,sample): return self.classes[sample[1]]

def label_index_from_dataset(sample): return sample[1]

class Dataset_Label(DatasetTree):
    def __init__(self, loader:Callable | list[Callable] = None, transform:Callable | list[Callable] = None, label_fn:Callable = None, cache_saver = None, cache_loader = None, label_encoding: str | None = 'num'):
        """
        This class converts sample into target path into label and returns `[target, label]` list
        1. Loads sample by passing `path` to `loader`
        2. Loads label by passing `path` to `label_fn`
        3. Returns `[transform(sample), target_transform(label)]` list, otherwise known as `[sample, label]`
        """

        self.transform = compose_if_needed(transform)
        super().__init__(loader = compose_if_needed(loader), returned=None, label_fn = label_fn, cache_saver = cache_saver, cache_loader = cache_loader, label_encoding=label_encoding)

    def _kwargs(self, kwargs:dict, start = 2, end = 0):
        lenkwargs =  len(kwargs)
        for i, k in enumerate(list(kwargs.keys())): # 0 is self, 1 is sample
            if i < start or i >= lenkwargs - end: del kwargs[k] 
            else:
                if isinstance(kwargs[k], (list, tuple)): kwargs[k] = compose_if_needed(kwargs[k])
                elif kwargs[k] is None: kwargs[k] = getattr(self, k)
        # for child classes processing goes there
        # it needs loader, returned, label_fn, cache_saver, cache_loader
        if 'loader' not in kwargs: kwargs['loader'] = None
        arguments = {}
        arguments['loader'] = [ 
                               kwargs['loader'],
                               kwargs['transform'], 
                               ]
        arguments['returned'] = [False, True]
        arguments['label_fn'] = kwargs['label_fn']
        arguments['cache_saver'] = kwargs['cache_saver'] if 'cache_saver' in kwargs else None
        arguments['cache_loader'] = kwargs['cache_loader'] if 'cache_loader' in kwargs else None
        return arguments
    
    def add_sample(self, sample,  # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform = None, label_fn = None, cache_saver = None, cache_loader = None, 
                   ):
        self._add(sample=sample, **self._kwargs(locals()))
        
    def add_loaded_sample(self, sample,  # pyright: ignore [reportIncompatibleMethodOverride]
                          transform = None, label_fn = None,
                          label = None): 
        self._add_loaded_sample(sample=sample, **self._kwargs(locals(), 2, 1), label=label)
        
    def add_dataset(self, dataset,  # pyright: ignore [reportIncompatibleMethodOverride]
                    loader = None, transform = None, label_fn = None, cache_saver = None, cache_loader = None,
                    label_attr = None, n_elems = None):
        self._add_dataset(dataset=dataset, **self._kwargs(locals(), 2, 2), label_attr=label_attr, n_elems = n_elems)
        
    def add_file(self, path, extensions: list = None, path_filter = None,  # pyright: ignore [reportIncompatibleMethodOverride]
                 loader = None, transform = None, label_fn = None, cache_saver = None, cache_loader = None,
                 folder_as_file = False):
        self._add_file(path=path,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 4, 1), folder_as_file = folder_as_file)
        
    def add_folder(self, path, recursive = True, extensions = None, path_filter = None,  # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform = None, label_fn = None, cache_saver = None, cache_loader = None,
                   ):
        self._add_folder(path=path,recursive=recursive,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 5, 0))
        
    def set_loader(self,  # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform = None, label_fn = None, cache_saver = None, cache_loader = None,
                   sample_filter = None, set_default=True):
        self._set_loader(**self._kwargs(locals(), 1, 2), sample_filter=sample_filter, set_default=set_default)
        


class Dataset_Simple(DatasetTree):
    def __init__(self, loader:Callable | list[Callable] = None, transform:Callable | list[Callable] = None, cache_saver = None, cache_loader = None):
        """
        This class does this!!!
        """

        self.transform = compose_if_needed(transform)
        super().__init__(loader = compose_if_needed(loader), returned=None, label_fn=None, cache_saver = cache_saver, cache_loader = cache_loader, label_encoding=None)

    def _kwargs(self, kwargs:dict, start = 2, end = 0):
        lenkwargs =  len(kwargs)
        for i, k in enumerate(list(kwargs.keys())): # 0 is self, 1 is sample
            if i < start or i >= lenkwargs - end: del kwargs[k] 
            else:
                if isinstance(kwargs[k], (list, tuple)): kwargs[k] = compose_if_needed(kwargs[k])
                elif kwargs[k] is None: kwargs[k] = getattr(self, k)
        # for child classes processing goes there
        # it needs loader, returned, label_fn, cache_saver, cache_loader
        if 'loader' not in kwargs: kwargs['loader'] = None
        arguments = {}
        arguments['loader'] = [ 
                               kwargs['loader'],
                               kwargs['transform'], 
                               ]
        arguments['returned'] = [False, True]
        arguments['label_fn'] = None
        arguments['cache_saver'] = kwargs['cache_saver'] if 'cache_saver' in kwargs else None
        arguments['cache_loader'] = kwargs['cache_loader'] if 'cache_loader' in kwargs else None
        return arguments
    
    def add_sample(self, sample,  # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform = None, cache_saver = None, cache_loader = None, 
                   ):
        self._add(sample=sample, **self._kwargs(locals()))
        
    def add_loaded_sample(self, sample, # pyright: ignore [reportIncompatibleMethodOverride]
                          transform = None,
                          label = None): 
        self._add_loaded_sample(sample=sample, **self._kwargs(locals(), 2, 1), label=label)
        
    def add_dataset(self, dataset, # pyright: ignore [reportIncompatibleMethodOverride]
                    loader = None, transform = None, cache_saver = None, cache_loader = None,
                    n_elems = None):
        self._add_dataset(dataset=dataset, **self._kwargs(locals(), 2, 1), label_attr=None, n_elems = n_elems)
        
    def add_file(self, path, extensions: list = None, path_filter = None, # pyright: ignore [reportIncompatibleMethodOverride]
                 loader = None, transform = None, cache_saver = None, cache_loader = None,
                 ):
        self._add_file(path=path,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 4, 0))
        
    def add_folder(self, path, recursive = True, extensions = None, path_filter = None, # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform = None, cache_saver = None, cache_loader = None,
                   ):
        self._add_folder(path=path,recursive=recursive,extensions=extensions,path_filter=path_filter, **self._kwargs(locals(), 5, 0))
        
    def set_loader(self, # pyright: ignore [reportIncompatibleMethodOverride]
                   loader = None, transform = None, cache_saver = None, cache_loader = None,
                   sample_filter = None, set_default = True):
        self._set_loader(**self._kwargs(locals(), 1, 2), sample_filter=sample_filter, set_default=set_default)
        