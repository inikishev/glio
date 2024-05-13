
from ..data.DatasetTreev2 import DatasetBase
from ..python_tools import get0
from ..design.EventModel import EventModel, Event
import torch
from torchvision.transforms import v2

class ExperimentBase(EventModel):
    def __init__(self, events: list[Event]):
        super().__init__(events)
        self.ds = []
        self.init()
        
    def init(self):
        self.event('enter')

class Experiment(ExperimentBase):
    def add_dataset(self, data = [], mean = None, std = None):
        self.data:list = data
        self.mean = mean
        self.std = std
        
        self.event('get_dataset')
        self.event('get_mean_std')
        
        self.loader_list = []
        self.event('get_loader_list')
        self.event('normalize')
        self.loader = v2.Compose(self.loader_list)
        
        self.transform_list = []
        self.event('get_transform_list')
        self.transform = v2.Compose(self.transform_list)
        
        self.label_fn = None
        self.event('get_label_fn')
        
        self.transform_sample_list = []
        self.event('get_transform_sample_list')
        self.transform_sample = v2.Compose(self.transform_sample_list)
        
        self.transform_target_list = []
        self.event('get_transform_target_list')
        self.transform_target = v2.Compose(self.transform_target_list)
        
        self.event('create_dataset')
        self.event('populate_dataset')

    def get(self):
        ds: DatasetBase = type(self.ds)
        for d in self.ds:
            ds.add_dataset(d)
        return ds
        
class GetDatasetBase(Event):
    ORDER = 0
    mean = ...
    std = ...
    loader_list = []
    label_fn = ...

    def get(self) -> list: raise NotImplementedError(f"{type(self)} must have a `get` method.")
    
    def get_dataset(self, e: "Experiment"):
        e.data.extend(self.get())
    
    def get_mean_std(self, e: "Experiment"):
        e.mean = self.mean
        e.std = self.std
        
    def get_loader_list(self, e: "Experiment"): e.loader_list.extend(self.loader_list)
    def get_label_fn(self, e: "Experiment"): return self.label_fn
        
class Get_CIFAR10(GetDatasetBase):
    mean = (0.4805, 0.4795, 0.4601)
    std = (0.2005, 0.1992, 0.2038)
    
    def __init__(self, loader_list = [get0, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]):
        self.loader_list = loader_list
        import torchvision.datasets
        self.ds_train = torchvision.datasets.CIFAR10(r'D:\datasets', download=True, train=True)
        self.ds_test = torchvision.datasets.CIFAR10(r'D:\datasets', download=True, train=False)
    
    
    def get(self): return [self.ds_train, self.ds_test]
    
