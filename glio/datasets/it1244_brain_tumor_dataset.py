import torch
from torchvision.transforms import v2
from ..data import DSClassification
from ..loaders.image import imread

MEAN = (0.2466959655, 0.2469369471, 0.2472953051)
STD = (0.2389927208, 0.2390660942, 0.2392502129)

def ensure_channels(x:torch.Tensor, channels = 3):
    if x.shape[0] == channels:
        return x
    return x.repeat(channels, 1, 1)
def get_dataset(path = "D:/datasets/it1244-brain-tumor-dataset/data", loader = v2.Compose([imread, ensure_channels, v2.Normalize(MEAN, STD)])):
    ds = DSClassification()
    with open(f"{path}/train/data.csv", 'r', encoding='utf8') as f:
        for line in f:
            num, label = line.strip().replace("\ufeff", '').split(',')
            ds.add_sample(f"{path}/train/{num}.jpg", target = label, loader=loader)
    return ds