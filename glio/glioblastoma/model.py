# Автор - Никишев Иван Олегович группа 224-31

from ..nn import act
from torch import nn
import torch

def grelu(): 
    """Leaky ReLU со смещением"""
    return act.GeneralReLU(0.3, 0.5)


def CNN_Callback(in_channels, out_channels, kernel_size=3, stride = 2, padding = 0, act = nn.ReLU, batch_norm = None, dropout = None):
    """Свёрточный блок с пакетной нормализацией и исключением"""
    
    # Список модулей со 3D свёрточным модулем
    modules = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),]
    
    # Пакетная нормализация
    if batch_norm is not None: modules.append(nn.BatchNorm3d(out_channels))
    
    # Функция активации
    if act is not None: modules.append(act())
    
    # Исключение
    if dropout is not None: modules.append(nn.Dropout3d(dropout))
    
    # Возвращается Sequential c распакованным списком модулей
    return nn.Sequential(*modules)

class CNNBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CNN_Callback(1, 16, 3, 2, 0, grelu, None, 0.0)
        self.conv2 = CNN_Callback(16, 32, 3, 2, 0, grelu, None, 0.0)
        self.conv3 = CNN_Callback(32, 48, 3, 2, 0, grelu, None, 0.0)
        self.conv4 = CNN_Callback(48, 64, 3, 2, 0, grelu, None, 0.0)
        self.conv5 = CNN_Callback(64, 96, 3, 2, 0, grelu, None, 0.0)
        self.conv6 = CNN_Callback(96, 128, (3,4,3), 2, 0, grelu, None, 0.0)
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(1)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x[:,0]
    

def CNN(): return CNNBase()
from ..nn import residual, HistogramLayer

def Hist_Callback(in_channels, out_channels, kernel_size, stride, padding = 0, norm = False, act = True):
    """Блок с гистограммным слоем"""
    
    modules = []
    
    # Список модулей со 3D гистограммным модулем
    modules.append(HistogramLayer(in_channels, num_bins=out_channels, dim = 3, kernel_size = kernel_size, stride=stride,padding= padding))
    
    # Функция активации
    if act: modules.append(nn.ReLU())
    
    # Нормализация
    if norm: modules.append(nn.BatchNorm3d(in_channels*out_channels))
    
    # Возвращается Sequential c распакованным списком модулей
    return nn.Sequential(*modules)

# Задаётся модель
class CNN_(nn.Module):
    """Гист -> CNN2 -> CNN2 -> CNN2 -> ResNet -> CNN2 -> CNN2 -> CNN2 -> ResNet -> CNN2 -> CNN(1,2,1) -> MLP. Активация - ReLU, softmax для классификации. Нужен LSUV"""
    def __init__(self):
        super().__init__()
        self.hist1 = nn.Sequential(HistogramLayer(1, 2, dim = 3, stride = 2),
                                   nn.ReLU())
        
        self.conv1 = CNN_Callback(1, 8, 2, 2) #  Размерность сигнала = 8, 120, 120, 77
        
        self.conv2 = CNN_Callback(4, 12, 2, 2, norm = True) # 12, 60, 60, 38
        
        self.conv3 = CNN_Callback(12, 16, 2, 2) # 16, 30, 30, 19
        
        self.res = residual.ResCallback(16, 16) # Остаточный блок не меняет размерность сигнала (т.к. он должен вычислять функцию идентичности в начале)
        
        self.conv4 = CNN_Callback(16, 24, 2, 2, norm = True) # 24, 15, 15, 9
        
        self.conv5 = CNN_Callback(24, 32, 2, 2) # 32, 7, 7, 4
        
        self.conv6 = CNN_Callback(32, 48, 2, 2, norm = True) # 48, 3, 3, 2
        
        self.res2 = residual.ResCallback(48, 48) # Остаточный блок не меняет размерность сигнала (т.к. он должен вычислять функцию идентичности в начале)
        
        self.conv7 = CNN_Callback(48, 64, 2, 2) # 64, 1, 2, 1
        
        self.conv8 = CNN_Callback(64, 96, (1,2,1), 1, norm = True) # 96, 1, 1, 1
        
        self.flatten = nn.Flatten() # 96, 1, 1, 1 -> 96
        
        self.linear = nn.Linear(96, 2) # 96 -> 2
        
        self.act = nn.Softmax(dim=0)
        
        self.bn = nn.BatchNorm3d(1)
        
    def forward(self, x):
        """Прямой проход"""
        x = self.hist1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.res2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.act(x)
        return x
    
# Параметры обучения
n_epochs = 4
lr = 1e-03

# Блоки для обучения
from .. import callbacks
def get_model(): return CNN()
def get_callbacks(lr, n_epochs): return [callbacks.Optimizer(torch.optim.AdamW, lr=lr),
            callbacks.SchedulerStep(torch.optim.lr_scheduler.OneCycleLR, 1, None, max_lr = lr,  total_steps = 65536*n_epochs),
            callbacks.FastProgressBar(metrics = ['train loss', 'test loss'], plot = True, step_batch=2),
            callbacks.PlotPath(10)
            ]

# Загрузчик для вывода
from torchvision.transforms import v2
from functools import partial

# Значения среднего и дисперсии заранее просчитаны
mean, std = ((234.26439405168807,), (431.60501534598217,))

# Загрузчик конвертирует в формат Channel Last - меняет форму с (x,y,z,1) на (1,x,y,z), и нормализует значениями среднего и дисперсии набора данных
# loader = v2.Compose([partial(torch.squeeze, dim = 3), partial(torch.unsqueeze, dim = 0), v2.Normalize(mean, std)])
# Уже есть в Observation
# def prepare_tensor(tensor: torch.Tensor):
#     """Функция, применяющая загрузчик к тензору"""
#     return loader(tensor)

# Папка TRABIT для нормализации возраста
trabit_path = r'D:\datasets\trabit2019-imaging-biomarkers'
import csv
with open(f'{trabit_path}/train.csv', 'r') as f:
    reader = csv.reader(f)
    ages = []
    for row in reader:
        if row[0] == 'scan_id': continue
        ages.append(float(row[1]))
    
def unnormalize(x):
    """Функция, вычисляющая реальный возраст из предсказания модели, т.к. модель формирует значения от 0 до 1"""
    return x * (max(ages)-min(ages)) + min(ages)

# Загрузка предобученных параметров архитектуры модели и применение к модели
state_path = r'F:\Stuff\Programming\alienai\alienai\glioblastoma_diff\trabit2019 4.98.state'
model_trabit = get_model()
model_trabit.load_state_dict(torch.load(state_path))
model_trabit.eval()


# Test Time Augmentation
import monai.transforms as mtf
transforms = mtf.compose.Compose([mtf.RandZoom(1, 0.95, 1.05), mtf.RandRotate(0.05, 0.05, 0.05, 1)])
def prepare_tensor(tensor: torch.Tensor):
    """Функция для аугментации во время вывода"""
    #print(tensor.shape)
    return transforms(tensor[0])
    
# функция классификации предобученной моделью
def classifier(tensor, ttf_times = 2):
    # Отключение вычисления градиента
    with torch.no_grad():
        
        # Вычисление для первого (немодифицированного) тензора
        predictions = [model_trabit(tensor).detach().cpu()]
        
        # Вычисления для модифицированных тензоров
        for ttf_iter in range(ttf_times):
            
            # Создание тензора
            rand_transformed_tensor = prepare_tensor(tensor).unsqueeze(0)

            # Вычисление предсказания
            predictions.append(float(model_trabit(rand_transformed_tensor).detach().cpu()))
        
        # Среднее предсказаний
        prediction = sum(predictions) / len(predictions)
        prediction
        
    # XAI
    xai_pred = model_trabit(tensor + (0.1**0.5)*torch.randn(*tensor.shape)).detach().cpu()
    
    # Возвращается предсказание, преобразованное в возраст
    return f'{unnormalize(float(predictions[0]))}, уверенность - {abs((float(predictions[0])) - (float(xai_pred)))}%'
    