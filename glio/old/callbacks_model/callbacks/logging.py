from ..model import Callback
# from .conditions import condition, Cond_Times


class Metric_Loss(Callback):
    def __init__(self, step = 1, train = True, test = True):
        self.train = train
        self.test = test
        self.step = step

    def enter(self, l):
        self.test_losses = []
        
    def after_batch(self, l):
        if l.is_fitting: 
            if l.train and self.train and l.cur_batch % self.step == 0:
                l.log('train loss', l.loss)
            elif self.test and not l.train:
                self.test_losses.append(l.loss)
    
    def after_epoch(self, l):
        if l.is_fitting: 
            if self.test and not l.train and len(self.test_losses) > 0: 
                l.log('test loss', sum(self.test_losses) / len(self.test_losses))
                self.test_losses = []

from types import FunctionType, MethodType
class Metric_LossFn(Callback):
    def __init__(self, fn, name = None, step = 1, train = True, test = True):
        self.fn = fn
        if name is None: 
            if isinstance(fn, (FunctionType, MethodType)): self.name = fn.__name__
            else: self.name = fn.__class__.__name__
        self.train = train
        self.test = test
        self.step = step

    def enter(self, l):
        self.test_losses = []
        
    def after_batch(self, l):
        if l.status == 'fit': 
            if l.train and self.train and l.cur_batch % self.step == 0:
                l.log(f'train fn {self.name}', l.loss)
            elif self.test and not l.train:
                self.test_losses.append(l.loss)
    
    def after_epoch(self, l):
        if l.status == 'fit': 
            if self.test and not l.train and len(self.test_losses) > 0: 
                l.log(f'test fn {self.name}', sum(self.test_losses) / len(self.test_losses))
                self.test_losses = []

import torch
class Metric_Accuracy(Callback):
    def __init__(self, step = 1, times = None, train = True, test= True):
        self.step = step
        self.times = times
        self.train = train
        self.test = test
        
    def enter(self, l):
        self.test_accs = []
        
    def after_batch(self, l): 
        if l.status == 'fit': 
            if self.step is None:
                if self.times: self.step = int((len(l.dl_train)*l.n_epochs) / self.times)
                else: self.step == 1
            if l.is_training and self.train and l.cur_batch % self.step == 0:
                acc = torch.sum(l.preds.argmax(1) == l.batch[1])/len(l.batch[1])
                l.log('train accuracy', acc)
            elif l.is_testing and not l.train:
                acc = torch.sum(l.preds.argmax(1) == l.batch[1])/len(l.batch[1])
                self.test_accs.append(acc)
            
    def after_epoch(self, l):
        if l.status == 'fit': 
            if self.test and not l.train: 
                l.log('test accuracy', sum(self.test_accs) / len(self.test_accs))
                self.test_accs = []

import time
class Log_Time(Callback):
    order = 1000
    def __init__(self, batch = True, epoch = True):
        self.batch = batch
        self.epoch = epoch
        self.batch_start = time.perf_counter()
        self.epoch_start = time.perf_counter()
        self.measuring = False
        
    def before_batch(self, l): 
        if l.is_training and self.batch: 
            self.measuring = True
            self.batch_start = time.perf_counter()
        
    def after_batch(self, l): 
        if self.measuring: 
            l.log('batch time', time.perf_counter() - self.batch_start)
            self.measuring = False
        
    def before_epoch(self, l): 
        if l.is_training and self.epoch: self.epoch_start = time.perf_counter()
    
    def after_epoch(self, l): 
        if l.is_training and self.epoch: l.log('epoch time', time.perf_counter() - self.epoch_start)
        

class Log_LR(Callback):
    def __init__(self, step = 1): 
        self.step = step
    def after_batch(self, l): 
        if l.is_training and l.cur_batch % self.step == 0: l.log('lr', l.opt.param_groups[0]['lr'])

class Log_Betas(Callback):
    def __init__(self, step = 1, beta0 = True, beta1 = True): 
        self.step = step
        self.beta0 = beta0
        self.beta1 = beta1
        
    def after_batch(self, l): 
        if l.is_training and l.cur_batch % self.step == 0:
            if self.beta0: l.log('beta0', l.opt.param_groups[0]['betas'][0])
            if self.beta1: l.log('beta1', l.opt.param_groups[0]['betas'][1])
            
            
class Log_Preds(Callback):
    def __init__(self, train = True, test = True):
        self.train = train
        self.test = test
    def after_batch(self, l): 
        if l.status == 'fit': 
            if l.train and self.train: l.log('preds train', l.preds)
            elif self.test and not l.train: l.log('preds test', l.preds)