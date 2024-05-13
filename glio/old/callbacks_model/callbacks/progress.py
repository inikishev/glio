from ..model import Callback
# from .conditions import condition
class Print(Callback): # TODO MAKE IT A MARKDOWN TABLE THAT WOULD BE KINDA NICE
    "Simple progress tracker, prints metrics every n batches/epochs"
    order = 110
    def __init__(self, metrics = ['train loss', 'test loss'], step_batch = None, step_epoch = 1, on_fit = True):
        
        if isinstance(metrics, str): metrics = [metrics]
        self.metrics = metrics
        self.step_batch = step_batch
        self.step_epoch = step_epoch
        self.on_fit = on_fit
        
    def _print(self, l):
        print(f'epoch {l.cur_epoch}; batch {l.cur_batch}', end='')
        for metric in self.metrics: 
            if metric in l.logs:
                print(f'; {metric} = {round(float(l.logs.last(metric)), 5)}', end='')
        print()
        
    def after_batch(self, l):
        if l.is_training and self.step_batch and l.cur_batch % self.step_batch == 0: self._print(l)

    def after_epoch_full(self, l):
        if self.step_epoch and l.cur_epoch % self.step_epoch == 0: self._print(l)
        
    def after_fit(self, l):
        if l.is_fitting and self.on_fit and ((not self.step_epoch) or (self.step_epoch and l.cur_epoch % self.step_epoch != 0)): self._print(l)


import numpy as np
class SimpleProgressBar(Callback):
    def __init__(self, length = 10): self.length = length
    def before_epoch(self, l):
        self.update_on = max(1, int(len(l.dl) // self.length))
        self.cur = 0
        if l.train: print(f'epoch #{l.cur_epoch} - training: ', end = '')
        if not l.train: print(f'epoch #{l.cur_epoch} - testing: ', end = '')
    def after_batch(self, l):
        if self.cur % self.update_on == 0: print('⬜', end='')
        self.cur+=1
    def after_epoch(self, l):
        print('\r                                                                                   ', end = '\r')
    def after_fit(self, l): print()
    
class FastProgressBar(Callback):
    def __init__(self, metrics = ['train loss', 'test loss'], plot = False, step_batch = 16, step_epoch = None, fit = True, plot_max = 4096, smooth = None):
        from fastprogress.fastprogress import master_bar, progress_bar
        self._master_bar = master_bar
        self._progress_bar = progress_bar
        self.order = 90
        self.plot = plot
        if isinstance(metrics, str): metrics = [metrics]
        self.metrics = metrics
        self.plot_max = plot_max
        self.b = step_batch
        self.e = step_epoch
        self.fit = fit
        if isinstance(smooth, int): smooth = [smooth for _ in range(len(metrics))]
        self.smooth = smooth

    def before_fit(self, l): 
        if l.is_fitting:
            self.mbar = l.epochs = self._master_bar(l.epochs)
    
    def before_epoch(self, l): 
        if l.is_fitting:
            l.dl = self._progress_bar(l.dl, leave=False, parent=self.mbar)
    
    def _plot(self, l):
        if self.plot:
            metrics = [l.logs[metric] for metric in self.metrics if metric in l.logs]
            metrics = [i for i in metrics if len(i) > 0]
            if len(metrics) > 0:
                reduction = [max(int(len(metric) / self.plot_max), 1) for metric in metrics]
                metrics = [([list(m.keys())[::reduction[i]], list(m.values())[::reduction[i]]] if reduction[i]>1 else [list(m.keys()), list(m.values())]) for i, m in enumerate(metrics)]
                if self.smooth:
                    for i in range(len(metrics)):
                        if self.smooth[i] is not None and self.smooth[i] > 1 and len(metrics[i][1]) > self.smooth[i]:
                            metrics[i][1] = np.convolve(metrics[i][1], np.ones(self.smooth[i])/self.smooth[i], 'same')
                self.mbar.update_graph(metrics)

    def after_batch(self, l):
        if self.b and l.is_fitting and l.cur_batch % self.b == 0: self._plot(l)
    def after_epoch(self, l):
        if self.e and l.is_fitting and l.cur_epoch % self.e == 0: self._plot(l)
    def after_fit(self, l):
        if self.fit and l.is_fitting: self._plot(l)
        
        

from ...visualize import datashow
class Show(Callback):
    def __init__(self, times = 10, step_batch = None, step_epoch = None, metrics = [['train loss', 'test loss']], inputs = True,targets = True, outputs = True):
        self.b = step_batch
        self.e = step_epoch
        self.times = times
        self.metrics = metrics
        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        
    def _plot(self, l):
        stuff = []
        labels = []
        if self.metrics:
            for metric in self.metrics:
                if isinstance(metric, (tuple, list)):
                    stuff.append([l.logs[i] for i in metric if i in l.logs])
                    labels.append(', '.join(metric))
                else:
                    if metric in l.logs:
                        stuff.append(l.logs[metric])
                        labels.append(metric)
        if self.inputs: 
            if l.input_is_target: stuff.append(l.batch[0])
            else: stuff.append(l.batch[0][0])
            labels.append('inputs')
        if self.targets and not l.input_is_target: stuff.append(l.batch[1][0]); labels.append('targets')
        if self.outputs:
            if l.preds[0].ndim == 1:stuff.append(l.preds[0].argmax()); labels.append('outputs')
            else: stuff.append(l.preds[0]); labels.append('outputs')
        datashow(stuff, labels=labels, ncols=len(stuff))
        
    def after_batch(self, l):
        if l.is_training:
            if self.b is None and self.e is None: self.b = max(int((len(l.dl_train)*l.n_epochs) / self.times) , 1)
            if self.b and l.cur_batch % self.b == 0:
                self._plot(l)
    def after_epoch_full(self, l):
        if l.is_fitting:
            if self.e and l.cur_epoch % self.e == 0:
                self._plot(l)