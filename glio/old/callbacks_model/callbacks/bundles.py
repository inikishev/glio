from . import accelerate, hooks, logging, progress, save, summary

class Bundle: 
    def __init__(self, callbacks):
        self.callbacks = callbacks
        
    def __getitem__(self, x):
        return self.callbacks[x]

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks)
    
class CLASSIFICATION(Bundle):
    """
    callbacks list:
    
    Accelerate, Accuracy, Loss, ActMeanStd, ActHist, Print, FastProgress, SaveBest, SaveOn, Summary
    """
    def __init__(self, name, step = 1, times = None, print_step_epoch = 1, loss_smooth = None, precision = 'fp16'):
        self.callbacks =Bundle([
            accelerate.Accelerate(precision = precision),
            logging.Metric_Accuracy(step = step, times = times),
            logging.Metric_Loss(step = step if step is not None else 1),
            hooks.Log_ActHist(times = 256),
            hooks.Log_ActMeanStd(times = 128),
            progress.Print(metrics = ['test loss', 'test accuracy'], step_epoch=print_step_epoch),
            progress.FastProgressBar(metrics = ['train loss', 'test loss'], plot = True, step_batch = 16, smooth=[loss_smooth, None]),
            save.Save_Best(metric = 'test loss', value = 'min', mode = 'state_dict_full', name = name, postfix = None, keep_last_only = True),
            save.Save_Best(metric = 'test accuracy', value = 'min', mode = 'state_dict_full', name = name, postfix = None, keep_last_only = True),
            save.Save_On(mode = 'state_dict_full', name = name),
            summary.Summary(plot = [['train loss', 'test loss'], ['train accuracy', 'test accuracy'], ['act mean'], ['act std']], hist = ['hist'], text = ['test loss', 'test accuracy'])
            ])
        
class REGRESSION(Bundle):
    """
    callbacks list:
    
    Accelerate, Accuracy, Loss, ActMeanStd, ActHist, Print, FastProgress, SaveBest, SaveOn, Summary
    """
    def __init__(self, name, step = 1, times = None, print_step_epoch = 1, loss_smooth = None, precision = 'fp16'):
        self.callbacks =Bundle([
            accelerate.Accelerate(precision = precision),
            logging.Metric_Loss(step = step, times = times),
            hooks.Log_ActHist(times = 256),
            hooks.Log_ActMeanStd(times = 128),
            progress.Print(metrics = ['test loss'], step_epoch=print_step_epoch),
            progress.FastProgressBar(metrics = ['train loss', 'test loss'], plot = True, step_batch = 16, smooth=[loss_smooth, None]),
            save.Save_Best(metric = 'test loss', value = 'min', mode = 'state_dict_full', name = name, postfix = None, keep_last_only = True),
            save.Save_On(mode = 'state_dict_full', name = name),
            summary.Summary(plot = [['train loss', 'test loss'], ['act mean'], ['act std']], hist = ['hist'], text = ['test loss'])
            ])
        
class AUTOENCODER(Bundle):
    """
    callbacks list:
    
    Accelerate, Accuracy, Loss, ActMeanStd, ActHist, Print, FastProgress, SaveBest, SaveOn, Summary
    """
    def __init__(self, name, step = 1, times = None, print_step_epoch = 1, loss_smooth = None, precision = 'fp16'):
        self.callbacks =Bundle([
            accelerate.Accelerate(precision = precision),
            logging.Metric_Loss(step = step, times = times),
            hooks.Log_ActHist(times = 256),
            hooks.Log_ActMeanStd(times = 128),
            progress.Print(metrics = ['test loss'], step_epoch=print_step_epoch),
            progress.FastProgressBar(metrics = ['train loss', 'test loss'], plot = True, step_batch = 16, smooth=[loss_smooth, None]),
            save.Save_Best(metric = 'test loss', value = 'min', mode = 'state_dict_full', name = name, postfix = None, keep_last_only = True),
            save.Save_On(mode = 'state_dict_full', name = name),
            summary.Summary(plot = [['train loss', 'test loss'], ['act mean'], ['act std']], hist = ['hist'], text = ['test loss'])
            ])