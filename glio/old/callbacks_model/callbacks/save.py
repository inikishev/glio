from ..model import Callback
import os
import torch


class Save_Best(Callback):
    def __init__(self, metric = 'test loss', value = 'min', mode = 'state_dict_full', name = None, postfix = None, keep_last_only = True):
        '''Value - `min`/`max`; Mode - `state_dict`/`model`'''
        self.metric = metric
        self.value = value.lower()
        self.mode = mode.lower()
        self.name = name
        self.postfix = postfix
        self.best = float('inf') if value == 'min' else -float('inf')
        
        self.keep_last_only = keep_last_only
        
        self.last_path = None
        
        if not os.path.exists('models'): os.mkdir('models')
        
        self.definition_saved = False
        
    def after_batch(self, l):
        if self.metric in l.logs:
            if self.name is None: self.name = l.name
            if self.name is None: raise ValueError('Save_Best: `name` must be specified in either Learner or this callback')
            if self.postfix is not None:
                self.name += self.postfix
                self.postfix = None
            
            if l.is_fitting:
                #print(f'{l.logs.last(self.metric) = }', f'{self.best = }', self.value, (l.logs.last(self.metric) < self.best) if self.value == 'min' else (l.logs.last(self.metric) > self.best) )
                if (l.logs.last(self.metric) < self.best) if self.value == 'min' else (l.logs.last(self.metric) > self.best):
                    self.best = l.logs.last(self.metric)
                    
                    if self.mode == 'state_dict':
                        if self.last_path is not None and self.keep_last_only and os.path.isfile(self.last_path): os.remove(self.last_path)
                        self.last_path = f'models/{self.name} {self.metric} = {float(self.best)}.pt_state_dict'
                        torch.save(l.model.state_dict(), self.last_path)
                        if not self.definition_saved:
                            with open(f'models/{self.name}.txt', 'w') as f: f.write(str(l.model)) # saving model definition
                            self.definition_saved = True
                            
                    elif self.mode == 'model':
                        if self.last_path is not None and self.keep_last_only and os.path.isfile(self.last_path): os.remove(self.last_path)
                        self.last_path = f'models/{self.name} {self.metric} = {float(self.best)}.pt_model'
                        torch.save(l.model.state_dict(), self.last_path)
                        
                    elif self.mode == 'state_dict_full':
                        if self.last_path is not None and self.keep_last_only and os.path.isfile(self.last_path): os.remove(self.last_path)
                        self.last_path = f'models/{self.name} {self.metric} = {float(self.best)}.pt_state_dict_full'
                        torch.save({
                            'model_state_dict': l.model.state_dict(),
                            'optimizer_state_dict': l.opt.state_dict(),
                            'logs': l.logs,
                            }, self.last_path)
                        if not self.definition_saved:
                            with open(f'models/{self.name}.txt', 'w') as f: f.write(str(l.model)) # saving model definition
                            self.definition_saved = True

class Save_On(Callback):
    def __init__(self, mode = 'state_dict', step_batch = None, step_epoch = None, on_fit = True, name = None, postfix = None, postfix_metric = 'test loss'):
        self.mode = mode
        self.name = name
        self.postfix = postfix
        self.postfix_metric = postfix_metric
        
        self.step_batch = step_batch
        self.step_epoch = step_epoch
        self.on_fit = on_fit
        
        if not os.path.exists('models'): os.mkdir('models')
    
    def _save(self, l):
        if self.name is None: self.name = l.name
        if self.postfix is not None:
            self.name += self.postfix
            self.postfix = None
        if self.postfix_metric is not None:
            self.name = f'{self.name} {self.postfix_metric} = {l.logs.last(self.postfix_metric)}'
            
        info_on = f'{l.total_batches} batches, {l.total_epochs} epochs'
        
        if self.mode == 'state_dict':
            torch.save(l.model.state_dict(), f'models/{self.name} ({info_on}).pt_state_dict')
            with open(f'{self.name}', 'w') as f: f.write(str(l.model)) # saving model definition
                
        elif self.mode == 'model':
            torch.save(l.model.state_dict(), f'models/{self.name} ({info_on}).pt_model')
            
        elif self.mode == 'state_dict_full':
            torch.save({
                'model_state_dict': l.model.state_dict(),
                'optimizer_state_dict': l.opt.state_dict(),
                'logs': l.logs,
                }, f'models/{self.name} ({info_on}).pt_state_dict_full')
            with open(f'{self.name}', 'w') as f: f.write(str(l.model)) # saving model definition
    
    def after_batch(self, l):
        if l.is_training and self.step_batch is not None and l.total_batches % self.step_batch == 0: self._save(l)
    def after_epoch(self, l):
        if l.is_training and self.step_epoch is not None and l.total_epochs % self.step_epoch == 0: self._save(l)
    def after_fit(self, l):
        if l.is_fitting and self.on_fit: self._save(l)
        
        

def load(path, model: torch.nn.Module = None, opt: torch.optim.Optimizer = None):
    if path.lower().endswith('.pt_state_dict'):
        if model is None: raise ValueError(f'`model_cls` is must be specified for loading state dict from `{path}`')
        model.load_state_dict(torch.load(path))
        return model
    
    elif path.lower().endswith('.pt_model'):
        return torch.load(path)
    
    elif path.lower().endswith('.pt_state_dict_full'):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        logs = checkpoint['logs']
        return model, opt, logs
