from ..model import Callback, Cancel

class Nothing(Callback): pass
class OneBatch(Callback):
    """Stops fit after one batch"""
    def after_batch(self, l):
        if l.cur_batch >= 0: raise Cancel('fit')
        
class StopFitOnBatch(Callback):
    def __init__(self, batch): 
        self.batch = batch
    def after_batch(self, l):
        if l.cur_batch >= self.batch: raise Cancel('fit')

class StopFitOnTotalBatches(Callback):
    def __init__(self, batch): 
        self.batch = batch
    def after_batch(self, l):
        if l.total_batches >= self.batch: raise Cancel('fit')

class StopFitOnTotalAnyBatches(Callback):
    def __init__(self, batch): 
        self.batch = batch
        self.cur = 0
    def after_batch(self, l):
        self.cur+=1
        if self.cur >= self.batch: raise Cancel('fit')

class FitStatus(Callback):
    """Overrides status on fit"""
    def __init__(self, status): self.status = status
    def before_fit(self, l): 
        self.backup = l.status
        l.status = self.status
    def exit(self, l): l.status = self.backup
    
