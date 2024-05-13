from ..model import Callback

class Summary(Callback):
    def __init__(self, plot = [['train loss', 'test loss'], ['train accuracy'], ['test accuracy']], hist = ['hist'], text = ['test loss']):
        self.plot = plot
        self.hist = hist
        self.text = text
        
    def after_fit(self, l):
        if l.is_fitting: 
            for key in self.plot:
                if l.logs.has_substring(key) if isinstance(key, str) else any(l.logs.has_substring(k) for k in key): l.logs.plot_all(*key)
            for key in self.hist:
                if l.logs.has_substring(key) if isinstance(key, str) else any(l.logs.has_substring(k) for k in key): l.logs.hist_all(*key)
            for key in self.text:
                if key in l.logs: print(l.logs.last(key))