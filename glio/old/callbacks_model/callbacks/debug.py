from ..model import Callback
import torch

class DetectAnomaly(Callback):
    def enter(self, l):
        self.da = torch.autograd.detect_anomaly()
        self.da.__enter__()
    def exit(self, l): self.da.__exit__()