from ..design.CallbackModel import Callback
from .learner import Learner

class LRFinderPriming(Callback):
    def __init__(self, start = 1e-6, stop = 1, step = 1.3, max_increase = 3, niter = 2):
        pass