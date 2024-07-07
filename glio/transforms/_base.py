from collections.abc import Callable
from abc import ABC, abstractmethod
import random

class Transform(ABC):
    @abstractmethod
    def forward(self, x): raise NotImplementedError(self.__class__.__name__ + " doesn't have `transform` method.")
    def reverse(self, x): raise NotImplementedError(self.__class__.__name__ + " doesn't have `reverse` method.")
    def __call__(self, x): return self.forward(x)

class RandomTransform(Transform, ABC):
    p:float
    def __call__(self, x):
        if random.random() < self.p: return self.forward(x)
        return x
