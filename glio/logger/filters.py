from typing import TYPE_CHECKING
if TYPE_CHECKING: import torch

def does_something(module: 'torch.nn.Module'):
    """Фильтрует модули, имеющие параметры или не являющиеся контейнерами"""
    if len(list(module.parameters(recurse=False))) > 0: return True # содержит параметры
    if len(list(module.buffers(recurse=False))) > 0: return True # содержит буферы
    if len(list(module.modules())) > 1: return False # не содержит параметры и буферы, но содержит модули (сам модуль возвращается методом modules поэтому > 1)
    return True # не содержит параметры, буферы и модули
