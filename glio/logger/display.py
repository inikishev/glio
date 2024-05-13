from .hooks import ForwardHook
from typing import TYPE_CHECKING
from .filters import does_something
from ..visualize import Visualizer
if TYPE_CHECKING: 
    import torch, torch.utils.hooks
    
from ..python_tools import type_str
import math
class Display_Signal(ForwardHook):
    def __init__(self, model: 'torch.nn.Module', filter = does_something, grid = 4):
        self.grid = grid
        self.vis = Visualizer()
        super().__init__(model, filter)
        
    def hook(self, module: 'torch.nn.Module', input: 'torch.Tensor', output: 'torch.Tensor', name:str):
        name = f'{name}|{type_str(type(module))}'
        
        self.is_3D = False
        if len(self.vis) == 0: 
            inpx = input[0] if isinstance(input, (list,tuple)) else input
            inp_shape = inpx.shape
            if inpx.ndim == 5: 
                self.is_3D = True
                inpx = inpx[:,:,[int(inpx.shape[2]/2)]].squeeze(2)
            self.vis.imshow(inpx[0], mode='chw', label = f'INPUT\n{tuple(inp_shape)}')
            
        if hasattr(module, "named_parameters"):
            for param in module.named_parameters(): 
                data = param[1].data
                data_shape = data.shape
                # Если вход 3Д, берётся срез посередине
                if self.is_3D and data.ndim in (4,5): data = data[:,[int(data.shape[1]/2)]].squeeze(1)
                
                if data.ndim == 2: self.vis.imshow(data, mode='hw', label = f'{name}\n{param[0]}\n{tuple(data_shape)}')
                # веса не имеют пакетного измерения; например, conv2d может иметь размерности (out=16, in=3, kernel=*(5, 5))
                elif data.ndim == 3: 
                    # если 3 или  меньше канала, они объединяются в цветное изображение
                    if data.size(0) <= 3: self.vis.imshow(data, mode='chw', label = f'{name}\n{param[0]}\n{tuple(data_shape)}')
                    # если больше, создаётся решётка
                    else: self.vis.imshow_grid(data, mode = 'filters', nelems = self.grid, label = f'{name}\n{param[0]}\n{tuple(data_shape)}')
                elif data.ndim == 4: 
                    self.vis.imshow_grid(data, mode = 'filters', nelems = self.grid, label = f'{name}\n{param[0]}\n{tuple(data_shape)}')
                else: ...
        
        # выходной сигнал
        # если 3D, берётся срез посередине
        outpx = output[0,:,[int(output.shape[2]/2)]].squeeze(2) if output.ndim == 5 else output[0]
        # проверка что выход не 1D
        if outpx.ndim > 1:
            # если 3 или  меньше канала, они объединяются в цветное изображение
            if outpx.size(0) <= 3: self.vis.imshow(outpx, mode='chw', label = f'{name}\nOUTPUT\n{tuple(output.shape)}')
            # если больше, создаётся решётка
            else: self.vis.imshow_grid(outpx, mode='bhw', nelems = self.grid, label = f'{name}\nOUTPUT\n{tuple(output.shape)}')

    def remove(self): 
        for h in self.hooks: h.remove()
        self.hooks: list["torch.utils.hooks.RemovableHandle"] = []
        self.vis.show(nrows = int(math.ceil(len(self.vis)) ** 0.5), figsize = (30, 30))
        self.vis.clear()