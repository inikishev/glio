# Автор - Никишев Иван Олегович группа 224-31

import torch
import torchvision.transforms.v2
import logging

INSTALLED_IMAGE_LIBS = []
NOT_INSTALLED_IMAGE_LIBS = []
try: 
    import torchvision.io
    INSTALLED_IMAGE_LIBS.append('torchvision.io')
except ModuleNotFoundError: NOT_INSTALLED_IMAGE_LIBS.append('torchvision.io')
try: 
    import skimage.io
    INSTALLED_IMAGE_LIBS.append('skimage.io')
except ModuleNotFoundError: NOT_INSTALLED_IMAGE_LIBS.append('skimage.io')
try: 
    import cv2
    INSTALLED_IMAGE_LIBS.append('cv2')
except ModuleNotFoundError: NOT_INSTALLED_IMAGE_LIBS.append('cv2')
try: 
    import PIL.Image
    INSTALLED_IMAGE_LIBS.append('PIL.Image')
except ModuleNotFoundError: NOT_INSTALLED_IMAGE_LIBS.append('PIL.Image')
try: 
    import pyvips # type: ignore
    INSTALLED_IMAGE_LIBS.append('pyvips')
    INSTALLED_IMAGE_LIBS.append('pyvips sequential')
except (ModuleNotFoundError, OSError): NOT_INSTALLED_IMAGE_LIBS.append('pyvips')

def imread(path, lib= 'auto', libs = INSTALLED_IMAGE_LIBS, warn_errors = False):
    path = path.replace('\\', '/')

    PIL_transform = torchvision.transforms.v2.PILToTensor()
    float_transform = torchvision.transforms.v2.ToDtype(torch.float32, scale=True)
    if lib == 'auto':
        for i in libs:
            try:
                return imread(path, lib=i, warn_errors=warn_errors)
            except Exception as e:
                if warn_errors: logging.warning(e)
        else: raise ValueError(f'Could not read image at {path} with any of {libs}')

    elif lib == 'PIL.Image': image = PIL_transform(PIL.Image.open(path)) # pyright:ignore[reportPossiblyUnboundVariable]

    elif lib == 'torchvision.io': image = torchvision.io.read_image(path)

    elif lib == 'skimage.io': image = torch.as_tensor(skimage.io.imread(path))# pyright:ignore[reportPossiblyUnboundVariable]

    elif lib == 'cv2': 
        image = cv2.imread(path)# pyright:ignore[reportPossiblyUnboundVariable]
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# pyright:ignore[reportPossiblyUnboundVariable]
        image = torch.as_tensor(image)
        
    elif lib == 'pyvips': image = torch.as_tensor(pyvips.Image.new_from_file(path))# pyright:ignore[reportPossiblyUnboundVariable]
    elif lib == 'pyvips sequential': image = torch.as_tensor(pyvips.Image.new_from_file(path, access='sequential'))# pyright:ignore[reportPossiblyUnboundVariable]
    else: raise ValueError(f'Unknown image library {lib}')
    if lib in ('cv2', 'skimage.io', 'pyvips', 'pyvips sequential') and image.dim() == 3: image = torch.permute(image, (2, 0, 1))
    if image.dim() == 2: image = torch.unsqueeze(image, 0) 

    #print(f'{path}, {lib}, {image.shape}')
    return float_transform(image)
