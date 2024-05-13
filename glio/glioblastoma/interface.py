import os,shutil
import gradio as gr
from monai.networks.nets import SwinUNETR # type:ignore
from .dicomseg import save_seg,save_seg_mock
from .exceptions import GradioException

WEIGHTS = r"F:\Stuff\Programming\experiments\vkr\BRATS2024 2D segm v2 overnight\SwinUNETR lr=0.003 bs=32 loss = monai.losses.dice.DiceFocalLoss opt=MADGRAD sch=NoneType\1. 21.04.2024 16-10-47 (58-197430; testloss=0.14171; testacc=0.98250)"
WEIGHTS_LINUX = '/mnt/f/Stuff/Programming/experiments/vkr/BRATS2024 2D segm v2 overnight/SwinUNETR lr=0.003 bs=32 loss = monai.losses.dice.DiceFocalLoss opt=MADGRAD sch=NoneType/1. 21.04.2024 16-10-47 (58-197430; testloss=0.14171; testacc=0.98250)'
MODEL = SwinUNETR((96,96), 4, 4, spatial_dims = 2)
LINUX = True

TEMPFOLDER = '/mnt/f/Stuff/Programming/AI/glio_diff/glio/glioblastoma/GLIOTEMP'
def segment(files, progress = gr.Progress(True)):
    try:
        if os.path.exists(TEMPFOLDER): 
            shutil.rmtree(TEMPFOLDER)
        os.mkdir(TEMPFOLDER)
        progress(0.01, "Загрузка файлов, пожалуйста подождите...")
        progress(0.015, "Обработка файлов, пожалуйста подождите...")
        checkpoint_path = WEIGHTS_LINUX if LINUX else WEIGHTS
        filepaths = [i.name for i in files]
        progress(0.02, "Инициализация, пожалуйста подождите...")
        res = save_seg(filepaths, TEMPFOLDER, checkpoint_path, MODEL, progress=progress)
        progress(1, "Сегментация произведена. Пожалуйста, загрузите файлы")
        return res
    except GradioException as e: raise gr.Error(str(e) + '\n Пожалуйста, представьте обследования в форме, указанной в пункте «Требования к организации входных данных» документа технического задания.')
    except Exception as e: raise gr.Error(f"Ошибка: {e}")

interface = gr.Interface(
    segment,
    [
        gr.Files(file_count="directory", type="filepath", height = 100, label = 'Выберите директорию с обследованиями DICOM'), # type:ignore
    ],
    [
        gr.File()
    ],
    title = 'Автор - Никишев И.О.',
    description = '''Пожалуйста, выберите папку с необходимым обследованием и нажмите `Отправить`.
    
    В директории должно находиться четыре исследования - Т1 до и после контраста, Т2 и FLAIR. В директории могут находиться исследования других модальностей. Если одно из четырёх необходимых исследований отсутствует, будет выведено сообщение об ошибке, указывающее на отсутствующую модальность. 
    
    После нажатия на кнопку необходимо дождаться предобработки и сегментации изображений. Предобработка включает в себя корегистрацию, удаление черепа и нормализацию. Время предобработки и сегментации не превышает десяти минут. Пользовательский интерфейс отображает прогресс обработки.
    
    По завершении обработки нажмите на кнопку "Загрузить". Будет загружен файл DICOM стандарта DICOM-SEG с аттрибутами выбранного обследования.
    
    Для программного взаимодействия с ПО, запущенном на сервере, воспользуйтесь инструкцией в `Использование серверного API`.''',
    article='''''',
    allow_flagging = 'never',
    theme = gr.themes.Soft(),
)

interface.queue().launch()