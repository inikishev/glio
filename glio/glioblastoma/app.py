# Автор - Никишев Иван Олегович группа 224-31

from functools import partial
import io
import base64
import os
import random
import logging
import PIL.Image
import flet as ft
import torch, torchvision
from torchvision import transforms
from .DICOMFolder import DICOMDir, Study, Series, Patient

test_base64_string = 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII='

def to_base64(img_array, max):
    """Преобразует тензор в строку в формате Base64, кодирующую изображение"""
    transform = transforms.ToPILImage()
    pil_image = transform(img_array/max)
    image_bytes = pil_image.tobytes()
    base64_string = base64.b64encode(image_bytes).decode()

    return base64_string

def imgarr_to_base64(img_array, max):
    img_file = io.BytesIO()
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    #PIL.Image.fromarray(img_array).save(img_file, format="png")
    torchvision.utils.save_image(img_array/max, fp=img_file, format='png')
    img_file.seek(0)
    img_base64 = base64.b64encode(img_file.read()).decode()
    return img_base64


def to_file(tensor, max):
    """Преобразует тензор в изображение и сохраняет на жёстком диске как временный файл, возващает путь к изображению"""
    path = f'программа дифференциации некроза и рецидива глиобластомы TEMP - {random.uniform(0,1)} temp.png'
    torchvision.utils.save_image(tensor/max, path)
    return path

def clean_temp():
    """Очистка временных файлов"""
    for i in os.listdir('./'):
        if i.startswith('программа дифференциации некроза и рецидива глиобластомы TEMP - '):
            try:
                os.remove(i)
            except (PermissionError, FileNotFoundError): pass

class GlioApp:
    def __init__(self, classifier):
        self.classifier = classifier
        self.temp_controls = []

    def main(self, page: ft.Page):
        """Основная функция, являющаяся аргументом `target` для `ft.run`"""
        self.page = page
        #self.page.auto_scroll = True

        # Заданеи темы
        self.page.theme = ft.Theme(
            color_scheme=ft.ColorScheme(primary=ft.colors.BLACK, secondary=ft.colors.BLACK, tertiary=ft.colors.BLACK, background=ft.colors.WHITE),
            visual_density=ft.ThemeVisualDensity.COMPACT
            )
        self.page.title = "Никишев И.О. Программа дифференциации некроза и рецидива глиобластомы"

        # Задание настроек страницы
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.scroll = ft.ScrollMode.AUTO
        self.appbar = ft.AppBar(title=ft.Text("Никишев И.О. Программа дифференциации некроза и рецидива глиобластомы"))
        self.page.appbar = self.appbar
        self.page.update()
        self.folder_pick_menu()


    def folder_pick_menu(self):
        self.clear_temp()

        self.add_temp(ft.Text('Никишев И.О.\nПрограмма дифференциации некроза и рецидива глиобластомы'))
        # Выбор папки с обследованиями
        self.folder_picker = ft.FilePicker(on_result=self.on_folder_picked)
        self.page.overlay.append(self.folder_picker)
        self.folder_picker_button = ft.ElevatedButton(
                "Выберите папку с обследованиями",
                on_click=lambda _: self.folder_picker.get_directory_path(
                    "Пожалуйста, выберите папку с обследованиями"
                ),
            )
        self.page.add(self.folder_picker_button)

        # Использование выбранной папки
        if hasattr(self, 'file_picker_event'): self.on_folder_picked(None)

    def clear_temp(self):
        """Очищаяет и удаляет временные элементы интерфейса"""
        for temp_ctrl in self.temp_controls:
            #print(temp_ctrl, 'removed')
            self.page.remove(temp_ctrl)
            self.temp_controls = []
        if hasattr(self, 'pred_text'):
            try:
                self.page.remove(self.pred_text)
            except ValueError: pass

    def add_temp(self, control):
        """Добавляет временный элемент интерфейса"""
        self.temp_controls.append(control)
        self.page.add(self.temp_controls[-1])

    def on_search(self, e: ft.ControlEvent):
        """Обработчик поиска"""
        # Возможность поменять выбранную папку
        self.clear_temp()

        self.search = e.data if e is not None else None
        if self.search == '': self.search = None
        #print(type(self.search))

        # Интерфейс выбора обследований по пациентам
        self.study_cards = []
        for i in self.patients:
            studies = [s for s in sorted(i.studies, key = lambda x: x.date if hasattr(x, 'date') else x.age) if s._init]

            # Обработка поиска
            studies = [s for s in studies if self.search is None or self.search.lower() in str(s).lower() or self.search.lower() in str(i).lower()]

            # Создание интерфейса выбора обследований
            if len(studies) > 0:
                self.study_cards.append(ft.Card(
                    content = ft.Container(border = ft.border.all(1, ft.colors.GREY_400), padding = 8,

                        content=ft.Column(
                            [
                                ft.Text(str(i)),
                                *[ft.OutlinedButton(str(s), style=ft.ButtonStyle(shape = ft.RoundedRectangleBorder()), on_click=lambda _: self.on_study_picked(study=s, patient = i)) for s in studies],
                            ]
                        )
                    )))
                #self.study_cards.append(ft.Divider(height=1,color='black'))

        # Не найдены обследования
        if len(self.study_cards) == 0 and self.search is None:
            self.cur_error = ft.Text('ОШИБКА: Не найдены обследования, соответствующие области применения ПО.\n\nВ директории или поддиректориях должно содержаться 176 файлов обследования стандартным методом МРТ размерностью 256 на 232 пикселя. В директории или поддиректориях должно содержаться 20 файлов парамерических карт объёма церебрального кровотока размерностью 128 на 128 пикселей. Файлы должны иметь набор информационных объектов и пиксельную информацию в соответствии со стандартом “Digital Imaging and Communications in Medicine PS3.1”')
            self.add_temp(self.cur_error)

        if len(self.study_cards) == 0 and self.search is not None:
            self.cur_error = ft.Text(f'ОШИБКА: К сожалению, по запросу «{self.search}» не найдено обследований. Пожалуйста, проверьте правильность введённого запроса.')
            self.add_temp(self.cur_error)

        # Отображение обследований
        else: self.add_temp(ft.ListView([ft.Text("Пожалуйста, выберите необходимое обследование")] + self.study_cards, spacing=10, padding=20))


    def on_folder_picked(self, e: ft.FilePickerResultEvent):
        """Обработчик выбора папки с обследованиями"""
        self.file_picker_event = e

        # Поле поиска
        self.appbar.actions = [ft.TextField(label="Поиск обследований", border="underline", hint_text="Введите любое поле пациента/обследования",
                                            on_change=self.on_search)]
        self.appbar.update()

        # Загрузка обследований
        if e is not None:
            self.path = e.path
            self.loading = ft.Text('Загружаются обследования, пожалуйста, подождите...')
            self.page.add(self.loading)
            self.dir = DICOMDir(e.path)
            # self.patients = observation.dicom_load(e.path)

        self.on_search(None)

        # Не найдены обследования
        # if len(self.study_cards) == 0:
        #     self.add_temp(ft.Text('ОШИБКА: Не найдены обследования, соответствующие области применения ПО.\n\nВ директории или поддиректориях должно содержаться 176 файлов обследования стандартным методом МРТ размерностью 256 на 232 пикселя. В директории или поддиректориях должно содержаться 20 файлов парамерических карт объёма церебрального кровотока размерностью 128 на 128 пикселей. Файлы должны иметь набор информационных объектов и пиксельную информацию в соответствии со стандартом “Digital Imaging and Communications in Medicine PS3.1”'))
        # Удаление информации о загрузке обследований
        if e is not None: self.page.remove(self.loading)

    def get_slice(self, s, axis) -> list[str]:
        """Возращает список путей к временным файлам изображений"""
        try:
            if axis == 0: return [imgarr_to_base64(im[s], self.max[i]) for i,im in enumerate(self.images)]
            if axis == 1: return [imgarr_to_base64(im[:,s], self.max[i]) for i,im in enumerate(self.images)]
            if axis == 2: return [imgarr_to_base64(im[:,:,s], self.max[i]) for i,im in enumerate(self.images)]
        except IndexError: return [imgarr_to_base64(im[-1] if axis==0 else im[:,-1] if axis==1 else im[:,:,-1], self.max[i]) for i,im in enumerate(self.images)]

    def get_loc(self):
        """Определение нахождения текущего среза"""
        if self.slice_locations is None: return str(self.coord)

        elif self.axis == 0: return f'{self.slice_locations[int(self.coord)]} мм.'
        elif self.axis == 1: return f'{self.study.get_pixel_spacing()[0] * (self.coord - self.shape[1]/2)} мм.'
        elif self.axis == 2: return f'{self.study.get_pixel_spacing()[1] * (self.coord - self.shape[2]/2)} мм.'

    def on_axis_changed(self,e: ft.ControlEvent):
        """Обновление страницы при смене плоскости разреза"""
        #clean_temp()
        value = e.control.value
        self.axis = int(value)
        self.image_displays.controls = [ft.Image(src_base64=i, fit=ft.ImageFit.FILL, height=int(self.scale), gapless_playback = True) for i in self.get_slice(self.coord, self.axis)]
        self.image_displays.update()

        self.slider_text.value = f'Текущая координата среза - {self.get_loc()}.'
        self.slider_text.update()
        #self.page.add(self.image_displays)
        #self.page.update()

    def on_coord_changed(self,e: ft.ControlEvent):
        """Обновление страницы при смене координаты разреза"""
        #clean_temp()
        value = e.control.value
        self.coord = int(value)
        self.image_displays.controls = [ft.Image(src_base64=i, fit=ft.ImageFit.FILL, height=int(self.scale), gapless_playback = True) for i in self.get_slice(self.coord, self.axis)]
        self.image_displays.update()

        self.slider_text.value = f'Текущая координата среза - {self.get_loc()}'
        self.slider_text.update()
        #self.page.add(self.image_displays)
        #self.page.update()

    def on_scale_changed(self, e: ft.ControlEvent):
        #clean_temp()
        self.scale = int(e.control.value)
        self.image_displays.controls = [ft.Image(src_base64=i, fit=ft.ImageFit.FILL, height=int(self.scale), gapless_playback=True) for i in self.get_slice(self.coord, self.axis)]
        self.image_displays.update()

    def update_image_display(self):
        """Инициализация визуализации разреза"""
        #clean_temp()
        self.image_displays.controls = [ft.Image(src_base64=i, fit=ft.ImageFit.FILL, height=int(self.scale), gapless_playback=True) for i in self.get_slice(self.coord, self.axis)]
        self.image_displays.update()
        #self.page.add(self.image_displays)
        #self.page.update()


    def on_study_picked(self, study: Study, patient: Patient):
        """Обработчик выбора пациента"""
        self.appbar.actions = []
        self.appbar.update()

        # присваивание обследования
        self.study = study
        self.patient = patient
        print(f"Выбрано: {study.uitext()}")

        # Интерфейс обследования
        self.clear_temp()
        self.page.remove(self.folder_picker_button)
        self.add_temp(ft.ElevatedButton('Назад', on_click=lambda _: self.folder_pick_menu()))
        self.add_temp(ft.Text('Пожалуйста, удостоверьтесь, что выбрано необходимое обследование. Воспользуйтесь информацией и полосой прокрутки, чтобы подтвердить, что выбрано изображение рецидивирующей глиобластомы.'))

        # Загрузка 3D изображений
        self.images = []
        if 'trabit' in self.path.lower():
            self.images.append(torch.rot90(self.study.get_mods_tensor()[0], 1, (0,2)))
            self.slice_locations = None
        else:
            img = self.study.get_mods_tensor()
            self.images.extend([torch.rot90(t, 2, (0,2)) for t in img])
            self.slice_locations = self.study.get_slice_loc()
            self.slice_locations.extend([self.slice_locations[-1]]*100)

        # Определение параметров нормализации изображений для отображения
        self.max = [i.max() for i in self.images]
        self.slider_max = max([max(tuple(i.shape)) for i in self.images])

        self.shape = self.images[0].shape

        self.coord = 100
        self.axis = 0
        self.scale = 300

        # Элементы управления координатой среза и плоскостью среза
        self.add_temp(ft.RadioGroup(content=ft.Row([
                        ft.Radio(value=0, label="аксиальная плоскость разреза"),
                        ft.Radio(value=1, label="саггитальная плоскость разреза"),
                        ft.Radio(value=2, label="корональная плоскость разреза")]), value = 0, on_change=self.on_axis_changed))
        self.add_temp(ft.Row(controls = [ft.Text('Масштаб изображений'),
                                                       ft.Slider(min=0, max=600, divisions=600, round = 0, on_change = self.on_scale_changed, value=300, label="Масштаб {value}"),
                                                       ft.Text('Координата среза'),
                                                       ft.Slider(min=0, max=self.slider_max, divisions=1000, round = 0, value=0, on_change = self.on_coord_changed, label="Координата {value}")]))
        # self.add_temp(ft.Row(tight = True, controls = [ft.Text('Координата среза'), ft.Slider(min=0, max=self.slider_max, divisions=1000, round = 0, value=0, on_change = self.on_coord_changed, label="Координата {value}")]))
        self.slider_text = ft.Text()
        self.add_temp(self.slider_text)

        # Информация о пациенте
        self.study_info = ft.Container(ft.Text(f'{str(self.patient)}\n{str(self.study)}'),border = ft.border.all(1, ft.colors.GREY_200), padding=4)

        # Отображение среза
        self.image_displays = ft.Row(tight=True)
        self.add_temp(self.image_displays)
        self.add_temp(self.study_info)
        self.update_image_display()

        # Функция расчёта вывода модели
        self.add_temp(ft.ElevatedButton('Рассчитать вывод модели', on_click=self.inference))

    def inference(self, e):

        # Возможность перерасчёта
        if hasattr(self, 'pred_text'):
            try: self.page.remove(self.pred_text)
            except ValueError: pass

        # Информация о загрузке
        self.inference_loading = ft.Text('Рассчитывается предсказание модели, пожалуйста подождите...')
        self.page.add(self.inference_loading)

        logging.warning(f'{self.study.get_mods_tensor().shape = }')
        # Рассчет предсказания

        logging.warning(f'{self.path = }')
        if 'trabit' not in self.path.lower():
            # Проверка на соответствие выбранного обследования области применения ПО
            if tuple(self.study.get_mods_tensor().shape) != (2, 176, 256, 232) or len(self.study.paths_t1)!= 176 or len(self.study.paths_perf) not in (19,20):
                self.add_temp(ft.Text(f'ОШИБКА: выбранное обследование не соответствует требованиям ко входным данным ПО.\n\nВ директории или поддиректориях должно содержаться 176 файлов обследования стандартным методом МРТ размерностью 256 на 232 пикселя. В директории или поддиректориях должно содержаться 20 файлов парамерических карт объёма церебрального кровотока размерностью 128 на 128 пикселей. Файлы должны иметь набор информационных объектов и пиксельную информацию в соответствии со стандартом “Digital Imaging and Communications in Medicine PS3.1. Выбранное обследование содержит {len(self.study.paths_t1)} изображений МРТ и {len(self.study.paths_perf)} изображений параметрических карт. Пожалуйста, убедитесь в правильности выбранной директории.',color='dark red'),)
                self.predictions = None
            else: self.predictions = 'Ошибка - классификатор в разработке'
        else:
            self.predictions = self.classifier(self.study.tensor().unsqueeze(0))
        if self.predictions is not None:
            self.pred_text = ft.Text()
            self.page.add(self.pred_text)
            self.pred_text.value = f'Вывод модели: {self.predictions}'
            self.page.remove(self.inference_loading)
            self.pred_text.update()

    def run(self):
        ft.app(target=self.main)
