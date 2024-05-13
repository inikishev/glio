# Автор - Никишев Иван Олегович группа 224-31

import pydicom
import os
import torch
import scipy.ndimage
import numpy as np
from functools import partial
from torchvision.transforms import v2
import logging
import csv

from ..loaders import dicom, nifti

def date_str(date:str): return f'{date[0:4]}.{date[4:6]}.{date[6:8]}'
class Patient:
    def __init__(self, dcm: pydicom.FileDataset):
        self.ID = dcm.PatientID
        self.age = int(dcm.PatientAge[1:-1]) # 057Y => 57
        
        self.birth_day_num = int(dcm.PatientBirthDate)
        self.birth_day = dcm.PatientBirthDate # 19760908 => 1976.09.08
        self.birth_day = f'{self.birth_day[0:4]}.{self.birth_day[4:6]}.{self.birth_day[6:8]}'
        
        self.birth_time_num = float(dcm.PatientBirthTime)
        self.birth_time = dcm.PatientBirthTime
        
        self.name = dcm.PatientName
        self.position = dcm.PatientPosition
        self.sex = dcm.PatientSex
        self.height = float(dcm.PatientSize)
        self.weight = int(dcm.PatientWeight)
        self.studies: list[Study] = []
        
    def __str__(self):
        return f"""Пациент {self.name}
ID: {self.ID}
дата и время рождения: {self.birth_day} {self.birth_time}
возраст: {self.age}
пол: {self.sex}
вес: {self.weight}
даты обследований: {'; '.join([date_str(str(j)) for j in sorted([int(i.date) for i in self.studies])])}
"""
        
class Study:
    def __init__(self, paths_t1: list[str], paths_perf : list[str]) -> None:
        self.paths_t1 = paths_t1
        self.paths_perf = paths_perf
        self._init = False
    
    def _init_data(self):
        if len(self.paths_t1) > 0 and len(self.paths_perf) > 0:
            dcm_t1 = pydicom.dcmread(self.paths_t1[0])
            dcm_perf = pydicom.dcmread(self.paths_perf[0])
            self.date = dcm_perf.StudyDate
            self.descriptions = [dcm_t1.SeriesDescription, dcm_perf.SeriesDescription]
            self.ID = dcm_perf.StudyID
            assert dcm_t1.StudyInstanceUID == dcm_perf.StudyInstanceUID, f'Поля StudyInstanceUID у `{self.paths_t1[0]}` и `{self.paths_perf[0]}` не совпадают'
            self.StudyInstanceUID = dcm_perf.StudyInstanceUID
            self.time = dcm_perf.StudyTime
            self.pixel_spacing = dcm_t1.PixelSpacing
            self._init = True
    
    def add(self, dcm_t1: pydicom.FileDataset = None, dcm_perf: pydicom.FileDataset = None):
        """Добавление данных обследования"""
        if dcm_t1 is not None: self.paths_t1.append(dcm_t1)
        if dcm_perf is not None: self.paths_perf.append(dcm_perf)
        if not self._init: self._init_data()
        
    def _check(self): assert self._init, f'Нет данных обследование `{self.ID}`'
    
    def tensor_t1(self): 
        """Возвращает трёхмерный тензор Т1 изображения"""
        return dicom.read_paths_infer_order(sorted(self.paths_t1))
    
    def tensor_perf(self): 
        """Возвращает трёхмерный тензор перфузионной карты"""
        return dicom.read_paths_infer_order(sorted(self.paths_perf))
        
    def array(self):
        """Возвращает четырёхмерный массив трёхмерного изображения с двумя каналами"""
        t1, perf = np.array(self.tensor_t1(), copy=False), np.array(self.tensor_perf(), copy = False)
        t1_shape = t1.shape
        perf_shape = perf.shape
        scales = ( t1_shape[0] / perf_shape[0],  t1_shape[1] / perf_shape[1],  t1_shape[2] / perf_shape[2])
        perf = scipy.ndimage.zoom(perf,scales, order=0)
        return np.stack([t1, perf], axis=0)
    
    def tensor(self):
        """Возвращает четырёхмерный тензор трёхмерного изображения с двумя каналами, соответствующими Т1 изображению и перфузионной карте"""    
        return torch.as_tensor(self.array()), [i.SliceLocation for i in [pydicom.dcmread(j) for j in self.paths_t1]]
    
    def __str__(self):
        return f"""{date_str(self.date)}, {self.time}: обследование {self.StudyInstanceUID}"""


class StudyTRABIT2019:
    def __init__(self, path: str, age:str = None):
        self.path = path.replace('\\', '/')
        self.age = age
        self._init = True
        self.pixel_spacing = 1
        
    def tensor(self): 
        """Возвращает трёхмерный тензор Т1 изображения"""
        mean, std = ((234.26439405168807,), (431.60501534598217,)) # предварительно рассчитанные параметры нормализации
        loader = v2.Compose([nifti.read, partial(torch.squeeze, dim = 3), partial(torch.unsqueeze, dim = 0), v2.Normalize(mean, std)])
        return loader(self.path)
    
    def __str__(self):
        return f"""обследование {self.path.split('/')[-1][4:-4]}: возраст {self.age}"""
        
class PatientTRABIT2019:
    def __init__(self, path: str, age: str = None):
        self.path = path.replace('\\', '/')
        self.age = age
        self.studies = [StudyTRABIT2019(self.path, self.age)]
    
    def __str__(self):
        return f"""Пациент {self.path.split('/')[-1][4:-4]}
возраст: {self.age}
"""

def dicom_load(path: str) -> list[Patient]:
    """Возвращает список объектов типа Patient, каждый из которых содержит объекты типа Observation"""
    patients: dict[Patient] = {}
    studies: dict[Study] = {}
    
    # Обработка отмены выбора директории
    if path is None: return []
    # Рекурсивное сканирование директории
    if 'trabit' not in path:
        for r, d, f in os.walk(path):
            for file in f:
                # Открытие файлов DICOM
                if file.lower().endswith('.dcm'):
                    dcm = pydicom.dcmread(os.path.join(r, file)) # загрузка файлов
                    
                    # Определение пациента
                    patient = dcm.PatientID
                    if patient not in patients: patients[patient] = Patient(dcm)
                    
                    # Определение обследования
                    studyID = dcm.StudyInstanceUID
                    modality = dcm.SeriesDescription
                    if studyID not in studies: 
                        if modality.lower().startswith('t1'): studies[studyID] = Study([os.path.join(r, file)], [])
                        elif 'perf' in modality.lower(): studies[studyID] = Study([], [os.path.join(r, file)])
                        else: logging.warning(f'Неизвестная модальность: {modality}')
                        patients[patient].studies.append(studies[studyID])
                    else:
                        if modality.lower().startswith('t1'): studies[studyID].add(dcm_t1 = os.path.join(r, file))
                        elif 'perf' in modality.lower(): studies[studyID].add(dcm_perf = os.path.join(r, file))
                        else: logging.warning(f'Неизвестная модальность: {modality}')
                        
                elif file.lower().endswith(('.nii', '.nii.gz')):
                    patient = os.path.join(r, f)
                    if patient not in patients: patients[patient] = PatientTRABIT2019(path, 'Неизвестен')
                    
    
    else:
        p = r'D:\datasets\trabit2019-imaging-biomarkers'
        if 'train' in path or path.lower().endswith('s'):
            with open(f'{p}/train.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == 'scan_id': continue
                    nii_path = f'{p}/train/{row[2]}'
                    age = row[1]
                    patients[nii_path] = PatientTRABIT2019(nii_path, age=age)
        if 'test' in path or path.lower().endswith('s'):
                with open(f'{p}/test_sample_submission.csv', 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[0] == 'scan_id': continue
                        nii_path = f'{path}/test/mri_{row[0].rjust(8, "0")}.nii'
                        patients[nii_path] = PatientTRABIT2019(nii_path, age = 'Неивестен')
                    
    return list(patients.values())