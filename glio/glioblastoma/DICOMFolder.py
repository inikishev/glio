import os,shutil
import numpy as np
import torch
import dicom2nifti
import dicom2nifti.settings as settings
import pydicom, pydicom.errors
from .exceptions import GradioException
from ..python_tools import flexible_filter, listdir_fullpaths, reduce_dim, get_all_files

T1C_FILT = "T1", "+C"
T1N_FILT = "T1", lambda x: not (x.endswith("C") or x.endswith("CM") or "SENSE" in x)
T2F_FILT = "Flair"
T2W_FILT = "AX","T2"

def dicomfiles2nifti(files, outpath, compression = True, reorient = True):
    # настройки
    settings.disable_validate_orthogonal()
    settings.disable_validate_slice_increment()
    settings.enable_resampling()
    settings.set_resample_spline_interpolation_order(1)
    settings.set_resample_padding(-1000)
    # создание временной папки
    TEMP_FOLDER = os.path.join(outpath, "glio_TEMP")
    if os.path.exists(TEMP_FOLDER): shutil.rmtree(TEMP_FOLDER)
    os.mkdir(TEMP_FOLDER)
    # создание папки DICOM
    for file in files:
        shutil.copy(file, os.path.join(TEMP_FOLDER, os.path.basename(file)))
    # конвертация
    dicom2nifti.convert_directory(TEMP_FOLDER, outpath, compression=compression, reorient=reorient)
    shutil.rmtree(TEMP_FOLDER)

def try_read_dicom(path):
    try: return pydicom.read_file(path)
    except pydicom.errors.InvalidDicomError: return None

class Series:
    def __init__(self, files:dict[str, pydicom.Dataset], seriesid):
        # идентификатор
        self.id = str(seriesid)
        # сортировка файлов по InstanceNumber
        files_sorted = sorted(list(files.items()), key = lambda x: x[1].InstanceNumber)
        self.files: dict[str, pydicom.Dataset] = {k:v for k,v in files_sorted}
    def toarray(self):
        return np.stack([i.pixel_array for i in self.files.values()])
    def __repr__(self):
        return f"Серия {self.id} из {len(self.files)} изображений."
    def __iter__(self): return iter(self.files)
    def __len__(self): return len(self.files)
    def __getitem__(self,item):
        if isinstance(item, int): return list(self.files.values())[item]
        return self.files[item]
    def write_nifti(self, outpath, compression = True, reorient = True):
        dicomfiles2nifti(self.files.keys(), outpath, compression = compression, reorient = reorient)

    def get_slice_locations(self):
        return [i.SliceLocation for i in [pydicom.dcmread(j) for j in self.files.keys()]]


class Study:
    def __init__(self, files:dict[str, pydicom.Dataset], studyid, seriesid_attr: str):
        # аттрибуты
        self.files = files
        self.seriesid_attr = seriesid_attr
        # идентификатор
        self.id = str(studyid)
        # распределение по сериям
        # набор уникальных идентификаторов
        self.series_ids = set([getattr(file, seriesid_attr) for file in files.values()])
        # заполнение файлов по их идентификаторам
        series = {i:{} for i in self.series_ids}
        for path, file in self.files.items():
            series[getattr(file, seriesid_attr)][path] = file
        # создание обследований
        self.series = [Series(files=series, seriesid=seriesid) for seriesid, series in series.items()]
        self.series.sort(key = lambda x: x.id)

    def filter_series(self, *args):
        ids = [s.id for s in self.series]
        matches = flexible_filter(ids, args)
        return [s for s in self.series if s.id in matches]

    def __repr__(self):
        return f"Обследование {self.id} с {len(self.series)} сериями."
    def __iter__(self): return iter(self.series)
    def __len__(self): return len(self.series)
    def __getitem__(self,item):
        if isinstance(item, int): return self.series[item]
        for p in self.series:
            if p.id == item: return p

    def uitext(self):
        return str(self)

    def get_mods(self) -> dict[str, Series]:
        t1c = self.filter_series(*T1C_FILT)
        t1n = self.filter_series(*T1N_FILT)
        t2f = self.filter_series(*T2F_FILT)
        t2w = self.filter_series(*T2W_FILT)
        if len(t1c) == 0: raise GradioException(f"Не найдено обследование методом T1 с контрастом, доступны {[i.id for i in self.series]}")
        if len(t1n) == 0: raise GradioException(f"Не найдено обследование методом T1 без контраста, доступны {[i.id for i in self.series]}")
        if len(t2f) == 0: raise GradioException(f"Не найдено обследование методом FLAIR, доступны {[i.id for i in self.series]}")
        if len(t2w) == 0: raise GradioException(f"Не найдено обследование методом T2W, доступны {[i.id for i in self.series]}")

        return dict(T1c = t1c[0], T1 = t1n[0], FLAIR = t2f[0], T2 = t2w[0])

    def get_mods_tensor(self):
        return torch.stack([torch.from_numpy(i.toarray()) for i in self.get_mods().values()])

    def get_slice_loc(self):
        return self.get_mods()["T1C"].get_slice_locations()

    def get_pixel_spacing(self):
        return list(self.get_mods()["T1C"].files.values())[0].PixelSpacing

    def mods2nifti(self, outpath, compression = True, reorient = True):
        mods = self.get_mods()
        files = {}
        for mod, series in mods.items():
            if not os.path.exists(os.path.join(outpath, mod)): os.mkdir(os.path.join(outpath, mod))
            series.write_nifti(os.path.join(outpath, mod), compression = compression, reorient = reorient)
            files[mod] = listdir_fullpaths(os.path.join(outpath, mod))[-1]
        return files

class Patient:
    def __init__(self, files:dict[str, pydicom.Dataset], patientid, studyid_attr, seriesid_attr):
        # аттрибуты
        self.files = files
        self.studyid_attr = studyid_attr
        self.seriesid_attr = seriesid_attr
        # идентификатор
        self.id = str(patientid)
        # распределение по обследованиям
        # набор уникальных идентификаторов
        self.study_ids = set([getattr(file, studyid_attr) for file in files.values()])
        # заполнение файлов по их идентификаторам
        studies = {i:{} for i in self.study_ids}
        for path, file in self.files.items():
            studies[getattr(file, studyid_attr)][path] = file
        # создание обследований
        self.studies = [Study(study, studyid, seriesid_attr) for studyid, study in studies.items()]
        self.studies.sort(key = lambda x: x.id)

    def get_all_series(self) -> list[Series]:
        return reduce_dim([i.series for i in self.studies])

    def __repr__(self):
        return f"Пациент {self.id} с {len(self.studies)} обследованиями и {len(self.get_all_series())} сериями."
    def __iter__(self): return iter(self.studies)
    def __len__(self): return len(self.studies)
    def __getitem__(self,item):
        if isinstance(item, int): return self.studies[item]
        for p in self.studies:
            if p.id == item: return p

class DICOMDir:
    def __init__(self, path, recursive=True,
                 patientid_attr = "PatientName",
                 studyid_attr = "AccessionNumber",
                 seriesid_attr = "SeriesDescription"):
        if not (isinstance(path, list)) and os.path.isdir(path): raise GradioException(f"Указанная директория не существует или является файлом: {path}")
        self.path = path
        self.patientid_attr = patientid_attr
        self.studyid_attr = studyid_attr
        self.seriesid_attr = seriesid_attr
        # список файлов
        if not isinstance(path, list):
            if recursive: files = get_all_files(path)
            else: files = listdir_fullpaths(path)
        else: files = path
        # загрузка dicom
        dicom_files = {i:pydicom.read_file(i) for i in files}
        dicom_files = {k:v for k,v in dicom_files.items() if v is not None}
        # присвоение
        self.files: dict[str, pydicom.Dataset] = dicom_files # type:ignore
        # распределение по пациентам
        # набор уникальных идентификаторов
        self.patient_ids = set([getattr(file, patientid_attr) for file in self.files.values()])
        # заполнение файлов по их идентификаторам
        patients = {i:{} for i in self.patient_ids}
        for path, file in self.files.items():
            patients[getattr(file, patientid_attr)][path] = file
        # создание пациентов, обследований, серий
        self.patients = [Patient(patient, patientid, studyid_attr, seriesid_attr) for patientid,patient in patients.items()]
        self.patients.sort(key = lambda x: x.id)

    def get_all_studies(self) -> list[Study]:
        return reduce_dim([i.studies for i in self.patients])
    def get_all_series(self) -> list[Series]:
        return reduce_dim([i.series for i in self.get_all_studies()])
    def __repr__(self):
        return f"DICOMDir с {len(self.patients)} пациентами, {len(self.get_all_studies())} обследованиями, {len(self.get_all_series())} сериями."
    def __iter__(self): return iter(self.patients)
    def __len__(self): return len(self.patients)
    def __getitem__(self,item):
        if isinstance(item, int): return self.patients[item]
        for p in self.patients:
            if p.id == item: return p