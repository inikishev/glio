import os, shutil
import logging
from typing import Any

import torch
from monai.inferers import SlidingWindowInfererAdapt # type:ignore

from stuff.found.MRIPreprocessor.mri_preprocessor import Preprocessor

from .DICOMFolder import DICOMDir, Study
from ..loaders import niireadtensor
from ..transforms import z_normalize
from ..train2 import Learner


def preprocess(study:Study, outpath, progress:Any = None):
    logging.info(f"`preprocess`: Предобработка {study} в {outpath}") # pylint:disable=W1203
    # файлы конвертируются в nifti
    progress(0.08, "Корегистрация, пожалуйста подождите...")
    files = study.mods2nifti(outpath)
    logging.info(f"`preprocess`: Созданы nifti файлы {files}") # pylint:disable=W1203
    # создаётся папка для предобработки
    preprocessed = os.path.join(outpath, "preprocessed")
    if not os.path.exists(preprocessed): os.mkdir(preprocessed)
    # предобработка
    preprocessor = Preprocessor(files,
                    output_folder = preprocessed,
                    reference='T1',
                    label=None,
                    prefix='preprocessed_',
                    already_coregistered=False,
                    mni=True,
                    crop=True)
    progress(0.1, "Удаление черепа, пожалуйста подождите...")
    preprocessor.run_pipeline()
    # папка с предобработанными файлами
    skullstripped = os.path.join(preprocessed, "skullstripping")
    # пути к каждому файлу
    t1c = os.path.join(skullstripped, "preprocessed_T1c.nii.gz")
    t1n = os.path.join(skullstripped, "preprocessed_T1.nii.gz")
    t2f = os.path.join(skullstripped, "preprocessed_FLAIR.nii.gz")
    t2w = os.path.join(skullstripped, "preprocessed_T2.nii.gz")
    logging.info(f"`preprocess`: Созданы предобработанные файлы:\n{t1c}\n{t1n}\n{t2f}\n{t2w}") # pylint:disable=W1203
    return torch.stack([z_normalize(niireadtensor(i)) for i in (t1c,t1n,t2f,t2w)]).swapaxes(0,1).to(torch.float32)

def inference(dicom_folder:str, model, overlap = 0.5, progress:Any = None):
    logging.info(f"`inference`: расчёт вывода модели по обследованию из {dicom_folder}") # pylint:disable=W1203
    # Обследования
    d = DICOMDir(dicom_folder)
    logging.info(f"`inference`: просканирована папка: {d}") # pylint:disable=W1203
    studies = d.get_all_studies()
    logging.info(f"`inference`: просканированы обследования: {studies}") # pylint:disable=W1203
    if len(studies) == 0: raise ValueError("Не найдены обследования.")
    if len(studies) > 1: raise ValueError("Найдены более чем одно обследование.")
    study = studies[0]
    # предобработка
    TEMP_FOLDER = '/mnt/f/Stuff/Programming/AI/glio_diff/glio/glioblastoma/GLIOTEMP_PREPROCESSING'
    if os.path.exists(TEMP_FOLDER): shutil.rmtree(TEMP_FOLDER)
    os.mkdir(TEMP_FOLDER)
    progress(0.07, "Корегистрация и удаление черепа, пожалуйста подождите...")
    preprocessed = preprocess(study, TEMP_FOLDER, progress=progress)
    logging.info(f"`inference`: произведена предобработка: тензор {preprocessed.shape}") # pylint:disable=W1203
    inferer = SlidingWindowInfererAdapt(roi_size=(96, 96), sw_batch_size=24, overlap=overlap, mode='gaussian', progress=True)
    # инференс
    progress(0.5, "Сегментация, пожалуйста подождите...")
    with torch.no_grad():
        out = inferer(preprocessed, model).swapaxes(0,1) # type:ignore
    shutil.rmtree(TEMP_FOLDER)
    logging.info(f"`inference`: предсказанная карта: тензор {out.shape}") # pylint:disable=W1203
    return out

def inference_from_checkpoint(dicom_folder: str, checkpoint_folder:str, model, cbs = (), overlap = 0.5, progress:Any = None, **kwargs):
    progress(0.06, "Инициализация модели, пожалуйста подождите...")
    learner = Learner.from_checkpoint(checkpoint_folder,model=model, cbs=cbs, **kwargs)
    learner.fit(0, None, None)
    return inference(dicom_folder, learner.inference, overlap = overlap, progress = progress)

def inference_from_checkpoint_mock(dicom_folder: str, checkpoint_folder:str, model, cbs = (), progress:Any = None, **kwargs):
    learner = Learner.from_checkpoint(checkpoint_folder,model=model, cbs=cbs, **kwargs)
    d = DICOMDir(dicom_folder)
    print(d)
    return torch.randn((4,155,240,250))

def inference_from_checkpoint_mock2(*args, **kwargs):
    return torch.randn((4,155,240,250))

def inference_from_checkpoint_nii(t1c,t1n,t2f,t2w, checkpoint_folder:str, model, cbs = (), overlap = 0.5, **kwargs):
    learner = Learner.from_checkpoint(checkpoint_folder,model=model, cbs=cbs, **kwargs)
    learner.fit(0, None, None)
    logging.info(f"`inference`: расчёт вывода модели по обследованию из {t1c}, {t1n}, {t2f}, {t2w}") # pylint:disable=W1203
    # Обследования
    t1c = z_normalize(niireadtensor(t1c).to(torch.float32))
    t1n = z_normalize(niireadtensor(t1n).to(torch.float32))
    t2f = z_normalize(niireadtensor(t2f).to(torch.float32))
    t2w = z_normalize(niireadtensor(t2w).to(torch.float32))

    preprocessed = torch.stack([t1c, t1n, t2f, t2w])
    inferer = SlidingWindowInfererAdapt(roi_size=(96, 96), sw_batch_size=24, overlap=overlap, mode='gaussian', progress=True)
    # инференс
    with torch.no_grad():
        out = inferer(preprocessed.swapaxes(0,1), model).swapaxes(0,1) # type:ignore
    logging.info(f"`inference`: предсказанная карта: тензор {out.shape}") # pylint:disable=W1203
    return out