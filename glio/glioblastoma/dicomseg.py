# Автор - Никишев Иван Олегович
from typing import Any
import os, tarfile
import pydicom
import pydicom_seg
import numpy as np
import SimpleITK as sitk
from monai import transforms
from monai.inferers import SlidingWindowInfererAdapt # type:ignore
import torch
from .inference import inference_from_checkpoint, inference_from_checkpoint_mock
from .seg_json import seg_json
from .DICOMFolder import DICOMDir
from ..python_tools import get_all_files, find_file_containing
from ..loaders import niireadtensor, niiread_affine, niiwrite
from ..transforms import z_normalize
from ..train2 import Learner
from .exceptions import GradioException

def get_instance_uid(study_dir):
    dfiles = get_all_files(study_dir,recursive=False)
    return pydicom.dcmread(dfiles[0]).StudyInstanceUID

def find_file_containing_l(files, contains:str):
    for f in files:
        print(f, contains)
        if contains in f:
            return f
    return None

def save_seg(dcmpath, outpath, checkpoint_path, model, overlap=0.5, progress:Any = None):
    if '.nii.gz' in dcmpath[0]:
        t1c = niireadtensor(find_file_containing_l(dcmpath, "t1c")).to(torch.float32) # type:ignore
        t1n = niireadtensor(find_file_containing_l(dcmpath, "t1n")).to(torch.float32) # type:ignore
        t2f = niireadtensor(find_file_containing_l(dcmpath, "t2f")).to(torch.float32) # type:ignore
        t2w = niireadtensor(find_file_containing_l(dcmpath, "t2w")).to(torch.float32) # type:ignore
        affine = niiread_affine(find_file_containing_l(dcmpath, "t1c")) # type:ignore
        images = torch.stack((z_normalize(t1c),z_normalize(t1n),z_normalize(t2f),z_normalize(t2w))).permute(3,0,2,1)
        l = Learner.from_checkpoint(checkpoint_path, model, ())
        l.fit(0, None, None)
        inferer = SlidingWindowInfererAdapt(roi_size=(96, 96), sw_batch_size=32, overlap=overlap, progress=True)
        output = inferer(images, l.inference).argmax(1).permute(2,1,0).to(torch.float32) # type:ignore
        niipath = os.path.join(outpath, "segmentation.nii.gz")
        niiwrite(niipath, output, affine)
        return niipath

    else:
        # Шаблонный JSON
        template = pydicom_seg.template.from_dcmqi_metainfo(seg_json())

        # Объект для записи в DICOM
        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                    # x and y axes
            skip_empty_slices=False,  # Don't encode slices with only zeros
            skip_missing_segment=True,  # If a segment definition is missing in the
                                        # template, then raise an error instead of
                                        # skipping it.
        )

        # чтение DICOM
        reader = sitk.ImageSeriesReader()
        # dcm_files = reader.GetGDCMSeriesFileNames(study_dir, get_instance_uid(study_dir))
        # получение списка файлов обследования
        progress(0.03, "Распределение файлов по модальносям, пожалуйста подождите...")
        d = DICOMDir(dcmpath)
        studies = d.get_all_studies()
        if len(studies) == 0: raise GradioException("Не найдены обследования в указанной директории.")
        if len(studies) > 1: raise GradioException(f"В указанной директории найдены {len(studies)} обследований: {[i.id for i in studies]}")
        study = studies[0]
        series = study.get_mods()["T1c"]
        dcm_files = list(series.files.keys())

        # установление списка файлов к объекту для чтения
        reader.SetFileNames(dcm_files)
        progress(0.05, "Чтение пиксельной информации, пожалуйста подождите...")
        image = reader.Execute()
        size = image.GetSize()
        print(f"Размер исходного изображения: {size}")

        # расчёт вывода
        segmentation_data = inference_from_checkpoint(dcmpath, checkpoint_path, model, (), overlap=overlap, progress = progress)
        print(f"Размер изображения сегментации: {segmentation_data.shape}")

        outpaths = []
        # обратное изменение размера
        segmentation_data = transforms.Resize(size)(segmentation_data) # type:ignore
        print(f"Размер изображения после изменения размера: {segmentation_data.shape}")
        segmentation_numpy = (segmentation_data.argmax(0)).to(torch.uint8).permute(2,0,1).numpy()
        print(f'{segmentation_numpy.min() = }, {segmentation_numpy.max() = }, {segmentation_numpy.shape = }')
        print('Экспорт сегментации со всеми классами')
        np.save(os.path.join(outpath, "segmentation_all.npy"), segmentation_numpy)
        outpaths.append(os.path.join(outpath, "segmentation_all.npy"))
        try:
            segmentation = sitk.GetImageFromArray(segmentation_numpy)
            print('Копирование информации')
            segmentation.CopyInformation(image)
            print('Запись аттрибутов')
            dcm = writer.write(segmentation, dcm_files) # type:ignore
            channel_outpath = os.path.join(outpath, "segmentation_all.dcm")
            print(f'Путь {channel_outpath}, сохранение...')
            dcm.save_as(channel_outpath)
            outpaths.append(channel_outpath)
            print("Сохранено.")
        except Exception as e:
            print('НЕ УДАЛОСЬ СОХРАНИТЬ ВСЕ ПИКСЕЛИ')
            print(e)
        for i,seg_channel in enumerate(segmentation_data):
            try:
                print(f'Сохранение канала {i}')
                if i == 0: continue
                # Создание sitk изображения сегментации
                print("Создание sitk изображения сегментации")
                segmentation = sitk.GetImageFromArray(seg_channel.to(torch.uint8).permute(2,0,1).numpy())
                # Копирование информации из исходного изображения
                print("Копирование информации из исходного изображения")
                segmentation.CopyInformation(image)

                # Запись изображения в DICOM
                print("Запись изображения в DICOM объект")
                dcm = writer.write(segmentation, dcm_files) # type:ignore
                label = "edema" if i == 1 else "necrosis" if i == 2 else "enchancing_tumor"
                channel_outpath = os.path.join(outpath, f"segmentation_{i}_{label}.dcm")
                print(f"Запись {label} в {channel_outpath}")
                dcm.save_as(channel_outpath)
                outpaths.append(channel_outpath)
                print("Сохранено.")
            except Exception as e:
                print(f'НЕ УДАЛОСЬ СОХРАНИТЬ {i}')
                print(e)
        with tarfile.open(os.path.join(outpath, "сегментация пациента .tar"), "w:gz") as tar:
            for p in outpaths:
                tar.add(p, arcname = os.path.basename(p))

        return os.path.join(outpath, "сегментация пациента .tar")

def save_seg_mock(dcmpath, outpath, checkpoint_path, model, overlap=0.5, progress: Any = None):

    # Шаблонный JSON
    template = pydicom_seg.template.from_dcmqi_metainfo(seg_json())

    # Объект для записи в DICOM
    writer = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                # x and y axes
        skip_empty_slices=False,  # Don't encode slices with only zeros
        skip_missing_segment=True,  # If a segment definition is missing in the
                                    # template, then raise an error instead of
                                    # skipping it.
    )

    # чтение DICOM
    reader = sitk.ImageSeriesReader()
    # dcm_files = reader.GetGDCMSeriesFileNames(study_dir, get_instance_uid(study_dir))
    # получение списка файлов обследования
    progress(0.025, "Распределение файлов по модальносям, пожалуйста подождите...")
    d = DICOMDir(dcmpath)
    studies = d.get_all_studies()
    if len(studies) == 0: raise ValueError("Не найдены обследования.")
    if len(studies) > 1: raise ValueError("Найдены более чем одно обследование.")
    study = studies[0]
    series = study.get_mods()["T1c"]
    dcm_files = list(series.files.keys())

    # установление списка файлов к объекту для чтения
    reader.SetFileNames(dcm_files)
    progress(0.03, "Чтение пиксельной информации, пожалуйста подождите...")
    image = reader.Execute()
    size = image.GetSize()
    print(f"Размер исходного изображения: {size}")

    # расчёт вывода
    progress(0.05, "Предобработка, пожалуйста подождите...")
    segmentation_data = inference_from_checkpoint_mock(dcmpath, checkpoint_path, model, (), overlap=overlap, progress=progress)
    print(f"Размер изображения сегментации: {segmentation_data.shape}")

    # обратное изменение размера
    progress(0.8, "Пост-обработка, пожалуйста подождите...")
    segmentation_data = transforms.Resize(size)(segmentation_data) # type:ignore
    print(f"Размер изображения после изменения размера: {segmentation_data.shape}")
    progress(0.9, "Запись DICOM файлов, пожалуйста подождите...")
    for i,seg_channel in enumerate(segmentation_data):
        if i == 0: continue
        # Создание sitk изображения сегментации
        segmentation = sitk.GetImageFromArray(seg_channel.to(torch.uint8).permute(2,0,1).numpy())
        # Копирование информации из исходного изображения
        segmentation.CopyInformation(image)

        # Запись изображения в DICOM
        dcm = writer.write(segmentation, dcm_files) # type:ignore
        label = "edema" if i == 1 else "necrosis" if i == 2 else "enchancing_tumor"
        channel_outpath = os.path.join(outpath, f"segmentation_{i}_{label}.dcm")
        dcm.save_as(channel_outpath)

    progress(0.99, "Архивирование DICOM файлов, пожалуйста подождите...")
    with tarfile.TarFile(os.path.join(outpath, "segmentation.tar"), "w") as tar:
        tar.add(outpath)

    return os.path.join(outpath, "segmentation.tar")
