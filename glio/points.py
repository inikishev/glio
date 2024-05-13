"""Ограничительные рамки для нахождения места проведения операции"""
from collections.abc import Sequence
from typing import Any
import math
from .python_tools import try_copy, reduce_dim
def _arr_to_shape(arr, coords):
    if arr is None: return None
    if hasattr(arr, "shape"): arr = list(arr.shape)
    if len(arr) > len(coords): arr = arr[len(arr) - len(coords):]
    return list(arr)

class Point:
    def __init__(self, coords:Sequence[int|float], arr:Any = None, rel:bool=False):
        self.coords = list(coords)
        self.arr = _arr_to_shape(arr, self.coords)
        self.rel=rel

    @property
    def abs(self): return not self.rel
    @property
    def ndim(self): return len(self.coords)
    def __len__(self): return len(self.coords)
    def __str__(self) -> str:
        if self.abs: return f"Absolute point({', '.join([str(i) for i in self.coords])})"
        else: return f"Relative point({', '.join([str(i) for i in self.coords])})"

    def to_rel(self):
        if self.rel: return self
        if self.arr is None: raise ValueError
        return Point([coord / size for coord, size in zip(self.coords, self.arr)], arr=self.arr, rel=True)

    def to_abs(self, arr = None):
        if self.abs: return self
        if arr is None: arr = self.arr
        arr = _arr_to_shape(arr, self.coords)
        if arr is None: raise ValueError
        return Point([coord * size for coord, size in zip(self.coords, arr)], arr=arr, rel=False)

    def translate(self, new_arr):
        if self.rel: return self.to_abs(new_arr)
        return self.to_rel().to_abs(new_arr)

    def crop(self, slices:Sequence[slice | Sequence[int | float | None]]): # type:ignore
        point = self.to_abs() if self.rel else self
        slices: list[list[int | float | None]] = [[i.start, i.stop] if isinstance(i, slice) else i for i in slices] # type:ignore
        for i, s in enumerate(slices):
            if s[0] is None: s[0] = 0
            if s[1] is None and point.arr is not None: s[1] = point.arr[i]
            if s[1] is not None and s[1] < 0:
                if point.arr is not None: s[1] = point.arr[i] - s[1]
                else: s[1] = None

        if self.abs:
            return Point([coord-s[0] if s[0] is not None else coord for coord, s in zip(point.coords, slices)],
                    arr=[(min(dim, s[1]) if s[1] is not None else dim) for dim, s in zip(point.arr, slices)] if point.arr is not None else None,
                    rel = False)

        else:
            return Point([coord-s[0] if s[0] is not None else coord for coord, s in zip(point.coords, slices)],
                    arr=[(min(dim, s[1]) if s[1] is not None else dim) for dim, s in zip(point.arr, slices)] if point.arr is not None else None,
                    rel = False).to_rel()

    def toint(self):
        if self.rel: raise ValueError("Невозможно перевести относительные координаты в целочисл. значение.")
        return Point([int(coord) for coord in self.coords], self.arr, rel=False)

    def __add__(self, other:"Point"):
        if self.rel and other.abs: other = other.to_rel()
        if self.abs and other.rel: other = other.to_abs()
        return Point([coord1+coord2 for coord1, coord2 in zip(self.coords, other.coords)], self.arr, self.rel)
    def __sub__(self, other:"Point"):
        if self.rel and other.abs: other = other.to_rel()
        if self.abs and other.rel: other = other.to_abs()
        return Point([coord1-coord2 for coord1, coord2 in zip(self.coords, other.coords)], self.arr, self.rel)
    def __mul__(self, other:int|float):
        return Point([coord*other for coord in self.coords], self.arr, self.rel)
    def __truediv__(self, other:int|float):
        return Point([coord/other for coord in self.coords], self.arr, self.rel)
    def __floordiv__(self, other:int|float):
        return Point([coord//other for coord in self.coords], self.arr, self.rel)
    def __getitem__(self, item): return self.coords[item]

    def draw(self, arr, channel=0, opacity = 0.2, size = 5): return draw_point(arr, self, channel, opacity, size)

    def permute(self, *dims:int): return Point([self.coords[dim] for dim in dims], self.arr, self.rel)
    def copy(self): return Point(try_copy(self.coords), try_copy(self.arr), self.rel)

    def squeeze(self, dim:int):
        coords = self.coords.copy()
        coords.pop(dim)
        if self.arr is not None:
            arr = self.arr.copy()
            arr.pop(dim)
        else: arr = None
        return Point(coords, arr, self.rel)

class BoundingBox:
    def __init__(self, x1:Point | Sequence[int | float], x2:Point | Sequence[int | float], arr:Any = None, rel =False):
        self.x1 = x1 if isinstance(x1, Point) else Point(x1, arr, rel)
        self.x2 = x2 if isinstance(x2, Point) else Point(x2, arr, rel)
        self.arr = _arr_to_shape(arr, self.x1.coords)
        if self.arr is None: self.arr = self.x1.arr
        self.rel=rel

    def __str__(self) -> str: return f"Bounding box: {self.x1}, {self.x2}"
    @property
    def abs(self): return not self.rel
    @property
    def ndim(self): return self.x1.ndim

    def to_rel(self):
        if self.rel: return self
        return BoundingBox(self.x1.to_rel(), self.x2.to_rel(), self.arr, rel=True)

    def to_abs(self, arr = None):
        if self.abs: return self
        if arr is None: arr = self.arr
        if arr is not None: arr = _arr_to_shape(arr, self.x1.coords)
        return BoundingBox(self.x1.to_abs(arr), self.x2.to_abs(arr), arr, rel=False)

    def translate(self, new_arr):
        new_arr = _arr_to_shape(new_arr, self.x1.coords)
        return BoundingBox(self.x1.translate(new_arr), self.x2.translate(new_arr), new_arr, rel=self.rel)

    def crop(self, slices:Sequence[slice | Sequence[int | float]]):
        x1 = self.x1.crop(slices)
        x2 = self.x2.crop(slices)
        return BoundingBox(x1, x2, x1.arr, rel=self.rel)

    def toint(self): return BoundingBox(self.x1.toint(), self.x2.toint(), self.arr, rel=self.rel)

    def toslice(self):
        point = self.to_abs() if self.rel else self
        return [slice(int(round(i)), int(round(j))) for i, j in zip(point.x1.coords, point.x2.coords)]

    def center(self): return (self.x1 + self.x2) / 2

    def draw(self, arr, channel = 0, opacity = 0.2): return draw_bbox(arr, self, channel, opacity)

    def permute(self, *dims:int):
        return BoundingBox(self.x1.permute(*dims), self.x2.permute(*dims), self.arr, self.rel)
    def copy(self): return BoundingBox(self.x1.copy(), self.x2.copy(), try_copy(self.arr), self.rel)

    def slice(self, start:int, end:int, axis:int):
        x1 = self.x1.copy()
        x2 = self.x2.copy()
        x1.coords[axis] = min(x1.coords[axis] - start, 0)
        x2.coords[axis] = min(x2.coords[axis], end)
        return BoundingBox(x1, x2, self.arr, self.rel)

    def area(self): return math.prod((self.x2 - self.x1).coords)

    def isempty(self): return any([i >= j for i, j in zip(self.x1.coords, self.x2.coords)])

    def squeeze(self, dim:int):
        if self.arr is not None:
            arr = self.arr.copy()
            arr.pop(dim)
        else: arr = None
        return BoundingBox(self.x1.squeeze(dim), self.x2.squeeze(dim), arr, self.rel)

    def xxdd(self): return self.x1.coords + (self.x2 - self.x1).coords
    def xyxy(self): return reduce_dim([[x,y] for x,y, in zip(self.x1.coords, self.x2.coords)])

def bbox_from_xxdd(xxdd:Sequence[int | float], arr:Any = None, rel = False):
    x1 = Point(xxdd[:len(xxdd)//2], arr, rel)
    x2 = Point(xxdd[len(xxdd)//2:], arr, rel)
    return BoundingBox(x1, x1+x2, arr, rel)

def bbox_from_xyxy(xyxy:Sequence[int | float], arr:Any = None, rel = False):
    x1 = Point(xyxy[::2], arr, rel)
    x2 = Point(xyxy[1::2], arr, rel)
    return BoundingBox(x1, x2, arr, rel)

def draw_point(arr, point:Point | Sequence[int | float], channel=0, opacity = 0.2, size = 5):
    import torch
    if not isinstance(point, Point): point = Point(point, arr)
    else: point = point.to_abs(arr)
    area = [slice(int(round(i-size/2)), int(round(i+size/2))) for i in point.coords]
    max_value = arr.max()
    min_value = arr.min()
    point_value = (max_value + min_value) / (1/opacity)
    if arr.ndim == point.ndim: arr = torch.stack((arr, arr, arr))
    arr[:, *area] -= point_value
    arr[channel, *area] += 2 * point_value
    arr = arr.clip(min_value, max_value)
    return arr

def draw_bbox(arr, bbox:BoundingBox | Sequence[int | float], channel=0, opacity = 0.2):
    import torch
    if not isinstance(bbox, BoundingBox): bbox = bbox_from_xxdd(bbox, arr)
    else: bbox = bbox.to_abs(arr)
    max_value = torch.max(arr)
    min_value = torch.min(arr)
    bbox_value = (max_value + min_value) / (1/opacity)
    if arr.ndim == bbox.ndim: arr = torch.stack((arr, arr, arr))
    arr[:, *bbox.toslice()] -= bbox_value
    arr[channel, *bbox.toslice()] += 2 * bbox_value
    arr = arr.clip(min_value, max_value)
    return arr

