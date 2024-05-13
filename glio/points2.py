"""Ограничительные рамки для нахождения места проведения операции"""
from collections.abc import Iterable
from typing import Optional
import math
import itertools
import numpy as np
from .python_tools import SupportsIter, reduce_dim
def _arr_to_shape(arr, coords):
    if hasattr(arr, "shape"): arr = list(arr.shape)
    if len(arr) > len(coords): arr = arr[len(arr) - len(coords):]
    return np.array(list(arr))

class Point:
    def __init__(self, coords: Iterable[int | float], space: Iterable[int | float] | SupportsIter, rel = False):
        self.space: np.ndarray = _arr_to_shape(space, coords)
        if rel: self.coords_rel = np.array(list(coords))
        else: self.coords_rel = np.array([coord / dim for dim, coord in zip(self.space, coords)])

    @property
    def coords_abs(self):
        return np.array([coord * dim for dim, coord in zip(self.space, self.coords_rel)])

    @property
    def coords_int(self):
        return [int(round(i)) for i in self.coords_abs]

    @property
    def space_int(self):
        return [int(round(i)) for i in self.space]

    @property
    def ndim(self): return len(self.coords_rel)
    def __str__(self) -> str:
        return f"Point({', '.join([str(i) for i in self.coords_abs])}) in space: ({', '.join([str(i) for i in self.space])})"

    def translate(self, new_arr):
        return Point(self.coords_rel, new_arr, rel = True)

    def dim_arrays(self):
        dim_arrays = []
        for coord, length in zip(self.coords_abs, self.space):
            dim_arrays.append(np.zeros(length))
            dim_arrays[-1][int(coord)] = 1
        return dim_arrays

    def __getitem__(self, s:slice | tuple):
        if isinstance(s, slice): s = (s,)
        nspace = []
        ncoords = []
        for sl, length, coord in itertools.zip_longest(s, self.space, self.coords_abs):
            if isinstance(sl, slice):
                start,stop,step = sl.start, sl.stop, sl.step

                if start is None: start = 0
                if stop is None: stop = length
                if step is None: step = 1

                if start < 0: start += length
                if stop < 0: stop += length

                stop = min(stop, length)

                coord = (coord - start) / step
                length = (stop - start) / step

                nspace.append(length)
                ncoords.append(coord)
            elif isinstance(sl, int): pass
            elif sl is None:
                nspace.append(length)
                ncoords.append(coord)
        return Point(coords = ncoords, space = nspace, rel = False)

    def __add__(self, other: "Point | Iterable[int | float] | int | float"):
        if isinstance(other, Point): return Point(coords = self.coords_rel + other.coords_rel, space = self.space, rel = True)
        elif isinstance(other, int | float): return Point(coords = self.coords_rel + other, space = self.space, rel = True)
        else: return Point(coords = [s + o for s,o in zip(self.coords_abs, other)], space = self.space, rel = False)
    def __sub__(self, other: "Point"):
        if isinstance(other, Point): return Point(coords = self.coords_rel - other.coords_rel, space = self.space, rel = True)
        elif isinstance(other, int | float): return Point(coords = self.coords_rel - other, space = self.space, rel = True)
        else: return Point(coords = [s - o for s,o in zip(self.coords_abs, other)], space = self.space, rel = False)
    def __neg__(self): return Point(coords = -self.coords_rel, space = self.space, rel = True)
    def __truediv__(self, other: int | float): return Point(coords = self.coords_rel / other, space = self.space, rel = True)
    def __mul__(self, other: int | float): return Point(coords = self.coords_rel * other, space = self.space, rel = True)
    def __floordiv__(self, other: int | float): return Point(coords = self.coords_rel // other, space = self.space, rel = True)
    def permute(self, *dims:int):
        return Point(coords=[self.coords_rel[dim] for dim in dims], space=[self.space[dim] for dim in dims], rel = True)
    def copy(self): return Point(coords=self.coords_rel.copy(), space=self.space.copy(), rel = True)

    def squeeze(self, dim:int):
        coords:list = self.coords_rel.copy().tolist()
        coords.pop(dim)
        space: list = self.space.copy().tolist()
        space.pop(dim)
        return Point(coords=coords, space=space, rel=True)

    def draw(self, arr, channel=0, opacity = 0.2, size = 5): return draw_point(arr, self, channel, opacity, size)

class Points:
    def __init__(self, *points:Point):
        self.points = list(points)

    @property
    def space(self): return self.points[0].space
    @property
    def space_int(self): return self.points[0].space_int
    @property
    def ndim(self): return self.points[0].ndim
    def __len__(self): return len(self.points)
    def translate(self, new_arr): return self.__class__(*[point.translate(new_arr) for point in self.points])
    def __getitem__(self, s:slice | tuple): return self.__class__(*[point[s] for point in self.points])
    def permute(self, *dims:int): return self.__class__(*[point.permute(*dims) for point in self.points])
    def copy(self): return self.__class__(*[point.copy() for point in self.points])
    def squeeze(self, dim:int): return self.__class__(*[point.squeeze(dim) for point in self.points])

class BoundingBox(Points):
    def __init__(self, x1:Point | Iterable[int | float], x2:Point | Iterable[int | float], space: Optional[Iterable[int | float] | SupportsIter] = None, rel = False):
        if not isinstance(x1, Point):
            if space is None: raise ValueError("Space must be provided if x1 is not a Point")
            x1 = Point(x1, space, rel)
        if not isinstance(x2, Point):
            if space is None: raise ValueError("Space must be provided if x2 is not a Point")
            x2 = Point(x2, space, rel)
        super().__init__(x1, x2)

    @property
    def x1(self): return self.points[0]
    @property
    def x2(self): return self.points[1]
    def area_abs(self): return math.prod((self.x2 - self.x1).coords_abs)
    def area_rel(self): return math.prod((self.x2 - self.x1).coords_rel)

    def xxdd_abs(self): return self.x1.coords_abs + (self.x2 - self.x1).coords_abs
    def xxdd_rel(self): return self.x1.coords_rel + (self.x2 - self.x1).coords_rel
    def xyxy_abs(self): return reduce_dim([[x,y] for x,y, in zip(self.x1.coords_abs, self.x2.coords_abs)])
    def xyxy_rel(self): return reduce_dim([[x,y] for x,y, in zip(self.x1.coords_rel, self.x2.coords_rel)])
    def center(self): return (self.x1 + self.x2) / 2
    @classmethod
    def from_xxdd(cls, xxdd:Iterable[int | float], space: Iterable[int | float] | SupportsIter, rel = False):
        xxdd = list(xxdd)
        x1 = Point(xxdd[:len(xxdd)//2], space, rel)
        x2 = Point(xxdd[len(xxdd)//2:], space, rel)
        return cls(x1, x2, space, rel)
    @classmethod
    def from_xyxy(cls, xyxy:Iterable[int | float], space: Iterable[int | float] | SupportsIter, rel = False):
        xyxy = list(xyxy)
        x1 = Point(xyxy[:2], space, rel)
        x2 = Point(xyxy[2:], space, rel)
        return cls(x1, x2, space, rel)

    def toslice(self):
        return [slice(int(round(i)), int(round(j))) for i, j in zip(self.x1.coords_abs, self.x2.coords_abs)]

    def draw(self, arr, channel = 0, opacity = 0.2): return draw_bbox(arr, self, channel, opacity)

def draw_point(arr, point:Point | Iterable[int | float], channel=0, opacity = 0.2, size = 5):
    import torch
    if not isinstance(point, Point): point = Point(point, arr, False)
    area = [slice(int(round(i-size/2)), int(round(i+size/2))) for i in point.coords_abs]
    max_value = arr.max()
    min_value = arr.min()
    point_value = (max_value + min_value) / (1/opacity)
    if arr.ndim == point.ndim: arr = torch.stack((arr, arr, arr))
    arr[:, *area] -= point_value
    arr[channel, *area] += 2 * point_value
    arr = arr.clip(min_value, max_value)
    return arr

def draw_bbox(arr, bbox:BoundingBox | Iterable[int | float], channel=0, opacity = 0.2):
    import torch
    if not isinstance(bbox, BoundingBox): bbox = BoundingBox.from_xxdd(bbox, arr)
    max_value = torch.max(arr)
    min_value = torch.min(arr)
    bbox_value = (max_value + min_value) / (1/opacity)
    if arr.ndim == bbox.ndim: arr = torch.stack((arr, arr, arr))
    arr[:, *bbox.toslice()] -= bbox_value
    arr[channel, *bbox.toslice()] += 2 * bbox_value
    arr = arr.clip(min_value, max_value)
    return arr
