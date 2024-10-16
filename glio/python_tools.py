"""python stuff"""
import copy
import functools
import inspect
import operator
import os
import pathlib
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from time import perf_counter
from typing import Any, Optional, Protocol, overload


class HasGetItem(Protocol):
    def __getitem__(self, __key:Any) -> Any: ...

class HasGetItemAndLen(Protocol):
    def __getitem__(self, __key:Any) -> Any: ...
    def __len__(self) -> int: ...

class HasGetItemInt(Protocol):
    def __getitem__(self, __key:int) -> Any: ...

SupportsIter = Iterable | HasGetItemInt
"""Actually working type hint for anything that is iterable."""

# def isiterable(x:Any) -> bool:
#     """In some specific case this is better than `isinstance(x, Iterable)` but I don't remember why.

#     Args:
#         x (Any): _description_

#     Returns:
#         bool: _description_
#     """
#     if hasattr(x, '__iter__'): return not isinstance(x, type)
#     if hasattr(x, '__getitem__'):
#         try:
#             x[0] # pylint:disable=W0104
#             return True
#         except Exception: # pylint:disable=W0718
#             return False
#     return False

def hasgetitem(x:Any) -> bool: return hasattr(x, '__getitem__')

def ismapping(x:Any) -> bool:
    """In some specific case this is better than `isinstance(x, Mapping)` but I don't remember why.

    Args:
        x (Any): _description_

    Returns:
        bool: _description_
    """
    return hasattr(x,"keys") and hasattr(x, "values") and hasattr(x, "items") and hasattr(x, "__getitem__")

def key_value(iterable: Iterable):
    if ismapping(iterable): return iterable.items() # type:ignore
    return enumerate(iterable) #type:ignore

def flatten_generator(iterable:Iterable):
    if isinstance(iterable, Iterable):
        for i in key_value(iterable):
            if isinstance(i[1], Iterable):
                yield from flatten_generator(i[1])
            else: yield i[1]
    else: yield iterable

def flatten(iterable: Iterable) -> list[Any]:
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in flatten(i)]
    else:
        return [iterable]

def flatten_by_tree_generator(iterable: HasGetItem, tree):
    if isinstance(tree, Iterable):
        for k,v in key_value(tree):
            if hasgetitem(v):
                yield from flatten_by_tree_generator(iterable[k], v)
            else: yield iterable[k]
    else: yield iterable

def flatten_by_tree(iterable: HasGetItem, tree):
    return list(flatten_by_tree_generator(iterable, tree))


def apply_tree(x: Any, tree, cached: Any = None, flatten: bool = False): #pylint: disable=W0621
    """_summary_
    ```py
    tree = [func1, func2, [func3, func4], [func5]]
    ```

    If you apply the tree, you get:

    ```py
    [func1(x), func2(func1(x)), [func3(func2(func1(x))), func4(func3(func2(func1(x))))), [func5(func2(func1(x))))], [func5(func2(func1(x)) ))]]
    ```
    Args:
        x (Any): inputs
        tree (_type_): function tree
        cached (Any, optional): if specified, instead of passing `x` to the 1st function in the tree,
        `cached` will be used as its output. Defaults to None.
        flatten (bool, optional): Flattens the output. Defaults to False.

    Returns:
        _type_: _description_
    """
    if ismapping(tree):
        y = {}
        is_dict = True
    else:
        y = []
        is_dict = False

    for k, func in key_value(tree):

        # callable
        if callable(func):
            # only 1 elem can be cached
            if cached is not None:
                x = cached
                cached = None
            else: x = func(x)
            if is_dict: y[k] = x
            else: y.append(x)#pyright:ignore[reportAttributeAccessIssue]
        # not callable
        elif not func:
            if is_dict: y[k] = x
            else: y.append(x)#pyright:ignore[reportAttributeAccessIssue]
        # sequence, recurse
        else:
            if is_dict: y[k] = {}
            else: y.append([])#pyright:ignore[reportAttributeAccessIssue]
            y[k] = apply_tree(x, func, cached)
    return y if not flatten else flatten_by_tree(y, tree)


def apply_recursive_(iterable: Iterable, func: Callable):
    """Applies func to all elements of iterable recursively and in-place.

    Args:
        iterable (SupportsIter): _description_
        func (Callable): _description_
    """
    if isinstance(iterable, Iterable):
        for k,v in key_value(iterable):
            if isinstance(v, Iterable): apply_recursive_(v, func)
            else: iterable[k] = func(v) # type:ignore
    else: iterable = func(iterable)

def get_first_recursive(iterable: SupportsIter) -> Any:
    """Returns first element of a recursive iterable.

    Args:
        iterable (SupportsIter): _description_

    Returns:
        Any: _description_
    """
    try:
        for i in iterable: return get_first_recursive(i)
    except Exception: # pylint: disable=broad-except
        if ismapping(iterable): return get_first_recursive(iterable[list(iterable.keys())[0]]) # type:ignore
    return iterable

def perf_counter_deco(func: Callable, n: int = 1):
    def inner(*args, **kwargs):
        time_start = perf_counter()
        for _ in range(n):
            func(*args, **kwargs)
        print(f"{func.__name__} {n} iterations took {perf_counter() - time_start} perf-counter seconds")
    return inner

@contextmanager
def perf_counter_context(name: Optional[str | Any] = None, ndigits: Optional[int] = None):
    time_start = perf_counter()
    yield
    time_took = perf_counter() - time_start
    if name is None: name = "Context"
    if ndigits is not None: time_took = round(time_took, ndigits)
    print(f"{name} took {time_took} perf_counter seconds")


class PerfCounter:
    def __init__(self):
        self.times= []

    def step(self):
        self.times.append(perf_counter())

def flexible_filter(keys:Iterable[str], filters:Iterable[Callable | str | list[str] | tuple]) -> list[str]:
    """filters is a sequence of filters. Returns keys that match all filters.

    If a filter is a string, returns all keys that have it as a substring.

    If filter is a list/tuple, returns all keys that have at least one of the substrings in the list.

    If filter is a function, each key is passed to it and it must return true or false.

    Args:
        keys (_type_): _description_
        filters (_type_): _description_

    Returns:
        _type_: _description_
    """
    for f in filters:
        if callable(f): keys = [k for k in keys if f(k)]
        elif isinstance(f, (int, float, str)): keys = [k for k in keys if str(f).lower() in str(k).lower()]
        elif isinstance(f, (list, tuple)): keys = [k for k in keys if any([str(i).lower() in str(k).lower() for i in f])]
    return list(keys)

class ExhaustingIterator:
    """
    Iterates an objects but stops the iteration as if the object had a different length.
    """
    def __init__(self, obj:HasGetItemAndLen, length: int, shuffle:bool = True):
        """
        Object must have `__getitem___` and `__len__`.
        """
        self.obj = obj
        self.length = length
        self.cur = -1 # +1 on __next__ start
        self.iterations = 0
        self.indexes = list(range(len(self.obj)))
        self.shuffle = shuffle
        #if self.shuffle: random.shuffle(self.indexes) will be done in next . __next__ when cur = 0
        self.n_samples = len(self.indexes)
        self.not_doing_0th_element = False

    def __iter__(self):
        return self

    def __next__(self):
        self.cur+=1
        if self.not_doing_0th_element and self.cur % self.length == 0:
            self.not_doing_0th_element = False
            self.cur -= 1 # if len is 2 it takes 0, 1, stops at 2, cur -= 1 = 1? in next iter self.cur+=1 = 2
            raise StopIteration()
        self.not_doing_0th_element = True
        element = self.cur % self.n_samples # el, # with len 3 will be 0, 3, 6
        if element == 0:
            if self.shuffle: random.shuffle(self.indexes)
        return self.obj[self.indexes[element]]

    def __len__(self): return self.length


class ItemUnderIndex:
    """Basically itemgetter because I am dumb."""

    def __init__(self, obj:HasGetItem, index: int):
        self.obj = obj
        self.index = index

    def __call__(self):
        return self.obj[self.index]

class ItemUnder2Indexes:
    """Basically itemgetter because I am dumb."""
    def __init__(self, obj:HasGetItem, index1: int, index2:int):
        self.obj = obj
        self.index1 = index1
        self.index2 = index2

    def __call__(self):
        return self.obj[self.index1][self.index2]



def reduce_dim[T](a:Iterable[Iterable[T]]) -> list[T]: # pylint:disable=E0602
    return functools.reduce(operator.iconcat, a, [])


def subclasses_recursive(cls:type | Any) -> set[type]:
    """Recursively get a set of all subclasses of a class (can pass a type or an object of a type)."""
    if not isinstance(cls, type): cls = type(cls)
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in subclasses_recursive(c)])

def type_str(obj:object | type) -> str:
    """Returns class name that includes the namespace, e.g., `torch.nn.Conv2d`"""
    if not isinstance(obj, type): obj = type(obj)
    return repr(obj).replace("<class '", "").replace("'>", "")

def int_at_beginning(s:str) -> int | None:
    """If a string starts with an integer of any length, returns that integer. Otherwise returns None.

    >>> int_at_beginning('123abc')
    123

    >>> int_at_beginning('abc')
    None
    """
    i = 1
    num = None
    while True:
        try:
            num = int(s[:i])
            i+=1
        except ValueError:
            return num

def int_at_beginning_or_str(s:str) -> int | str:
    """If a string starts with an integer of any length, returns that integer. Otherwise returns the string."""
    res = int_at_beginning(s)
    if res is None: return s
    return res

def dict_to_table(data:Mapping, key, order = None, sort_key: Optional[Callable] = None) -> str:
    """Returns a table from a dictionary."""
    if sort_key is not None: data = {k: v for k, v in sorted(data.items(), key=sort_key)}
    keys = []
    if order is not None: keys = order
    if key not in keys: keys.append(key)
    for k in data:
        for k1 in sorted(list(data[k].keys()), key = int_at_beginning_or_str):
            if k1 not in keys: keys.append(k1)
    keys = {k: i for i, k in enumerate(keys)}

    table_list = [[[None] for i in range(len(keys))] for j in range(len(data))]
    i = 0
    for k, v in data.items():
        table_list[i][keys[key]] = k
        for k1, v1 in v.items():
            table_list[i][keys[k1]] = v1
        i += 1

    table = '|'+'|'.join(list(keys.keys()))+'|\n'
    table += '|'+'|'.join(['---']*len(keys))+'|\n'
    for row in table_list:
        table += '|'+'|'.join([str(x) if x is not None else '' for x in row])+'|\n'
    return table



def get_all_files(path:str, recursive:bool = True, extensions: Optional[str | Sequence[str]] = None, path_filter:Optional[Callable] = None) -> list[str]:
    """Get all files recursively.

    :param path: Path to search in.
    :param recursive: defaults to True
    :param extensions: Extensions. defaults to None
    :param path_filter: _description_, defaults to None
    :return: _description_
    """
    all_files = []
    if isinstance(extensions, str): extensions = [extensions]
    if extensions is not None: extensions = tuple([i.lower() for i in extensions])
    if recursive:
        for root, _, files in (os.walk(path)):
            for file in files:
                file_path = os.path.join(root, file)
                if path_filter is not None and not path_filter(file_path): continue
                if extensions is not None and not file.lower().endswith(extensions): continue
                if os.path.isfile(file_path): all_files.append(file_path)
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if path_filter is not None and not path_filter(file_path): continue
            if extensions is not None and not file.lower().endswith(extensions): continue
            if os.path.isfile(file_path): all_files.append(file_path)

    return all_files

def find_file_containing(folder, contains:str, recursive = True, error = True) -> str:
    for f in get_all_files(folder, recursive=recursive):
        if contains in f:
            return f
    if error: raise FileNotFoundError(f"File containing {contains} not found in {folder}")
    return None # type:ignore

def get0[T](x:Sequence[T]) -> T: return x[0] # pylint:disable = E0602
def get1[T](x:Sequence[T]) -> T: return x[1] # pylint:disable = E0602
def getlast[T](x:Sequence[T]) -> T: return x[-1] # pylint:disable = E0602

def identity[T](x:T) -> T: return x # pylint:disable = E0602
def identity_kwargs[T](x:T, *args, **kwargs) -> T: return x # pylint:disable = E0602
def identity_if_none(func:Callable | None) -> Callable:
    if func is None: return identity
    return func
def identity_kwargs_if_none(func:Callable | None) -> Callable:
    if func is None: return identity_kwargs
    return func

def ensure_list(x: Any) -> list:
    if isinstance(x, list): return x
    if isinstance(x, Iterable): return list(x)
    return [x]

def pretty_print_dict(d: Mapping) -> None:
    """returns recursive dicts using json lib."""
    import json
    print(json.dumps(dict(d), indent=4, sort_keys=False))

class Wrapper:
    """Wraps some object. Basically can let you store attributes in built-in types."""
    def __init__(self, obj):
        self.obj = obj
    def __getattr__(self, name):
        return getattr(self.obj, name)
    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)
    def __str__(self):
        return str(self.obj)
    def __repr__(self):
        return repr(self.obj)
    def __eq__(self, other):
        return self.obj == other
    def __hash__(self):
        return hash(self.obj)
    def __getitem__(self, item):
        return self.obj[item]
    def __setitem__(self, key, value):
        self.obj[key] = value
    def __delitem__(self, key):
        del self.obj[key]
    def __contains__(self, item):
        return item in self.obj
    def __iter__(self):
        return iter(self.obj)
    def __len__(self):
        return len(self.obj)
    def __add__(self, other):
        return self.obj + other
    def __radd__(self, other):
        return other + self.obj
    def __sub__(self, other):
        return self.obj - other
    def __rsub__(self, other):
        return other - self.obj
    def __mul__(self, other):
        return self.obj * other
    def __rmul__(self, other):
        return other * self.obj
    def __truediv__(self, other):
        return self.obj / other
    def __rtruediv__(self, other):
        return other / self.obj
    def __floordiv__(self, other):
        return self.obj // other
    def __rfloordiv__(self, other):
        return other // self.obj
    def __mod__(self, other):
        return self.obj % other
    def __rmod__(self, other):
        return other % self.obj
    def __pow__(self, other):
        return self.obj ** other
    def __rpow__(self, other):
        return other ** self.obj
    def __neg__(self):
        return -self.obj
    def __pos__(self):
        return +self.obj
    def __invert__(self):
        return ~self.obj
    def __lshift__(self, other):
        return self.obj << other
    def __rlshift__(self, other):
        return other << self.obj
    def __rshift__(self, other):
        return self.obj >> other
    def __rrshift__(self, other):
        return other >> self.obj
    def __and__(self, other):
        return self.obj & other
    def __rand__(self, other):
        return other & self.obj
    def __xor__(self, other):
        return self.obj ^ other
    def __rxor__(self, other):
        return other ^ self.obj
    def __or__(self, other):
        return self.obj | other
    def __ror__(self, other):
        return other | self.obj
    def __lt__(self, other):
        return self.obj < other
    def __le__(self, other):
        return self.obj <= other
    def __gt__(self, other):
        return self.obj > other
    def __ge__(self, other):
        return self.obj >= other
    def __ne__(self, other):
        return self.obj != other
    def __bool__(self):
        return bool(self.obj)
    def __abs__(self):
        return abs(self.obj)
    def __round__(self, ndigits=None):
        return round(self.obj, ndigits)
    def __complex__(self):
        return complex(self.obj)
    def __int__(self):
        return int(self.obj)
    def __float__(self):
        return float(self.obj)
    def __format__(self, format_spec):
        return format(self.obj, format_spec)
    def __dir__(self):
        return dir(self.obj)

def wrap(x):
    """May or may not mess up the namespace."""
    class a(type(x)):pass
    return a(x)

def call_if_callable[T](x:Callable[..., T] | T) -> T: # pylint:disable = E0602
    if callable(x):
        return x()
    else:
        return x

class Compose:
    """Compose"""
    def __init__(self, *transforms):
        self.transforms = flatten(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __add__(self, other: "Compose | Callable | SupportsIter"):
            return Compose(*self.transforms, other)

    def __str__(self):
        return f"Compose({', '.join(str(t) for t in self.transforms)})"

    def __iter__(self):
        return iter(self.transforms)

    def __getitem__(self, i): return self.transforms[i]
    def __setitem__(self, i, v): self.transforms[i] = v
    def __delitem__(self, i): del self.transforms[i]

def auto_compose(func: Optional[Callable | Sequence[Callable]]):
    """Composes `func` if it is a sequence, returns `identity` if it is None.

    Args:
        func (Optional[Callable  |  Sequence[Callable]]): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(func, Sequence): return Compose(*func)
    if func is None: return identity
    return func

def try_copy[T](x: T, force: bool = False) -> T: # pylint:disable = E0602
    if hasattr(x, 'copy'):
        if callable(x.copy): return x.copy() # type:ignore
        else: return x.copy # type:ignore
    else:
        try: return copy.copy(x)
        except TypeError: return copy.deepcopy(x) if force else x


def get_last_folder(path: str) -> str:
    return os.path.basename(os.path.normpath(path))

def ndims(sliceable: HasGetItem | Any) -> int:
    """ndims for inhomogeneous sequences (only takes 1st element into account)"""
    try: return ndims(sliceable[0])+1
    except TypeError: return 0

def shape(sliceable: HasGetItemAndLen | Any) -> list[int]:
    """Shape for inhomogeneous sequences (only takes 1st element into account)"""
    s = []
    if isinstance(sliceable, str): return [1]
    try: s.extend(shape(sliceable[0]))
    except (TypeError,IndexError): pass
    try: s.insert(0, len(sliceable))
    except (TypeError,IndexError): pass
    return s


class CacheRepeatIterator:
    """Iterator, caches each iterated value of the iterated object and yields it `n` times."""
    def __init__(self, obj:SupportsIter, n:int):
        self.obj = obj
        self.iter = iter(self.obj)
        self.n = n

    def __iter__(self):
        try:
            while True:
                elem = next(self.iter)
                for _ in range(self.n): yield elem
                del elem
        except StopIteration:
            self.iter = iter(self.obj)
            return

    def __len__(self):
        return len(self.obj)*self.n # type:ignore

@functools.cache
def factors_cached(n:int):
    """https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python"""
    return set(functools.reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
def factors(n:int):
    """https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python"""
    return set(functools.reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get__name__(obj:Any) -> str:
    return obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__


class ContinuingIterator:
    """Keeps iterating from the last position."""
    def __init__(self, obj):
        self.obj = obj
        self.iter = iter(self.obj)
        self.cur = 0
        self.len = len(obj)
    def __iter__(self):
        return self
    def __next__(self):
        self.cur+=1
        return next(self.iter)
    def __len__(self):
        return self.len - self.cur
    def reset(self):
        self.iter = iter(self.obj)
        self.cur = 0

class EndlessContinuingIterator:
    """Keeps iterating from the last position, restarts if iterated over the object."""
    def __init__(self, obj):
        self.obj = obj
        self.iter = iter(self.obj)
        self.cur = 0
        self.len = len(obj)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            self.cur+=1
            return next(self.iter)
        except StopIteration:
            self.reset()
            return next(self)
    def __len__(self):
        return len(self.obj)
    def reset(self):
        self.iter = iter(self.obj)
        self.cur = 0

class IterateSingleItem:
    """Yields `obj` `n` times."""
    def __init__(self, obj, n):
        self.obj = obj
        self.n = n
        self.cur = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.cur += 1
        if self.cur > self.n:
            self.cur = 0
            raise StopIteration
        return self.obj
    def __len__(self):
        return self.n


def getfoldersizeMB(folder) -> float:
    total = 0
    for root, _, files in os.walk(folder):
        for file in files:
            path = pathlib.Path(root) / file
            total += path.stat().st_size
    return total / 1024 / 1024


def listdir_fullpaths(folder) -> list[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def printargs(*args, **kwargs) -> None:
    for a in args: print(a)
    mlen = max([len(str(k)) for k in kwargs]) if len(kwargs) > 0 else 0
    for k,v in kwargs.items(): print(f'{k.ljust(mlen)} = {v}')


class SliceContainer:
    def __init__(self, obj, slices:int | Sequence[int | slice | Any]):
        self.obj = obj
        self.slices = slices

    def __call__(self):
        return self.obj[self.slices]


def sec_to_timestr(sec:float) -> str:
    """Converts seconds float to time string.

    >>> sec_to_time(0.45)
    "450ms"
    >>> sec_to_time(45.21)
    "45.21s"
    >>> sec_to_time(60)
    "01m:00s"
    >>> sec_to_time(3600)
    "01h:00m:00s"
    """
    if sec<1: return f'{sec*1000:.0f}ms'
    if sec<60: return f'{sec:.2f}s'
    m, s = divmod(sec, 60)
    if sec<3600:
        return f'{int(m):02d}m:{int(s):02d}s'
    h, m = divmod(m, 60)
    return f'{int(h):02d}h:{int(m):02d}m:{int(s):02d}s'

def get_unique_values[T](arr:Iterable[T]) -> list[T]: # pylint:disable=E0602
    return list(set(arr))


def rename_dict_key[K,V](d:dict[K,V], key:K, new_key:Any) -> dict[K,Any]: # pylint:disable=E0602
    d = d.copy()
    d[new_key] = d.pop(key)
    return d

def rename_dict_keys[K,V](d:dict[K,V], keys:dict[K,Any], allow_missing=False) -> dict[K,Any]:# pylint:disable=E0602
    d = d.copy()
    for k,v in keys.items():
        if k not in d:
            if allow_missing: continue
            else: raise KeyError(f'key {k} not found in dict')
        d[v] = d.pop(k)
    return d

def rename_dict_key_[K](d:dict[K, Any], key:K, new_key:Any) -> None: # pylint:disable=E0602
    d[new_key] = d.pop(key)

def rename_dict_keys_[K](d:dict[K, Any], keys:dict[K,Any], allow_missing=False) -> None: # pylint:disable=E0602
    for k,v in keys.items():
        if k not in d:
            if allow_missing: continue
            else: raise KeyError(f'key {k} not found in dict')
        d[v] = d.pop(k)

def rename_dict_key_contains_(d:dict, contains:Any, new_key:Any) -> None:
    """Rename first key that contains `contains` substring to `new_key`.

    Args:
        d (dict): _description_
        contains (Any): _description_
        new_key (Any): _description_
    """
    for k in d:
        if contains in k:
            d[new_key] = d.pop(k)
            return

def rename_dict_key_contains(d:dict, contains:Any, new_key:Any) -> dict:
    """Rename first key that contains `contains` substring to `new_key`.

    Args:
        d (dict): _description_
        contains (Any): _description_
        new_key (Any): _description_

    Returns:
        dict: _description_
    """
    d = d.copy()
    rename_dict_key_contains_(d, contains, new_key)
    return d

class ImpossibleException(Exception): pass


def sequence_to_md_table(s:Sequence[Sequence], keys:Optional[Sequence] = None, first_row_keys = False, transpose=False) -> str:
    """s: [row1, row2, row3, ...], or if `transpose` is [col1, col2, col3].

    Args:
        s (Sequence): _description_
        keys (Optional[Sequence], optional): _description_. Defaults to None.
        first_row_keys (bool, optional): _description_. Defaults to False.
    """
    if transpose: s = list(zip(*s))

    if keys is None:
        if first_row_keys:
            keys = s[0]
            s = s[1:]
        else: keys = list(range(len(s)))

    if keys is None: raise ValueError("This can't happen...")

    md = '| ' + ' | '.join([str(k) for k in keys]) + ' |\n'
    md += '| ' + ' | '.join([':-'] * len(keys)) + ' |\n'
    for row in s:
        md += '| ' + ' | '.join([str(v) for v in row]) + ' |\n'

    return md

__valid_fname_chars = frozenset(" -_.()")
def is_valid_fname(string:str) -> bool:
    if len(string) == 0: return False
    return all([c in __valid_fname_chars or c.isalnum() for c in string])

def to_valid_fname(string:str, fallback = '-', empty_fallback = 'empty', maxlen = 127, valid_chars = __valid_fname_chars) -> str:
    """Makes sure filename doesn't have forbidden characters and isn't empty or too long,
    this does not ensure a valid filename as there are a lot of other rules,
    but does a fine job most of the time.

    Args:
        string (str): _description_
        fallback (str, optional): _description_. Defaults to '-'.
        empty_fallback (str, optional): _description_. Defaults to 'empty'.
        maxlen (int, optional): _description_. Defaults to 127.

    Returns:
        _type_: _description_
    """
    if len(string) == 0: return empty_fallback
    return ''.join([(c if c in valid_chars or c.isalnum() else fallback) for c in string[:maxlen]])

def to_valid_varname(string:str, fallback = '_', empty_fallback = 'empty', firstdigit_fallback = 'd', maxlen = None) -> str:
    """Turn arbitrary string to a valid python variable name.

    Args:
        string (str): _description_
        fallback (str, optional): _description_. Defaults to '_'.
        empty_fallback (str, optional): _description_. Defaults to 'empty'.
        firstdigit_fallback (str, optional): _description_. Defaults to 'd'.
        maxlen (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if len(string) == 0: return empty_fallback
    name =  ''.join([(c if c == '_' or c.isalnum() else fallback) for c in string[:maxlen]])
    if name[0].isdigit(): name = f'{firstdigit_fallback}{name}'
    return name


def print_callable_defaults(c, end = '\n') -> None:
    signature = inspect.signature(c)
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            print(f"{k} = {v.default}", end=end)
        else:
            print(f"{k} = ", end  = end)