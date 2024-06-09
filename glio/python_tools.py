"""Инструменты, использующие встроенные библиотеки Python"""
from collections.abc import Sequence,Callable, Iterable, Mapping
from typing import Any, Protocol, Optional
from time import perf_counter
from contextlib import contextmanager
import random, os, pathlib
import functools, operator
import copy
from functools import partial

class HasGetItem(Protocol):
    def __getitem__(self, __key:Any) -> Any: ...

class HasGetItemAndLen(Protocol):
    def __getitem__(self, __key:Any) -> Any: ...
    def __len__(self) -> int: ...

class HasGetItemInt(Protocol):
    def __getitem__(self, __key:int) -> Any: ...

SupportsIter = Iterable | HasGetItemInt

def isiterable(x:Any) -> bool:
    if hasattr(x, '__iter__'): return not isinstance(x, type)
    if hasattr(x, '__getitem__'):
        try:
            x[0] # pylint:disable=W0104
            return True
        except Exception: # pylint:disable=W0718
            return False
    return False

def hasgetitem(x:Any) -> bool: return hasattr(x, '__getitem__')

def ismapping(x:Any) -> bool: return hasattr(x,"keys") and hasattr(x, "values") and hasattr(x, "items") and hasattr(x, "__getitem__")

def key_value(iterable: SupportsIter):
    """
    Создаёт аналогичный интерфейс для итерации по последовательностям и словарям. Возвращает итератор, возвращающий ключ и значение.
    """
    if ismapping(iterable): return iterable.items() # type:ignore
    return enumerate(iterable) #type:ignore

def flatten_generator(iterable:SupportsIter):
    """
    Генератор по вложенной последовательности. Например, `[1, [2, 3], [[4]]]` -> `[1,2,3,4]`. Пустые под-последовательности удаляются.
    """
    if isiterable(iterable):
        for i in key_value(iterable):
            if isiterable(i[1]):
                yield from flatten_generator(i[1])
            else: yield i[1]
    else: yield iterable

def flatten(iterable: SupportsIter) -> list[Any]:
    """
    Выравнивает вложенную последовательность. Например, `[1, [2, 3], [[4]]]` -> `[1,2,3,4]`. Пустые под-последовательности удаляются.
    """
    if isiterable(iterable):
        return [a for i in iterable for a in flatten(i)]
    else:
        return [iterable]

def flatten_by_tree_generator(iterable: HasGetItem, tree):
    """
    Генератор, выравнивает вложенную последовательность согласно структуре "дерева".
    """
    if isiterable(tree):
        for k,v in key_value(tree):
            if hasgetitem(v):
                yield from flatten_by_tree_generator(iterable[k], v)
            else: yield iterable[k]
    else: yield iterable

def flatten_by_tree(iterable: HasGetItem, tree):
    """
    Выравнивает рекурсивную последовательность согласно структуре "дерева".
    """
    return list(flatten_by_tree_generator(iterable, tree))


def apply_tree(x: Any, funcs, cached: Any = None, flatten: bool = False): #pylint: disable=W0621
    """
    Применяет дерево функций к объекту x. Возвращает список той же вложенной формы, что и `funcs`, с `x` после применения каждой функции. `None` может использоваться вместо функции и идентичен тождественной функции, для ветвления без обработки.

    Дерево — это рекурсивная последовательность вызываемых объектов:

    ```py
    [функция1, функция2, [функция3, функция4], [функция5]]
    ```

    Такое дерево возвращает:

    ```py
    [func1(x), func2(func1(x)), [func3(func2(func1(x))), func4(func3(func2(func1(x))))), [func5(func2(func1(x))))], [func5(func2(func1(x)) ))]]
    ```

     Таким образом, применяется `func1`, затем `func2`, затем `X` ветвится на две копии, `[func3, func4]` применяется к первой и `[func5]` ко второй. Глубина дерева не ограничена программно.

     `cached` — первая функция в дереве будет заменена значением `cached`, если оно не `None`. Например, первой функцией может быть `read_image`, и если есть уже загруженное изображение, его можно передать в качестве аргумента для `cached`, чтобы повторно не декодировать."""
    if ismapping(funcs):
        y = {}
        is_dict = True
    else:
        y = []
        is_dict = False

    for k, func in key_value(funcs):

        # Объект вызываем
        if callable(func):
            # Только 1 элемент может быть кэширован
            if cached is not None:
                x = cached
                cached = None
            else: x = func(x)
            if is_dict: y[k] = x
            else: y.append(x)#pyright:ignore[reportAttributeAccessIssue]
        # Объект не вызываем
        elif not func:
            if is_dict: y[k] = x
            else: y.append(x)#pyright:ignore[reportAttributeAccessIssue]
        # объект - последовательность, данная функция вызывается рекурсивно на этой часте объекта
        else:
            if is_dict: y[k] = {}
            else: y.append([])#pyright:ignore[reportAttributeAccessIssue]
            y[k] = apply_tree(x, func, cached)
    return y if not flatten else flatten_by_tree(y, funcs)


def apply_recursive_(iterable: SupportsIter, func: Callable):
    """
    Применяет func ко всем элементам во вложенном итерируемом объекте.
    """
    if isiterable(iterable):
        for k,v in key_value(iterable):
            if isiterable(v): apply_recursive_(v, func)
            else: iterable[k] = func(v) # type:ignore
    else: iterable = func(iterable)

def get_first_recursive(iterable: SupportsIter) -> Any:
    """
    Возвращает первый элемент вложенной последовательности.
    """
    try:
        for i in iterable: return get_first_recursive(i)
    except Exception: # pylint: disable=broad-except
        if ismapping(iterable): return get_first_recursive(iterable[list(iterable.keys())[0]]) # type:ignore
    return iterable

def perf_counter_deco(func: Callable, n: int = 1):
    """Декоратор для замера времени выполнения функции"""
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

def flexible_filter(keys, filters):
    """Если фильтр - функция, используется его возвращаемое значение для каждого ключа;

    если строка: возвращает True, если является подстрокой ключа;

    если список: возвращает True, если хотя бы одна подстрока ключей есть в списке;"""
    for f in filters:
        if callable(f): keys = [k for k in keys if f(k)]
        elif isinstance(f, (int, float, str)): keys = [k for k in keys if str(f).lower() in str(k).lower()]
        elif isinstance(f, (list, tuple)): keys = [k for k in keys if any([str(i).lower() in str(k).lower() for i in f])]
    return keys

class ExhaustingIterator:
    """
    Выполняет итерацию по объекту, но останавливает итерацию, как если бы объект имел другую длину.

    Итерация возобновляется начиная с последнего элемента после её остановки.
    """
    def __init__(self, obj:HasGetItemAndLen, length: int, shuffle:bool = True):
        """
        У объекта должны быть методы `__getitem___` и `__len__`.
        """
        self.obj = obj
        self.length = length
        self.cur = -1 # +1 в начале __next__
        self.iterations = 0
        self.indexes = list(range(len(self.obj)))
        self.shuffle = shuffle
        #if self.shuffle: random.shuffle(self.indexes) будет сделан в след. __next__ когда cur = 0
        self.n_samples = len(self.indexes)
        self.not_doing_0th_element = False

    def __iter__(self):
        return self

    def __next__(self):
        self.cur+=1
        if self.not_doing_0th_element and self.cur % self.length == 0:
            self.not_doing_0th_element = False
            self.cur -= 1 # если длина 2 будут взяты 0, 1, стоп на 2, cur -= 1 = 1? в сл. итерации self.cur+=1 = 2
            raise StopIteration()
        self.not_doing_0th_element = True
        element = self.cur % self.n_samples # элемент, # при длине 3 будет 0, 3, 6
        if element == 0:
            if self.shuffle: random.shuffle(self.indexes)
        return self.obj[self.indexes[element]]

    def __len__(self): return self.length


class ItemUnderIndex:
    """Содержит ссылку на индексируемый объект и индекс элемента в объекте, возвращает элемент под индексом."""

    def __init__(self, obj:HasGetItem, index: int):
        self.obj = obj
        self.index = index

    def __call__(self):
        return self.obj[self.index]

class ItemUnder2Indexes:
    """Содержит ссылку на индексируемый объект и два индекса элемента в объекте, возвращает элемент под индексами."""

    def __init__(self, obj:HasGetItem, index1: int, index2:int):
        self.obj = obj
        self.index1 = index1
        self.index2 = index2

    def __call__(self):
        return self.obj[self.index1][self.index2]



def reduce_dim(a:SupportsIter) -> list:
    """Удаляет один уровень вложенности в последовательности."""
    return functools.reduce(operator.iconcat, a, []) # type:ignore


def subclasses_recursive(cls:type | Any) -> set[type]:
    """Рекурсивно получает все под-классы класса."""
    if not isinstance(cls, type): cls = type(cls)
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in subclasses_recursive(c)])

def type_str(obj:object | type) -> str:
    """Возвращает название класса объекта с пространством имен. Например, `torch.nn.Conv2d`"""
    if not isinstance(obj, type): obj = type(obj)
    return repr(obj).replace("<class '", "").replace("'>", "")



def to_callable(obj, *args, **kwargs):
    """Возвращает вызываемый объект."""
    if isinstance(obj, bool): return lambda _: obj
    elif not isinstance(obj, type):
        if len(args) == 0 and len(kwargs) == 0: return obj
        else: return partial(obj, *args, **kwargs)
    else: return obj(*args, **kwargs)


def int_at_beginning(s:str) -> int | str:
    """Возвращает целое число любой длины, если строка с него начинается."""
    i = 1
    num = None
    while True:
        try:
            num = int(s[:i])
            i+=1
        except ValueError:
            if num is None: return s
            return num

def dict_to_table(data:Mapping, key, order = None, sort_key: Optional[Callable] = None) -> str:
    """Возвращает таблицу по словарю."""
    if sort_key is not None: data = {k: v for k, v in sorted(data.items(), key=sort_key)}
    keys = []
    if order is not None: keys = order
    if key not in keys: keys.append(key)
    for k in data:
        for k1 in sorted(list(data[k].keys()), key = int_at_beginning):
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
    """Возвращает список полных путей ко всем файлам в папке, рекурсивно, если `recursive` = `True`, фильтрует с использованием `extensions` и `path_filter`"""
    all_files = []
    if isinstance(extensions, str): extensions = [extensions]
    if extensions is not None: extensions = tuple(extensions)
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

def get0(x:HasGetItem) -> Any: return x[0]
def get1(x:HasGetItem) -> Any: return x[1]
def getlast(x:HasGetItem) -> Any: return x[-1]

def identity(x:Any) -> Any: return x
def identity_kwargs(x:Any, *args, **kwargs) -> Any: return x
def identity_if_none(func:Callable | None):
    if func is None: return identity
    return func
def identity_kwargs_if_none(func:Callable | None) -> Callable:
    if func is None: return identity_kwargs
    return func

def ensure_list(x) -> list:
    if isinstance(x, list): return x
    if isiterable(x): return list(x)
    return [x]

def pretty_print_dict(d: dict) -> None:
    """Выводит вложенные словари при помощи JSON."""
    import json
    print(json.dumps(d, indent=4, sort_keys=False))

class Wrapper:
    """Оборачивает объект и копирует его методы. Позволяет хранить аттрибуты во встроенных типах."""
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
    """Оборачивает объект и копирует его методы. Позволяет хранить аттрибуты во встроенных типах. Возможно влияет на пространство имён."""
    class a(type(x)):pass
    return a(x)

def call_if_callable(x:Callable | Any) -> Callable | Any:
    if callable(x):
        return x()
    else:
        return x

class Compose:
    """Композиция функций в обратном порядке"""
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
    if isinstance(func, Sequence): return Compose(*func)
    if func is None: return identity
    return func

def try_copy(x: Any, force: bool = False) -> Any:
    if hasattr(x, 'copy'):
        if callable(x.copy): return x.copy()
        else: return x.copy
    else:
        try: return copy.copy(x)
        except TypeError: return copy.deepcopy(x) if force else x


def get_last_folder(path: str) -> str:
    return os.path.basename(os.path.normpath(path))

def ndims(sliceable: HasGetItem | Any) -> int:
    """Возвращает ранг объекта с методом __getitem__.
    Для однородных массивов эквивалентен `numpy.ndarray.ndim`.
    Учитывает только первый элемент каждого измерения, поэтому работает с неоднородными объектами."""
    try: return ndims(sliceable[0])+1
    except TypeError: return 0

def shape(sliceable: HasGetItemAndLen | Any) -> list[int]:
    """Возвращает размерность объекта с методом __getitem__.
    Для однородных массивов эквивалентен `numpy.ndarray.shape`.
    Учитывает только первый элемент каждого измерения, поэтому работает с неоднородными объектами."""
    s = []
    if isinstance(sliceable, str): return [1]
    try: s.extend(shape(sliceable[0]))
    except (TypeError,IndexError): pass
    try: s.insert(0, len(sliceable))
    except (TypeError,IndexError): pass
    return s

# def full_shape(sliceable) -> list:
#     """Возвращает размерность объекта с методом __getitem__.
#     Для однородных массивов эквивалентен `numpy.ndarray.shape`.
#     Учитывает только первый элемент каждого измерения, поэтому работает с неоднородными объектами."""
#     s = []
#     try:
#         for sub in sliceable:
#             try: s.append(full_shape(sub))
#             except (TypeError, IndexError): pass
#     except (TypeError, IndexError): return len(sliceable)
#     return  s


class CacheRepeatIterator:
    """Итератор, кэширует элемент итерируемого объекта и возвращает n раз."""
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
    """Продолжает итерацию с последнего места."""
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
    """Продолжает итерацию с последнего места, перезапускает при окончании итерируемого объекта."""
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


def getfoldersizeMB(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for file in files:
            path = pathlib.Path(root) / file
            total += path.stat().st_size
    return total / 1024 / 1024


def listdir_fullpaths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def printargs(*args, **kwargs):
    for a in args: print(a)
    mlen = max([len(str(k)) for k in kwargs])
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

def get_unique_values(arr:Iterable):
    return list(set(arr))


def rename_dict_key(d:dict, key:Any, new_key:Any) -> dict:
    d = d.copy()
    d[new_key] = d.pop(key)
    return d

def rename_dict_keys(d:dict, keys:dict[Any,Any], allow_missing=False) -> dict:
    d = d.copy()
    for k,v in keys.items():
        if k not in d:
            if allow_missing: continue
            else: raise KeyError(f'key {k} not found in dict')
        d[v] = d.pop(k)
    return d

def rename_dict_key_(d:dict, key:Any, new_key:Any) -> None:
    d[new_key] = d.pop(key)

def rename_dict_keys_(d:dict, keys:dict[Any,Any], allow_missing=False) -> None:
    for k,v in keys.items():
        if k not in d:
            if allow_missing: continue
            else: raise KeyError(f'key {k} not found in dict')
        d[v] = d.pop(k)
        d[v] = d.pop(k)

def rename_dict_key_contains_(d:dict, key:Any, new_key:Any) -> None:
    for k in d:
        if key in k:
            d[new_key] = d.pop(k)
            return

def rename_dict_key_contains(d:dict, key:Any, new_key:Any) -> dict:
    d = d.copy()
    rename_dict_key_contains_(d, key, new_key)
    return d

class ImpossibleException(Exception): pass


def __cat(iterable, value):
    for seq in iterable:
        yield [value] + seq

def __unsqueeze(iterable):
    for i in iterable: yield [i]

def icartesian(iterables):
    if len(iterables) == 1:
        yield from __unsqueeze(iterables[0])
    for seq in iterables:
        for val in seq:
            yield from __cat(iterable=icartesian(iterables[1:]), value=val)
