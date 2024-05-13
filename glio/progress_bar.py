from collections.abc import Callable
import math
import time
from IPython import display
from .python_tools import sec_to_timestr

def _print_inline(*args, **kwargs): print(*args, **kwargs, end = "                                          \r")

class _Printer(str):
    def __repr__(self): # pylint:disable=E0306
       return self

def _create_text_displayer(dislay_:display.DisplayHandle):
    def display_text(text): dislay_.update(_Printer(text))
    return display_text

class PBar:
    def __init__(self, obj, length = 20, step = 0.05, symbol = "▉", display_fn:str | Callable='display'):
        self.obj = obj
        self.symbol = symbol
        self.obj_len = len(obj)
        self.pbar_len = length
        if self.obj_len != 0: self.symbols_per_step = self.pbar_len / self.obj_len
        else: self.symbols_per_step = 1
        self.step = step
        if self.step < 1: self._actual_step = int(math.ceil(self.obj_len * self.step))
        else: self._actual_step = int(self.step)

        self.text = ""

        if display_fn == "print": display_fn = _print_inline
        elif display_fn == "display":

            self.display:display.DisplayHandle = display.display("",display_id=True) # type:ignore
            self.display_fn = _create_text_displayer(self.display) # type:ignore
        elif callable(display_fn): self.display_fn = display_fn
        else: raise ValueError(f"display_fn must be 'print', 'display' or a callable, not {display_fn} of type {type(display_fn)}")

        self.display_fn('')
        self.start = time.time()

    def set_obj(self, obj):
        self.obj = obj
        self._restart_iter()

    def _restart_iter(self):
        self.i = 0
        self.obj_len = len(self.obj)
        if self.obj_len != 0: self.symbols_per_step = self.pbar_len / self.obj_len
        else: self.symbols_per_step = 1
        self.iterable = iter(self.obj)
        if self.step < 1: self._actual_step = int(math.ceil(self.obj_len * self.step))
        else: self._actual_step = int(self.step)
        self.start = time.time()

    def __iter__(self):
        self._restart_iter()
        return self

    def __next__(self):
        if self.i == 0: self.start = time.time()
        try: n = next(self.iterable)
        except StopIteration as exc:
            self._restart_iter()
            raise exc

        if self.i % self._actual_step == 0:
            bar_len = int(math.ceil(self.symbols_per_step * self.i+1))
            elapsed = time.time() - self.start
            ops_per_sec = self.i / max(1e-6, elapsed)
            if ops_per_sec > 1: ops_per_sec_str = f"{ops_per_sec:.2f}ops/s, "
            elif ops_per_sec == 0: ops_per_sec_str = ""
            else: ops_per_sec_str = f"{(sec_to_timestr(1/ops_per_sec))}/ops, "
            if ops_per_sec == 0: remaining = 0
            else: remaining = (self.obj_len - self.i) / ops_per_sec
            self.display_fn(f"{(self.symbol * (bar_len)).ljust(self.pbar_len+1)}| {self.i+1}/{self.obj_len} | {ops_per_sec_str}elapsed: {sec_to_timestr(elapsed)}, rem: {sec_to_timestr(remaining)} | {self.text}")

        self.i += 1

        return n

    def write(self, text):
        self.text = text


    def __getitem__(self, key): return self.obj[key]
    def __setitem__(self, key, value): self.obj[key] = value
    def __delitem__(self, key): del self.obj[key]
    def __len__(self): return len(self.obj)
    def __getattr__(self, attr): return getattr(self.obj, attr)

