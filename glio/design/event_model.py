"""Attachemnt model"""
import time
from collections.abc import Iterable, Callable, Hashable
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, final
from types import MethodType
from contextlib import contextmanager
from functools import partial
import bisect

from ..python_tools import type_str, get__name__

__all__ = [
    "Cancel",
    "Callback",
    "EventCallback",
    "MethodCallback",
    "ConditionCallback",
    "BasicCallback",
    "EventModel",
    "EventModelWithPerformanceDebugging"
]
class Cancel(Exception): pass

def _raise_not_implemented(self, *args, **kwargs):
    raise NotImplementedError(f"{self.__class__.__name__} is missing `__call__`")

class Callback(ABC):
    order: float | int = 0
    __call__: Callable = _raise_not_implemented

    @abstractmethod
    def attach(self, __model: "EventModel") -> None: raise NotImplementedError(f"{self.__class__.__name__} is missing `attach` method.")
    def attach_default(self, __model: "EventModel") -> None: raise NotImplementedError(f"{self.__class__.__name__} is missing `attach_default` method.")

    def __str__(self):
        return f"{type_str(self)}()"
    @final
    @contextmanager
    def context(self, model: "EventModel"):
        try:
            self.attach(model)
            yield
        finally: model.remove(self)

class EventCallback(Callback, ABC):
    """Attaches to method `event`, requires `__call__`."""
    event: str

    @final
    def attach(self, model: "EventModel"): model._attach(event = self.event, fn = self)
    @final
    def attach_default(self, model: "EventModel"): model._attach_default(event = self.event, fn = self)

RESERVED_CALLBACK_ATTRIBUTES = (
    "attach",
    "attach_default",
    "order",
    "enter",
    "exit",
    "add",
    "remove",
    "remove_by_name",
    "add",
    "add_default",
    "event",
    "context",
    "extra",
    "without",
    "cond",
    "cbs"
)

class MethodCallback(Callback, ABC):
    """Attaches to methods that have the same name as methods of this callback, unless reserved."""
    cond: Callable | None = None
    @final
    def _attach(self, model: "EventModel", default=False):
        counter = 0
        for attr_name in dir(self):
            if not (attr_name.startswith("_") or attr_name in RESERVED_CALLBACK_ATTRIBUTES):
                method = getattr(self, attr_name)
                if isinstance(method, MethodType):
                    counter += 1
                    if default: model._attach_default(fn = method, event = attr_name, cond=self.cond, name=get__name__(self), ID=id(self))
                    else: model._attach(event = attr_name, fn = method, cond=self.cond, name=get__name__(self), ID=id(self))
        if counter == 0: logging.warning("There are no methods to attach in callback %s.", self)
    @final
    def attach(self, model: "EventModel"): self._attach(model,default=False)
    @final
    def attach_default(self, model: "EventModel"): self._attach(model, default=True)


class ConditionCallback(Callback, ABC):
    """Attaches to methods with conditions that are specified by calling special methdos like `on`, `every`, `first`."""
    default_events: Iterable[tuple[str, Callable | None]] | Any = ()

    def __init__(self):
        self.events: list[tuple[str, Callable | None]] = []

    # @abstractmethod
    # def __call__(self, model:"CallbackModel") -> Any: ...

    @final
    def cond_fn(self, event: str, cond: "Optional[Callable[[EventModel, int], bool]]" = None):
        """Runs `cond` on `even` and runs callback if `cond` returns True"""
        self.events.append((event, cond))
        return self
    @final
    def cond_every(self, event:str, every:int):
        """Runs callback `every` times `event` is fired."""
        self.events.append((event, lambda _,cur: cur % every == 0))
        return self
    @final
    def cond_first(self, event:str):
        self.events.append((event, lambda _,cur: cur == 0))
        return self

    @final
    def attach(self, model: "EventModel"):
        events = self.events if len(self.events) > 0 else self.default_events
        if len(events) == 0: logging.warning("No conditions are given for callback %s.", self) # type:ignore
        for event, cond in events:
            model._attach(event=event, fn=self, cond=cond)

    @final
    def attach_default(self, model: "EventModel"):
        events = self.events if len(self.events) > 0 else self.default_events
        if len(events) == 0: logging.warning("No conditions are given for default callback %s.", self) # type:ignore
        for event, cond in events:
            model._attach_default(event=event, fn=self, cond=cond)

class BasicCallback(Callback):
    """Doesn't attach to methods, can define `enter` and `exit`"""
    @final
    def attach(self, model: "EventModel"): model._attach(event = "__BasicCallback", fn = self, cond = lambda x: False)
    @final
    def attach_default(self, model: "EventModel"): model._attach_default(event = "__BasicCallback", fn = self, cond = lambda x: False)

class Event:
    def __init__(self, event:str):
        self.event = event

        self.cur = -1
        """Increments every `__call__`"""

        self.cbs: list[tuple[Callable | Callback, Callable | None]] = []
        self.orders: list[int | float] = []
        self.names: list[str] = []
        self.ids: list[Hashable] = []
        """
        List of tuples with the following signature: `(func, cond, order, name)`.

        e.g.: `[(func1, cond1, 0, "func1"), (func2, None, -4, "func2"), ...]`
        """

    def add(
        self,
        fn: Callable | Callback,
        cond: Optional[Callable],
        order: Optional[int | float] = None,
        name: Optional[str] = None,
        ID: Optional[Hashable] = None,
    ):
        if order is None: order = fn.order if hasattr(fn, 'order') else 0 #type:ignore
        if ID is None: ID = id(fn)
        if name is None: name = get__name__(fn)

        idx = bisect.bisect(self.orders, order) # type:ignore
        self.cbs.insert(idx, (fn, cond))
        self.orders.insert(idx, order) # type: ignore
        self.ids.insert(idx, ID)
        self.names.insert(idx, name)

    def remove(
        self, fn: Hashable | Callable | Callback | Iterable[Hashable | Callable | Callback],
    ) -> list[tuple[str, Callable | Callback, Callable | None, int | float, str, Hashable, ]]:
        removed = []
        if isinstance(fn, Iterable):
            for cb in fn: removed.extend(self.remove(cb))
            return removed
        if callable(fn): fn = id(fn)
        for i, ID in enumerate(self.ids):
            if ID == fn:
                removed.append((self.event, *self.cbs[i], self.orders[i], self.names[i], self.ids[i]))
                del self.cbs[i]
                del self.orders[i]
                del self.names[i]
                del self.ids[i]
        return removed

    def remove_by_name(
        self, name: str | Callable | Callback | Iterable[str | Callable | Callback]
    ) -> list[tuple[str, Callable | Callback, Callable | None, int | float, str, Hashable, ]]:
        removed = []
        if callable(name): name = get__name__(name)
        if not isinstance(name, str):
            for cb in name: removed.extend(self.remove_by_name(cb))
            return removed
        else:
            for i, self_name in enumerate(self.names): # type:ignore
                if name == self_name:
                    removed.append((self.event, *self.cbs[i], self.orders[i], self.names[i], self.ids[i]))
                    del self.cbs[i]
                    del self.orders[i]
                    del self.names[i]
                    del self.ids[i]
            return removed

    def __call__(self, model: "EventModel", *args, **kwargs) -> Any:
        self.cur += 1
        return [fn(model, *args, **kwargs) for fn, cond in self.cbs if (cond is None or cond(model, self.cur))]


class EventModel(ABC):
    def __init__(self, cbs: Optional[Iterable[Callback]] = None, default_cbs: Optional[Iterable[Callback]] = None):
        # create events
        self._events:dict[str, Event] = {}
        self._default_events:dict[str, Event] = {}

        # set cbs for access
        self.cbs = []
        self.default_cbs = []

        # add cbs
        if cbs is not None:
            for cb in cbs: self.add(cb)
        if default_cbs is not None:
            for cb in default_cbs: self.add_default(cb)


    @final
    def __getattr__(self, attr: str) -> Any:
        if (attr not in self._events) and (attr not in self._default_events): raise AttributeError(f'`{attr}` not found in {self.__class__.__name__} or any of the callbacks')
        return partial(self.event, attr)

    def clear(self):
        self._events:dict[str, Event] = {}
        self._default_events:dict[str, Event] = {}

        self.cbs = []
        self.default_cbs = []

    def _attach(
        self,
        event: str,
        fn: Callback | Callable,
        cond: Optional[Callable] = None,
        order: Optional[int | float] = None,
        name: Optional[str] = None,
        ID: Optional[Hashable] = None,
    ):
        """Attach function `fn` to method `event`."""
        if event not in self._events: self._events[event] = Event(event)
        self._events[event].add(fn = fn, cond = cond, order = order, ID = ID, name=name)

    def _attach_default(
        self,
        event: str,
        fn: Callback | Callable,
        cond: Callable | None = None,
        order: Optional[int | float] = None,
        name: Optional[str] = None,
        ID: Optional[Hashable] = None,
    ):
        """Attach function `fn` to default method `event`."""
        if event not in self._default_events: self._default_events[event] = Event(event)
        self._default_events[event].add(fn = fn, cond = cond, order = order, ID = ID, name=name)

    @final
    def add(self, cb: Callback | Iterable[Callback]):
        """Attach callback `cb` using its method `attach`."""
        if isinstance(cb, Callback): cb = [cb]
        for i in cb: i.attach(self)
        self.cbs.extend(cb)

        for i in sorted(cb, key=lambda x: x.order if hasattr(x, 'order') else 0):
            if hasattr(i, "enter"): i.enter(self) # type:ignore

    @final
    def add_default(self, cb: Callback | Iterable[Callback]):
        """Attach callback `cb` using its method `attach_default`."""
        if isinstance(cb, Callback): cb = [cb]
        for i in cb: i.attach_default(self)
        self.default_cbs.extend(cb)

        for i in sorted(cb, key=lambda x: x.order if hasattr(x, 'order') else 0):
            if hasattr(i, "enter"): i.enter(self) # type:ignore

    # @property
    # def cbs(self):
    #     return [j[0] for i in self.events.values() if len(i.cbs) > 0 for j in i.cbs] + [j[0] for i in self.default_events.values() if len(i.cbs) > 0 for j in i.cbs]

    def event(self, event:str, *args, **kwargs) -> Any:
        if event in self._events:
            return self._events[event](self, *args, **kwargs)
        elif event in self._default_events:
            return self._default_events[event](self, *args, **kwargs)

    @final
    def remove(self, cbs: Callback | Callable | Iterable[Callable | Callback]):
        if callable(cbs): cbs = [cbs]

        for cb in sorted(cbs, key=lambda x: x.order if hasattr(x, 'order') else 0): # type:ignore
            if hasattr(cb, "exit"): cb.exit(self) # type:ignore

            if cb in self.cbs: self.cbs.remove(cb)
            if cb in self.default_cbs: self.default_cbs.remove(cb)

        for event in self._events.copy().values():
            event.remove(cbs)
            if len(event.cbs) == 0: del self._events[event.event]

    @final
    def remove_by_name(self, names:str|Iterable[str]):
        removed: list[tuple[str, Callable | Callback, Callable | None, int | float, str, Hashable, ]] = []

        for event in self._events.copy().values():
            removed.extend(event.remove_by_name(names))
            if len(event.cbs) == 0: del self._events[event.event]

        for cb in sorted(removed, key=lambda x: x[3]):
            if hasattr(cb, "exit"): cb.exit(self) # type:ignore

            if cb in self.cbs: self.cbs.remove(cb)
            if cb in self.default_cbs: self.default_cbs.remove(cb)

        return removed

    @final
    @contextmanager
    def context(self, name:str, extra:Optional[Callback | Iterable[Callback]] = (), without: Optional[str | Iterable[str]] = (), fire_events = False):
        if isinstance(extra, Callback): extra = [extra]
        if isinstance(without, str): without = [without]

        #add and remove cbs to the model inside the context
        if extra is not None: self.add(extra)
        if without is not None:
            removed: list[tuple[str, Callable | Callback, Callable | None, int | float, str, Hashable, ]] = self.remove_by_name(without)
        else: removed = []

        try:
            if fire_events: self.event(f'before_{name}')
            yield
            if fire_events: self.event(f'after_{name}')
        except Cancel as cancel:
            if str(cancel) != name: raise cancel
        finally:
            if extra is not None: self.remove(extra)
            for i in removed: self._attach(*i)

    @final
    @contextmanager
    def extra(self, callbacks: Callback | Iterable[Callback]):
        try:
            self.add(callbacks)
            yield
        finally: self.remove(callbacks)

    @final
    @contextmanager
    def without(self, callbacks: str | Iterable[str]):
        removed: list[tuple[str, Callable | Callback, Callable | None, int | float, str, Hashable, ]] = self.remove_by_name(callbacks)
        try: yield
        finally:
            for i in removed: self._attach(*i) # type:ignore

class EventWithPerformanceDebugging(Event):
    def __init__(self, event: str):
        super().__init__(event)
        self.cbs_time : dict[Any, list[float]] = {}

    def __call__(self, model: "EventModel", *args, **kwargs) -> Any:
        self.cur += 1
        res = []
        for cb, name, id_ in zip(self.cbs, self.names, self.ids):
            fn, cond = cb
            start = time.perf_counter()
            if (cond is None or cond(model, self.cur)):
                res.append(fn(model, *args, **kwargs))
            time_took = time.perf_counter() - start
            n = f'{name} ({id_})'
            if n not in self.cbs_time: self.cbs_time[n] = [time_took]
            else: self.cbs_time[n].append(time_took)
        return res

class EventModelWithPerformanceDebugging(EventModel, ABC):
    def __init__(self, cbs: Optional[Iterable[Callback]], default_cbs: Optional[Iterable[Callback]] = None):
        super().__init__(cbs, default_cbs)
        self.events_time : dict[str, list[float]] = {}

    def _attach(
        self,
        event: str,
        fn: Callback | Callable,
        cond: Optional[Callable] = None,
        order: Optional[int | float] = None,
        name: Optional[str] = None,
        ID: Optional[Hashable] = None,
    ):
        if event not in self._events: self._events[event] = EventWithPerformanceDebugging(event)
        self._events[event].add(fn = fn, cond = cond, order = order, ID = ID, name=name)

    def _attach_default(
        self,
        event: str,
        fn: Callback | Callable,
        cond: Callable | None = None,
        order: Optional[int | float] = None,
        name: Optional[str] = None,
        ID: Optional[Hashable] = None,
    ):
        if event not in self._default_events: self._default_events[event] = EventWithPerformanceDebugging(event)
        self._default_events[event].add(fn = fn, cond = cond, order = order, ID = ID, name=name)

    def event(self, event:str, *args, **kwargs) -> Any:
        start = time.perf_counter()
        if event in self._events:
            res = self._events[event](self, *args, **kwargs)
        elif event in self._default_events:
            res = self._default_events[event](self, *args, **kwargs)
        else: res = None
        time_took = time.perf_counter() - start
        if event not in self.events_time: self.events_time[event] = [time_took]
        else: self.events_time[event].append(time_took)
        return res

    def get_events_sum_time(self): return {k:sum(v) for k,v in self.events_time.items()}
    def get_events_avg_time(self): return {k:sum(v)/len(v) for k,v in self.events_time.items()}
    def get_events_percent_time(self):
        sums = self.get_events_sum_time()
        total = sum(list(sums.values()))
        return {k:v/total for k,v in sums.items()}


    def get_cbs_time(self):
        times:dict[str,float] = {}
        for event in list(self._events.values()) + list(self._default_events.values()):
            event_cbtimes:dict[str,list[float]] = event.cbs_time # type:ignore
            event_cbsumtimes = {k:sum(v) for k,v in event_cbtimes.items()}
            for k,v in event_cbsumtimes.items():
                if k not in times: times[k] = v
                else: times[k] += v
        return times

    def get_cbs_percent_time(self):
        times = self.get_cbs_time()
        total = sum(list(times.values()))
        return {k:v/total for k,v in times.items()}