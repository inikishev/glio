

from collections.abc import Callable
from types import MethodType
class Event:
    ORDER = 0
RESERVED = ('add',)

class SimpleEventModel:
    def __init__(self, events: list[Event]):
        self.events: dict[str, list[Callable]] = {}
        self.add(events)

    def add(self, events: Event | list[Event]):
        if isinstance(events, Event): events = [events]
        for event in events:
            for method_name in dir(event):
                if not (method_name.startswith('_') or method_name in RESERVED):
                    method = getattr(event, method_name)
                    if isinstance(method, MethodType):
                        if method_name not in self.events: self.events[method_name] = [method]
                        else: self.events[method_name].append(method)

    def event(self, name:str):
        if name in self.events:
            for event in self.events[name]:
                event(self)
