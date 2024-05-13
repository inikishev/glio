"""Callback moedl"""
from functools import partial
from contextlib import contextmanager
from types import MethodType, FunctionType
from typing import Callable, Optional, Any
from collections.abc import Iterable
from copy import copy
from ..python_tools import subclasses_recursive, type_str
def context(inner):
    """Decorator that adds context to a Callback method"""
    def outer(*args, **kwargs):
        with args[1].context(inner.__name__):
            return inner(*args, **kwargs)
    return outer

class Callback:
    ORDER = 0
    DEFAULT = False
    TEMP = False

    def _remove(self, model: "CallbackModel"):
        model.remove([self])

    def copy(self):
        """Copies self and one level of nesting"""
        self_copy = copy(self)
        self_copy.__dict__ = self.__dict__.copy()
        self_copy.__dict__ = {k: (v.copy() if (hasattr(v, 'copy') and callable(v.copy)) else copy(v)) for k, v in self_copy.__dict__.items()}
        return self_copy

    def __str__(self):
        return f"{type_str(self)}()"


RESERVED = ('enter', 'exit', 'callbacks', 'context', 'add', 'remove', 'remove_by_name', 'extra', 'without', 'event')
class Cancel(Exception): pass

class CallbackModel:
    """
    A reasonaly fast implementation of Jeremy Howard course style callbacks model, except this time the whole thing is made of callbacks.

    Whenever a callback is added to the model, `enter` method is called in it. That includes creating a CallbackModel object, using `add` method or setting `callbacks` variable.

    Whenever a callback is removed from the model, `exit` method is called in it. This includes using `remove` method or setting `callbacks` variable.

    Note this this is taken from a deep learning framework which is why it might have more stuff than it needs to.
    """
    def __init__(self, callbacks:Optional[Iterable[Callback]], default_callbacks:Optional[Iterable[Callback]], *args, **kwargs):
        """
        Initializes the object with the given callbacks.
        Parameters:
            callbacks (list or Callback): callbacks to add to the object.
            default_callbacks (list or Callback): callbacks with a single method, will be added only if that method isn't in any of the `callbacks`.
        """
        self._fs_main: dict[str, list[Callable]] = {}
        self._fs_before: dict[str, list[Callable]] = {}
        self._fs_after: dict[str, list[Callable]] = {}
        self._order_main: dict[str, list[float]] = {}
        self._order_before: dict[str, list[float]] = {}
        self._order_after: dict[str, list[float]] = {}
        self._callbacks: list[Callback] = []

        if isinstance(callbacks, Callback): callbacks = (callbacks, )
        elif callbacks is None: callbacks = []
        self.add(callbacks, enter = False)
        try: self.init(*args, **kwargs)
        except AttributeError: pass

        if isinstance(default_callbacks, Callback): default_callbacks = (default_callbacks, )
        callbacks_to_add = []
        if default_callbacks is not None:
            for callback in default_callbacks:
                method_name = [method_name for method_name in dir(callback) if
                               isinstance(getattr(callback, method_name), MethodType)
                               and not (method_name.startswith('_') or method_name in RESERVED)]
                if len(method_name) != 1:
                    if not callback.DEFAULT: raise ValueError(f'default callbacks must have only one main method or default=True; {callback} has {method_name} and default = {callback.DEFAULT}')
                    else: callbacks_to_add.append(callback)
                else:
                    method_name = method_name[0]
                    if method_name not in self._fs_main:
                        callbacks_to_add.append(callback)
        if len(callbacks_to_add) > 0: self.add(callbacks_to_add)

        [b.enter(self) for b in sorted(callbacks, key = lambda b: b.ORDER) if hasattr(b, 'enter')] # pylint:disable=W0106 #type:ignore

    def _add(self, callback: Callback):
        self._callbacks.append(callback)
        for method_name in dir(callback):
            if not (method_name.startswith('_') or method_name in RESERVED):
                method = getattr(callback, method_name)
                if isinstance(method, MethodType):
                    # we do this to maybe get a lil performance?
                    if method_name.startswith('before_'):
                        if method_name in self._fs_before:
                            self._fs_before[method_name].append(method)
                            self._order_before[method_name].append(callback.ORDER)
                        else:
                            self._fs_before[method_name] = [method]
                            self._order_before[method_name] = [callback.ORDER]
                    elif method_name.startswith('after_'):
                        if method_name in self._fs_after:
                            self._fs_after[method_name].append(method)
                            self._order_after[method_name].append(callback.ORDER)
                        else:
                            self._fs_after[method_name] = [method]
                            self._order_after[method_name] = [callback.ORDER]
                    else:
                        if method_name in self._fs_main:
                            self._fs_main[method_name].append(method)
                            self._order_main[method_name].append(callback.ORDER)
                        else:
                            self._fs_main[method_name] = [method]
                            self._order_main[method_name] = [callback.ORDER]

    def add(self, callbacks:Callback | Iterable[Callback], enter = True):
        """
        Add callbacks to the current object.

        Parameters:
            callbacks (list[Callback]): The callbacks to be added.
            enter (bool): Whether to enter the callbacks or not. Defaults to True.
        """
        if isinstance(callbacks, Callback): callbacks = (callbacks, )
        [self._add(b) for b in callbacks] # pylint:disable=W0106 #type:ignore
        self._sort_fs()
        if enter: [b.enter(self) for b in sorted(callbacks, key = lambda b: b.ORDER) if hasattr(b, 'enter')] # pylint:disable=W0106 #type:ignore

    def _remove(self, callback: Callback):
        self._callbacks.remove(callback)
        for method_name in dir(callback):
            if not (method_name.startswith('_') or method_name in RESERVED):
                method = getattr(callback, method_name)
                if isinstance(method, MethodType):
                    if method_name.startswith('before_'):
                        self._fs_before[method_name].remove(method)
                        self._order_before[method_name].remove(callback.ORDER)
                    elif method_name.startswith('after_'):
                        self._fs_after[method_name].remove(method)
                        self._order_after[method_name].remove(callback.ORDER)
                    else:
                        self._fs_main[method_name].remove(method)
                        self._order_main[method_name].remove(callback.ORDER)

    def remove(self, callbacks:Callback | Iterable[Callback]):
        """
        Remove the given callbacks from the list of callbacks.

        Parameters:
            callbacks (list[Callback]): A list of callbacks to be removed.
        """
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        [self._remove(b) for b in callbacks] # pylint:disable=W0106 #type:ignore
        [b.exit(self) for b in sorted(callbacks, key = lambda b: b.ORDER) if hasattr(b, 'exit')]  # pylint:disable=W0106 #type:ignore

    def remove_by_name(self, callbacks: str | Iterable[str]):
        """
        Remove callbacks by their names.

        Args:
            callbacks (list[str]): A list of callback names to be removed.

        Returns:
            list: A list of callbacks that were removed (in case you want to add them back)
        """
        if isinstance(callbacks, str): callbacks = [callbacks]
        callbacks_to_remove = [b for b in self._callbacks if b.__class__.__name__ in callbacks]
        if len(callbacks_to_remove)>0:self.remove(callbacks_to_remove)
        return callbacks_to_remove

    def _sort_fs(self):
        for method, fs in self._fs_before.items():
            z = list(zip(fs, self._order_before[method]))
            z.sort(key = lambda x: x[1])
            self._fs_before[method] = [x[0] for x in z]
            self._order_before[method] = [x[1] for x in z]
        for method, fs in self._fs_after.items():
            z = list(zip(fs, self._order_after[method]))
            z.sort(key = lambda x: x[1])
            self._fs_after[method] = [x[0] for x in z]
            self._order_after[method] = [x[1] for x in z]
        for method, fs in self._fs_main.items():
            z = list(zip(fs, self._order_main[method]))
            z.sort(key = lambda x: x[1])
            self._fs_main[method] = [x[0] for x in z]
            self._order_main[method] = [x[1] for x in z]

    @property
    def callbacks(self): return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: Optional[Callback | Iterable[Callback]]):
        self.remove(self._callbacks)
        if callbacks is not None: self.add(callbacks)

    def _run_main(self, name:str, *args, **kwargs):
        returned = []
        for method in self._fs_main[name]:
            returned.append(method(self, *args, **kwargs))
        return returned

    def _run_before(self, name:str, *args, **kwargs):
        if name in self._fs_before:
            #print('before', name, self._fs_before[name])
            for method in self._fs_before[name]: method(self, *args, **kwargs)

    def _run_after(self, name:str, *args, **kwargs):
        if name in self._fs_after:
            for method in self._fs_after[name]:
                method(self, *args, **kwargs)

    def event(self, name:str):
        try:
            if name.startswith('before_'): self._run_before(name)
            elif name.startswith('after_'): self._run_after(name)
            elif name in self._fs_main: return self._run_main(name)
        except Cancel as e:
            if str(e) != name: raise e
        except Exception as e:
            raise e

    @contextmanager
    def context(self, name:str, extra:Optional[Callback | Iterable[Callback]] = None, without:Optional[str | Iterable[str]] = None):
        """
        Context manager. Runs all `before_{name}` methods on enter and `after_{name}` on exit.

        Args:
            name (str): The name of the context.
            xtra (Any, optional): Run context with extra callbacks.
            ignore (Any, optional): Run context without callbacks, must be a string or list of strings - class names of callbacks that will be ignored
        """
        #print('context', name, extra, without)
        # if name not in self._fs_main:
        #     logging.warning(f'Context: `{name}` not found in {self.__class__.__name__} or any of the callbacks')
        #     print(self._fs_main)
        if extra: self.add(extra)
        if without: removed = self.remove_by_name(without)
        try:
            #if name in self._fs_before: print('before', self._fs_before[name])
            self._run_before(f"before_{name}")
            yield
            #if name in self._fs_after: print('after',self._fs_after[name])
            self._run_after(f"after_{name}")
        #except AttributeError: raise AttributeError(f'`{name}` not found in {self.__class__.__name__} or any of the callbacks')
        except Cancel as e:
            if str(e) != name: raise e
        except Exception as e:
            raise e
        finally:
            if extra: self.remove(extra)
            if without and len(removed)>0: self.add(removed) # type: ignore


    def __getattr__(self, attr: str) -> Any:
        if attr not in self._fs_main: raise AttributeError(f'`{attr}` not found in {self.__class__.__name__} or any of the callbacks')
        return partial(self._run_main, attr)


    @contextmanager
    def extra(self, callbacks: Callback | Iterable[Callback]):
        """
        Context manager. Adds callbacks on enter and removes them on exit.

        Parameters:
            callbacks (list[Callback]): The list of callbacks to be added
        """
        self.add(callbacks)
        yield
        self.remove(callbacks)

    @contextmanager
    def without(self, callbacks: str | Iterable[str]):
        """
        Context manager. Removes specified callbacks on enter and adds them back on exit. Callbacks must be strings - class names.

        Args:
            callbacks (list[str]): The list of class names of callbacks to be removed.
        """
        removed = self.remove_by_name(callbacks)
        yield
        self.add(removed)





def callbacks_to_yaml():
    import yaml
    callbacks_classes = subclasses_recursive(Callback)
    callbacks_yaml = {}
    methods_yaml = {}
    for callback_cls in callbacks_classes:
        clsname = type_str(callback_cls)
        callback_methods = [m for m in dir(callback_cls) if isinstance(getattr(callback_cls, m), FunctionType)]
        callbacks_yaml[clsname] = dict()
        callbacks_yaml[clsname]['ORDER'] = callback_cls.ORDER
        callbacks_yaml[clsname]['description'] = callback_cls.__doc__
        callbacks_yaml[clsname]['methods main'] = [i for i in callback_methods if not i.startswith(('_', 'before_', 'after_'))]
        callbacks_yaml[clsname]['mehods before'] = [i for i in callback_methods if i.startswith('before_')]
        callbacks_yaml[clsname]['mehods after'] = [i for i in callback_methods if i.startswith('after_')]
        for i in callback_methods:
            if not i.startswith(('before_', 'after_', '_')):
                iname = i
                if i not in methods_yaml: methods_yaml[i] = {'main':{clsname:callback_cls.ORDER}, 'before':{}, 'after':{}}
                else: methods_yaml[i]['main'][clsname] = callback_cls.ORDER

            elif i.startswith('before_'):
                iname = i.replace('before_', '')
                if iname not in methods_yaml: methods_yaml[iname] = {'main':{}, 'before':{clsname:callback_cls.ORDER}, 'after':{}}
                else: methods_yaml[iname]['before'][clsname] = callback_cls.ORDER

            elif i.startswith('after_'):
                iname = i.replace('after_', '')
                if iname not in methods_yaml: methods_yaml[iname] = {'main':{}, 'before':{}, 'after':{clsname:callback_cls.ORDER}}
                else: methods_yaml[iname]['before'][clsname] = callback_cls.ORDER

            else:continue
            methods_yaml[iname]['before'] = {k:v for k,v in sorted(methods_yaml[iname]['before'].items(), key=lambda x: x[1])}
            methods_yaml[iname]['after'] = {k:v for k,v in sorted(methods_yaml[iname]['before'].items(), key=lambda x: x[1])}
            methods_yaml[iname]['main'] = {k:v for k,v in sorted(methods_yaml[iname]['main'].items(), key=lambda x: x[1])}

    callbacks_yaml = {k:v for k, v in sorted(list(callbacks_yaml.items()), key=lambda x: x[1]['ORDER'])}
    methods_yaml = {k:v for k, v in sorted(list(methods_yaml.items()), key=lambda x: x[0])}
    class Dumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, *args, **kwargs): # pylint:disable=W1113
            return super().increase_indent(flow=flow, indentless=False)
    with open('Callbacks.yaml', 'w', encoding='utf8') as f:
        for k,v in callbacks_yaml.items():
            f.write(f"# {''.join([i.title() if i.islower() else ' '+i for i in k.split('.')[-1] if i!='_'])}\n")
            f.write(yaml.dump({k:v}, sort_keys=False, Dumper=Dumper))
            f.write("\n")

    with open('Callback methods.yaml', 'w', encoding='utf8') as f:
        for k,v in methods_yaml.items():
            f.write(f"# {k.title()}\n")
            f.write(yaml.dump({k:v}, sort_keys=False, Dumper=Dumper))
            f.write("\n")
