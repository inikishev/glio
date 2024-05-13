# Автор - Никишев Иван Олегович группа 224-31

from functools import partial
#import logging
from contextlib import contextmanager
from types import MethodType, FunctionType

def context(inner):
    """Decorator that adds context to a Callback method"""
    def outer(*args, **kwargs):
        with args[1].context(inner.__name__):
            return inner(*args, **kwargs)
    return outer

class Callback:
    order = 0
    default = False
    description = ""

RESERVED = ('enter', 'exit', 'callbacks', 'context', 'add', 'remove', 'remove_by_name', 'extra', 'without', 'event')
class Cancel(Exception): pass

class CallbackModel:
    """
    A reasonaly fast implementation of Jeremy Howard course style callbacks model, except this time the whole thing is made of callbacks. So I decided it makes more sense to call them callbacks.
    
    Whenever a callback is added to the model, `enter` method is called in it. That includes creating a CallbackModel object, using `add` method or setting `callbacks` variable.
    
    Whenever a callback is removed from the model, `exit` method is called in it. This includes using `remove` method or setting `callbacks` variable.
    """
    def __init__(self, callbacks, default_callbacks, *args, **kwargs):
        """
        Initializes the object with the given callbacks.
        Parameters:
            callbacks (list or Callback): callbacks to add to the object.
            default_callbacks (list or Callback): callbacks with a single method, will be added only if that method isn't in any of the `callbacks`.
        """
        self._fs_main = {}
        self._fs_before = {}
        self._fs_after = {}
        self._order_main = {}
        self._order_before = {}
        self._order_after = {}
        self._callbacks = []
        
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        self.add(callbacks, enter = False)
        try: self.init(*args, **kwargs)
        except AttributeError: pass
        
        if isinstance(default_callbacks, Callback): default_callbacks = [default_callbacks]
        callbacks_to_add = []
        if default_callbacks is not None:
            for callback in default_callbacks:
                method_name = [method_name for method_name in dir(callback) if 
                               isinstance(getattr(callback, method_name), MethodType) 
                               and not (method_name.startswith('_') or method_name in RESERVED)]
                if len(method_name) != 1: 
                    if not callback.default: raise ValueError(f'default callbacks must have only one main method or default=True; {callback} has {method_name} and default = {callback.default}')
                    else: callbacks_to_add.append(callback)
                else:
                    method_name = method_name[0]
                    if method_name not in self._fs_main:
                        callbacks_to_add.append(callback)
        if len(callbacks_to_add) > 0: self.add(callbacks_to_add)
        
        [b.enter(self) for b in sorted(callbacks, key = lambda b: b.order) if hasattr(b, 'enter')]
                    
    def _add(self, callback: Callback):
        self._callbacks.append(callback)
        for method_name in dir(callback):
            if not (method_name.startswith('_') or method_name in RESERVED):
                method = getattr(callback, method_name)
                if isinstance(method, MethodType):
                    # we do this to maybe get a lil performance?
                        if method_name.startswith('before_'):
                            name = method_name.replace('before_', '')
                            if name in self._fs_before:
                                self._fs_before[name].append(method)
                                self._order_before[name].append(callback.order)
                            else:
                                self._fs_before[name] = [method]
                                self._order_before[name] = [callback.order]
                        elif method_name.startswith('after_'):
                            name = method_name.replace('after_', '')
                            if name in self._fs_after:
                                self._fs_after[name].append(method)
                                self._order_after[name].append(callback.order)
                            else:
                                self._fs_after[name] = [method]
                                self._order_after[name] = [callback.order]
                        else:
                            if method_name in self._fs_main:
                                self._fs_main[method_name].append(method)
                                self._order_main[method_name].append(callback.order)
                            else:
                                self._fs_main[method_name] = [method]
                                self._order_main[method_name] = [callback.order]
                                
    def add(self, callbacks:list[Callback], enter = True):
        """
        Add callbacks to the current object.

        Parameters:
            callbacks (list[Callback]): The callbacks to be added.
            enter (bool): Whether to enter the callbacks or not. Defaults to True.
        """
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        [self._add(b) for b in callbacks]
        self._sort_fs()
        if enter: [b.enter(self) for b in sorted(callbacks, key = lambda b: b.order) if hasattr(b, 'enter')]
    
    def _remove(self, callback: Callback):
        self._callbacks.remove(callback)
        for method_name in dir(callback):
            if not (method_name.startswith('_') or method_name in RESERVED):
                method = getattr(callback, method_name)
                if isinstance(method, MethodType):
                    if method_name.startswith('before_'):
                        self._fs_before[method_name.replace('before_', '')].remove(method)
                        self._order_before[method_name.replace('before_', '')].remove(callback.order)
                    elif method_name.startswith('after_'):
                        self._fs_after[method_name.replace('after_', '')].remove(method)
                        self._order_after[method_name.replace('after_', '')].remove(callback.order)
                    else:
                        self._fs_main[method_name].remove(method)
                        self._order_main[method_name].remove(callback.order)
                        
    def remove(self, callbacks:list[Callback]):
        """
        Remove the given callbacks from the list of callbacks.

        Parameters:
            callbacks (list[Callback]): A list of callbacks to be removed.
        """
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        [self._remove(b) for b in callbacks]
        [b.exit(self) for b in sorted(callbacks, key = lambda b: b.order) if hasattr(b, 'exit')]
                    
    def remove_by_name(self, callbacks: list[str]):
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
    def callbacks(self, callbacks):
        self.remove(self._callbacks)
        self.add(callbacks)

    def _run_main(self, name:str, *args, **kwargs):
        for method in self._fs_main[name]:
            method(self, *args, **kwargs)
    def _run_before(self, name:str, *args, **kwargs):
        if name in self._fs_before:
            #print('before', name, self._fs_before[name])
            for method in self._fs_before[name]: method(self, *args, **kwargs)
    def _run_after(self, name:str, *args, **kwargs):
        if name in self._fs_after:
            #print('after', name, self._fs_after[name])
            for method in self._fs_after[name]: method(self, *args, **kwargs)
    
    def event(self, name):
        if name.startswith('before_'): self._run_before(name)
        elif name.startswith('after_'): self._run_after(name)
        elif name in self._fs_main: self._run_main(name)
        
    @contextmanager
    def context(self, name:str, extra:list[Callback] = None, without:list[str] = None):
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
            self._run_before(name)
            yield
            #if name in self._fs_after: print('after',self._fs_after[name])
            self._run_after(name)
        #except AttributeError: raise AttributeError(f'`{name}` not found in {self.__class__.__name__} or any of the callbacks')
        except Cancel as e:
            if str(e) != name: raise e
        except KeyboardInterrupt as e:
            raise(Cancel('fit'))
        except Exception as e:
            raise e
        finally: 
            if extra: self.remove(extra)
            if without and len(removed)>0: self.add(removed)


    def __getattr__(self, attr):
        if attr not in self._fs_main: raise AttributeError(f'`{attr}` not found in {self.__class__.__name__} or any of the callbacks')
        return partial(self._run_main, attr)
        
        
    @contextmanager
    def extra(self, callbacks: list[Callback]):
        """
        Context manager. Adds callbacks on enter and removes them on exit.

        Parameters:
            callbacks (list[Callback]): The list of callbacks to be added
        """
        self.add(callbacks)
        yield
        self.remove(callbacks)
        
    @contextmanager
    def without(self, callbacks: list[str]):
        """
        Context manager. Removes specified callbacks on enter and adds them back on exit. Callbacks must be strings - class names.

        Args:
            callbacks (list[str]): The list of class names of callbacks to be removed.
        """
        removed = self.remove_by_name(callbacks)
        yield
        self.add(removed)

        
        

from ..python_tools import subclasses_recursive, type_str
def callbacks_to_yaml():
    import yaml
    callbacks_classes = subclasses_recursive(Callback)
    callbacks_yaml = {}
    methods_yaml = {}
    for callback_cls in callbacks_classes:
        clsname = type_str(callback_cls)
        callback_methods = [m for m in dir(callback_cls) if isinstance(getattr(callback_cls, m), FunctionType)]
        callbacks_yaml[clsname] = dict()
        callbacks_yaml[clsname]['order'] = callback_cls.order
        callbacks_yaml[clsname]['description'] = callback_cls.description
        callbacks_yaml[clsname]['methods main'] = [i for i in callback_methods if not i.startswith(('_', 'before_', 'after_'))]
        callbacks_yaml[clsname]['mehods before'] = [i for i in callback_methods if i.startswith('before_')]
        callbacks_yaml[clsname]['mehods after'] = [i for i in callback_methods if i.startswith('after_')]
        for i in callback_methods:
            if not i.startswith(('before_', 'after_', '_')):
                iname = i
                if i not in methods_yaml: methods_yaml[i] = {'main':{clsname:callback_cls.order}, 'before':{}, 'after':{}}
                else: methods_yaml[i]['main'][clsname] = callback_cls.order
                
            elif i.startswith('before_'): 
                iname = i.replace('before_', '')
                if iname not in methods_yaml: methods_yaml[iname] = {'main':{}, 'before':{clsname:callback_cls.order}, 'after':{}}
                else: methods_yaml[iname]['before'][clsname] = callback_cls.order

            elif i.startswith('after_'): 
                iname = i.replace('after_', '')
                if iname not in methods_yaml: methods_yaml[iname] = {'main':{}, 'before':{}, 'after':{clsname:callback_cls.order}}
                else: methods_yaml[iname]['before'][clsname] = callback_cls.order
            
            else:continue
            methods_yaml[iname]['before'] = {k:v for k,v in sorted(methods_yaml[iname]['before'].items(), key=lambda x: x[1])}
            methods_yaml[iname]['after'] = {k:v for k,v in sorted(methods_yaml[iname]['before'].items(), key=lambda x: x[1])}
            methods_yaml[iname]['main'] = {k:v for k,v in sorted(methods_yaml[iname]['main'].items(), key=lambda x: x[1])}
    
    callbacks_yaml = {k:v for k, v in sorted(list(callbacks_yaml.items()), key=lambda x: x[1]['order'])}
    methods_yaml = {k:v for k, v in sorted(list(methods_yaml.items()), key=lambda x: x[0])}
    class Dumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, *args, **kwargs):
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