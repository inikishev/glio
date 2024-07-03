from collections.abc import Sequence
from typing import Any,Optional
from ..design.EventModel import CBMethod
from ..python_tools import type_str
from .Learner import Learner

from ..plot import Figure, LiveFigure
from ._liveplot_shortcuts import _SHORTCUTS

__all__ = [
    "LivePlotCB",
    "LivePlot2CB",
    "PlotSummaryCB",
]

class LivePlotCB(CBMethod):
    order = 95
    def __init__(  # pylint:disable = W0102
        self,
        step:int=16,
        plot_keys: dict[str, Sequence] | Sequence [str | dict[str, Sequence]]= ("combo",),
        img_keys: dict[str, Sequence] | Sequence [str | dict[str, Sequence]]= {},
        path_keys: dict[str, Sequence] | Sequence [str | dict[str, Sequence]]= {},
        figsize=(15, 7),
    ):
        if isinstance(plot_keys, Sequence):
            pk = {}
            for i in plot_keys:
                if isinstance(i, dict): pk.update(i)
                else: pk.update(_SHORTCUTS[i])
            plot_keys = pk
        if isinstance(img_keys, Sequence):
            pk = {}
            for i in img_keys:
                if isinstance(i, dict): pk.update(i)
                else: pk.update(_SHORTCUTS[i])
            img_keys = pk
        if isinstance(path_keys, Sequence):
            pk = {}
            for i in path_keys:
                if isinstance(i, dict): pk.update(i)
                else: pk.update(_SHORTCUTS[i])
            path_keys = pk

        self.step = step
        self.lfig = LiveFigure()

        for k,v in plot_keys.items():
            if len(v) == 3: kwargs = v[2]
            else: kwargs = {}
            self.lfig.add_plot(k, v[0:2], **kwargs)
        for k,v in img_keys.items():
            if len(v) == 3: kwargs = v[2]
            else: kwargs = {}
            self.lfig.add_image(k, v[0:2], **kwargs)
        for k,v in path_keys.items():
            if len(v) == 3: kwargs = v[2]
            else: kwargs = {}
            self.lfig.add_path10d(k, v[0:2], **kwargs)
        self.lfig.draw(figsize=figsize)

        self.plot_keys, self.img_keys, self.path_keys = plot_keys, img_keys, path_keys

    def exit(self, learner: "Learner"):
        self.lfig.close()

    def _plot(self, learner: Learner):
        for k in self.plot_keys:
            if k in learner.logger:
                self.lfig.update(k, learner.logger(k))
        for k in self.img_keys:
            if k in learner.logger:
                self.lfig.update(k, learner.logger.last(k))
        for k in self.path_keys:
            if k.replace("\\","") in learner.logger:
                self.lfig.update(k, learner.logger.toarray(k.replace("\\","")))
        self.lfig.draw()

    def after_train_batch(self, learner: Learner):
        if self.step is not None and learner.total_batch % self.step == 0: self._plot(learner)
    def after_fit(self, learner: Learner):
        self._plot(learner)
        self.lfig.close()



class LivePlot2CB(CBMethod):
    order = 95
    def __init__(  # pylint:disable = W0102
        self,
        fig:LiveFigure,
        keys:Sequence[str],
        step=16,
        figsize=(15, 7),
        **kwargs,
    ):
        self.lfig = fig
        self.keys = keys
        self.step = step
        self.lfig.draw(figsize=figsize**kwargs,)

    def exit(self, learner: "Learner"):
        self.lfig.close()

    def _plot(self, learner: Learner):
        for k in self.keys:
            if k.replace("\\","") in learner.logger:
                if k in learner.logger.get_keys_num(): self.lfig.update(k, learner.logger(k))
                elif "path" in k: self.lfig.update(k, learner.logger.toarray(k.replace("\\","")))
                else: self.lfig.update(k, learner.logger.last(k))
        self.lfig.draw()

    def after_train_batch(self, learner: Learner):
        if self.step and learner.cur_batch % self.step == 0: self._plot(learner)
    def after_fit(self, learner: Learner):
        self._plot(learner)
        self.lfig.close()



class PlotSummaryCB(CBMethod):
    def __init__(self, figsize=8, nrow = None, ncol=2, save=True, show=False, path = '.', save_ext='png'):
        self.figsize = figsize
        self.nrow, self.ncol = nrow, ncol
        self.save, self.show = save, show
        self.save_ext = save_ext
        self.path = path

    def after_fit(self, learner: Learner):
        fig = Figure()
        train_test = []
        sep = []
        for k in learner.logger.keys():
            if k in learner.logger.get_keys_num():
                if k.startswith("train "):
                    if k.replace("train ", "") not in train_test: train_test.append(k.replace("train ", ""))
                elif k.startswith("test "):
                    if k.replace("test ", "") not in train_test: train_test.append(k.replace("test ", ""))
            else: sep.append(k)

        for k in train_test:
            train = learner.logger(f"train {k}") if f"train {k}" in learner.logger else None
            test = learner.logger(f"test {k}") if f"test {k}" in learner.logger else None
            if train is None: fig.add().plot(*test, label=f"test {k}").style_chart(xlabel="batch", ylabel=k, title=learner.logger.stats_str(f"test {k}"))
            elif test is None: fig.add().plot(*train, label=f"train {k}",linewidth=1).style_chart(xlabel="batch", ylabel=k, title=learner.logger.stats_str(f"train {k}"))
            else: fig.add().plot(*train, label=f"train {k}",linewidth=0.7).plot(*test, label=f"test {k}").style_chart(xlabel="batch", ylabel=k, title=f'{learner.logger.stats_str(f"train {k}")}\n{learner.logger.stats_str(f"test {k}")}')

        for k in sep:
            if k in learner.logger.get_keys_num():
                fig.add().plot(*learner.logger(k), label=k, linewidth=1).style_chart(xlabel="batch", ylabel=k, title=learner.logger.stats_str(k))
            elif 'path' in k:
                fig.add().path10d(learner.logger.toarray(k), label=k).style_chart(title=k)
            else:
                fig.add().imshow(learner.logger.last(k), label=k).style_img(title=k)

        if self.show: fig.show(figsize=self.figsize, nrow=self.nrow, ncol=self.ncol)
        else: fig.create(figsize=self.figsize, nrow=self.nrow, ncol=self.ncol)
        if self.save:
            lowest_test_loss = f' testloss-{learner.logger.min("test loss"):.3f}' if "test loss" in learner.logger else ""
            fig.savefig(f"{self.path}/{learner.name}{lowest_test_loss}.{self.save_ext}")

        fig.close()