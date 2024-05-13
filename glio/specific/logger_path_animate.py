import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.style as mplstyle

from ..plot import LiveFigure
from ..progress_bar import PBar
from ..logger import Logger

def logger_path_animate(logger: Logger, out, figsize=(8, 4.5), fps=30, tail=100, dpi=150):
    #mplstyle.use('fast')
    writer = FFMpegWriter(fps=fps)
    lfig = LiveFigure()
    lfig.add_plot('train loss', (0,0), axlabelsize=5)
    lfig.add_plot('test loss', (0,0), axlabelsize=5)
    lfig.add_plot('train accuracy', (1,0), axlabelsize=5)
    lfig.add_plot('test accuracy', (1,0), axlabelsize=5)
    lfig.add_path10d('param path L1', (0,1), axlabelsize=5)
    lfig.add_path10d('param path L1 - last 100', (1,1), axlabelsize=5)
    lfig.draw(update=False, figsize=figsize, layout=None)
    plt.tight_layout()
    r = PBar(range(1, list(logger["train loss"].keys())[-1]), 50, 1)
    #r = PBar(range(1, 10), 50, 1)

    with writer.saving(lfig.fig, out, dpi): # type:ignore
        for i in r:
            hist_logger = logger.range(0, i, metrics=("train loss", "test loss", "train accuracy", "test accuracy", "param path L1"))
            trainloss = hist_logger("train loss")
            testloss = hist_logger("test loss")
            trainacc = hist_logger("train accuracy")
            testacc = hist_logger("test accuracy")
            path = hist_logger.toarray('param path L1')
            if len(path) > 0:
                path2 = path[-tail:]
                lfig.update('train loss', trainloss)
                lfig.update('test loss', testloss)
                lfig.update('train accuracy', trainacc)
                lfig.update('test accuracy', testacc)
                lfig.update('param path L1', path)
                lfig.update('param path L1 - last 100', path2)
                writer.grab_frame()
    lfig.close()