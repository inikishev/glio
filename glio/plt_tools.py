import matplotlib, matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
from .progress_bar import PBar

def animate_path(path, out, label=None, fps = 10, dpi=100, progress=True):
    from .plot import LiveFigure
    from matplotlib.animation import FFMpegWriter

    writer = FFMpegWriter(fps=fps)
    lfig = LiveFigure()
    if label is None: label = 'path'
    lfig.add_path10d(label, (0,0))
    lfig.draw(update=False)
    if progress: r = PBar(range(1, len(path)), 50, 1)
    else: r = range(1, len(path))

    with writer.saving(lfig.fig, out, dpi): # type:ignore
        for i in r:
            lfig.update(label, path[:i])
            writer.grab_frame()
    lfig.close()

def animate_path_gif(path):
    import gif
    from .plot import Figure

    @gif.frame
    def path_animate(i, path=path):
        print(i, end='\r')
        fig = Figure()
        path = path[:i]
        fig.add().path10d(path, "param path L1")
        fig.create()
    frames = [path_animate(i) for i in range(1, len(path))]
    gif.save(frames, 'example.gif', duration=len(path)) # type:ignore


def scatter_heatmap_interpolate(x, y, vals, cmap=None, grid = 500, levels=500):
    from scipy.interpolate import griddata
    import numpy as np

    X, Y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), grid),
        np.linspace(np.min(y), np.max(y), grid)
    )

    interpolated_vals = griddata((x, y), vals, (X, Y), method='cubic')

    plt.contourf(X, Y, interpolated_vals, levels=levels, cmap=cmap)
    plt.show()