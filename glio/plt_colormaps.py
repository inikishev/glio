import matplotlib, matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

__cmap_RYG_c = ["darkred","red","yellow", "green","darkgreen"]
__cmap_RYG_v = [0,.15,.5,.9,1.]
__cmap_RYG_l = list(zip(__cmap_RYG_v,__cmap_RYG_c))
cmap_RYG=LinearSegmentedColormap.from_list('RYG',__cmap_RYG_l, N=256)
cmap_GYR=cmap_RYG.reversed("GYR")
matplotlib.colormaps.register(cmap_RYG)
matplotlib.colormaps.register(cmap_GYR)