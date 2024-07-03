
def _ylim(lim = (0,1), zorder:int = 0):
    return {"ylim":lim,"zorder":zorder}

_SHORTCUTS = {
    "loss": {
        "train loss": (0, 0, {"color":"orange","zorder":3,"linewidth":0.5}),
        "test loss": (0, 0, {"color":"red","zorder":4}),
    },
    "loss0": {
        "train loss": (0, 0, _ylim((0, None), 3)),
        "test loss": (0, 0, _ylim((0, None), 4)),
    },
    "loss01": {
        "train loss": (0, 0, _ylim((0,1), 3)),
        "test loss": (0, 0, _ylim((0,1), 4)),
    },
    "accuracy": {
        "train accuracy": (0, 0, {"color":"blue","zorder":1,"linewidth":0.5}),
        "test accuracy": (0, 0, {"color":"purple","zorder":2}),
    },
    "grad-update cosine": {
        "grad-update cosine": (0, 0, {"color":"green","zorder":0}),
    },
    "combo01": {
        "train loss": (0, 0, _ylim((0,1), 3)),
        "test loss": (0, 0, _ylim((0,1), 4)),
        "train accuracy": (0, 0, _ylim((0,1), 1)),
        "test accuracy": (0, 0, _ylim((0,1), 2)),
        "grad-update cosine": (0, 0,_ylim((0,1), 0)),
    },
    "combo": {
        "train loss": (0, 0, _ylim((0,None), 3)),
        "test loss": (0, 0, _ylim((0,None), 4)),
        "train accuracy": (1, 0, _ylim((0,None), 1)),
        "test accuracy": (1, 0, _ylim((0,None), 2)),
        "grad-update cosine": (0, 0,_ylim((0,None), 0)),
    },
    "4plotsplot": {
        "train loss": (0, 0, {"color" : "orange", "zorder" : 3, "linewidth" : 0.5, "alpha" : 0.5, }),
        "test loss": (0, 0, {"color" : "red", "zorder" : 4}),
        "train accuracy": (1, 0, {"color" : "lightblue","zorder" : 1,"linewidth" : 0.5}),
        "test accuracy": (1, 0, {"color" : "blue", "zorder" : 2}),
    },
    "4plotsplot01": {
        "train loss": (0, 0, {"color" : "orange", "zorder" : 3, "linewidth" : 0.5, "ylim" : (0,1), "alpha" : 0.5, }),
        "test loss": (0, 0, {"color" : "red", "zorder" : 4, "ylim" : (0,1),}),
        "train accuracy": (1, 0, {"color" : "lightblue", "zorder" : 1, "linewidth" : 0.5, "ylim" : (0,1),}),
        "test accuracy": (1, 0, {"color" : "blue", "zorder" : 2, "ylim" : (0,1),}),
    },
    "4plotspath100": {
        "param path mean": (0, 1),
        "param path mean\\": (1, 1, {"lastn":100}),
    },
    "4plotspath250": {
        "param path mean": (0, 1),
        "param path mean\\": (1, 1, {"lastn":250}),
    },
    "4plotspath500": {
        "param path mean": (0, 1),
        "param path mean\\": (1, 1, {"lastn":500}),
    },
    "4plotspath1000": {
        "param path mean": (0, 1),
        "param path mean\\": (1, 1, {"lastn":1000}),
    },
    "10metrics": {
        "train f1": (1, 0, {"color" : "orange", "zorder" : 0, "linewidth" : 0.5}),
        "test f1": (1, 0, {"color" : "red", "zorder" : 1}),
        "train iou": (1, 0, {"color" : "lime", "zorder" : 0, "linewidth" : 0.5}),
        "test iou": (1, 0, {"color" : "green", "zorder" : 1}),
        "train average precision": (1, 0, {"color": "gray", "zorder" : 0, "linewidth" : 0.5}),
        "test average precision": (1, 0, {"color": "black", "zorder" : 1}),
        "train roc auc": (1, 0, {"color" : "pink", "zorder" : 0, "linewidth" : 0.5}),
        "test roc auc": (1, 0, {"color" : "violet", "zorder" : 1}),
    },
    "10metrics01": {
        "train dice": (1, 0, {"color" : "orange", "zorder" : 0, "linewidth" : 0.5, "ylim" : (0,1),}),
        "test dice": (1, 0, {"color" : "red", "zorder" : 1, "ylim" : (0,1),}),
        "train iou": (1, 0, {"color" : "lime", "zorder" : 0, "linewidth" : 0.5, "ylim" : (0,1),}),
        "test iou": (1, 0, {"color" : "green", "zorder" : 1, "ylim" : (0,1),}),
        "train average precision": (1, 0, {"color": "gray", "zorder" : 0, "linewidth" : 0.5, "ylim" : (0,1),}),
        "test average precision": (1, 0, {"color": "black", "zorder" : 1, "ylim" : (0,1),}),
        "train roc auc": (1, 0, {"color" : "pink", "zorder" : 0, "linewidth" : 0.5, "ylim" : (0,1),}),
        "test roc auc": (1, 0, {"color" : "violet", "zorder" : 1, "ylim" : (0,1),}),
    },
}
