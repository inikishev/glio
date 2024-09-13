from collections.abc import Sequence
from typing import Any
import cv2
import numpy as np

BLUE = np.asanyarray([0,0,255])
GREEN = np.asanyarray([0,255,0])
RED = np.asanyarray([255,0,0])

def render_scatter_video(outfile: str, bg, x, y, c = RED, s = 4, fps = 60, ppf = 1):
    """Render a list of x and y coords as a video. Coords must be integers.

    :param outfile: Path to the output video file, which must end in .mp4
    :param bg: Background image. Can be black and white, channels first or channels last. Or use `np.zeroes` with whatever shape you need.
    :param x: A sequence of x coordinates of pixels on the `bg` image, starting from left.
    :param y: A sequence of y coordinates of pixels on the `bg` image, starting from top.
    :param c: A sequence of colors, or a single color. Each color must be a length 3 RGB sequence with values from 0 to 255, defaults to (255, 0, 0)
    :param s: A sequence of marker sizes or a single marker size value. Defaults to 4
    :param fps: frames per second, defaults to 60.
    :param ppf: points per frame, defaults to 1.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    c = np.asanyarray(c, dtype=np.uint8)
    if c.ndim == 1:
        c = c[np.newaxis, :].repeat(len(x), axis = 0)
    # bgr to rgb
    c = c[:, ::-1]

    s = np.asanyarray(s)
    if s.ndim < 1:
        s = s[np.newaxis].repeat(len(x), axis = 0)

    if bg is None:
        size = [int(y.max() - y.min()), int(x.max() - x.min())]
        x0 = int(x.min())
        y0 = int(y.min())
        bg = np.zeros(shape = size)

    else:
        bg = np.asanyarray(bg)
        size = bg.shape[:2]
        x0 = 0
        y0 = 0

    if bg.ndim == 2:
        bg = bg[:,:,np.newaxis].repeat(3, 2)
    elif bg.ndim == 3 and bg.shape[0] < bg.shape[2]: bg = np.moveaxis(bg, 0, 2)

    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))

    mask = np.zeros(size, dtype=bool)
    for i, (X, Y, C, S) in enumerate(zip(x, y, c, s)):
        X = int(X) - x0
        Y = int(Y) - y0
        S = int(S)
        C = np.asanyarray(C, dtype=np.uint8)

        mask[Y - S : Y + S, X - S : X + S] = True
        if i % ppf == 0:
            frame = bg.copy()
            frame[mask] = np.asanyarray([0, 0, 255])
            out.write(frame)
    out.release()