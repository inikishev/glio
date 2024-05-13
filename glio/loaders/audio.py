import numpy as np
import pedalboard

def audioread(path) -> tuple[np.ndarray, int]:
    with pedalboard.io.AudioFile(path, 'r') as f: # pylint:disable=E1129 # type:ignore
        audio = f.read(f.frames)
        sr = f.samplerate
    return audio, sr
