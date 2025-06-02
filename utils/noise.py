"""Funkcje redukcji szumów."""

import noisereduce as nr
import numpy as np

def reduce_noise(audio: np.ndarray, sr: int = 16_000):
    """Zwraca odszumione audio."""
    return nr.reduce_noise(y=audio, sr=sr)
