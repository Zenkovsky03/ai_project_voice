"""Nagrywanie audio i zapisywanie WAV."""
import queue
import sounddevice as sd
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16_000


def record(seconds: int = 5, out_path: Path | None = None):
    """Nagrywa `seconds` sekund dźwięku i opcjonalnie zapisuje do pliku.

    Zwraca tablicę float32 (‑1..1) ⚠ mono.
    """
    q = queue.Queue()

    def _callback(indata, frames, time, status):  # noqa: D401
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=_callback):
        print(f"▶️ Nagrywam {seconds}s …")
        sd.sleep(int(seconds * 1000))

    audio = b"".join(list(q.queue))
    import numpy as np

    waveform = np.frombuffer(audio, dtype='float32')

    if out_path:
        sf.write(out_path, waveform, SAMPLE_RATE)
    return waveform