from collections.abc import Callable

import numpy as np
import sounddevice as sd


def record(sample_rate: int, until: Callable[[], None]) -> np.ndarray:
    """Record mono float32 from the default input device for as long as `until()` blocks."""
    chunks: list[np.ndarray] = []
    # PortAudio lists devices once, at initialization, and keeps that list: a microphone connected since, or a
    # Windows session that was locked and unlocked, leaves it opening a device that is gone or silent. Listing again
    # costs a few milliseconds.
    sd._terminate()
    sd._initialize()
    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        callback=lambda data, frames, t, status: chunks.append(data[:, 0].copy()),
    ):
        until()
    # until() can return before the first audio callback (the stop edge was already queued), leaving no audio at all
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
