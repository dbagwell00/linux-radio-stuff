"""Shared audio helpers (numpy only, so the transcriber venv doesn't need scipy)."""
import numpy as np


def level_speech(pcm, fs=16000, target_dbfs=-16.0, max_gain_db=30.0, knee=0.6):
    """Speech-level normalisation for int16 PCM.

    Gain comes from the RMS of the active (speech) 20 ms frames rather than the
    single loudest sample, so one squelch click can't cap the gain. A soft
    limiter above `knee` (about -4.4 dBFS) keeps clicks and pops from clipping.
    """
    x = pcm.astype(np.float32) / 32768.0
    if len(x) < fs // 10:
        return pcm
    n = fs // 50                                            # 20 ms frames
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    ref = np.percentile(rms, 90)
    active = rms[rms > ref * 0.1]                           # within 20 dB of the loud part
    level = np.sqrt((active ** 2).mean()) if len(active) else ref
    g = 10 ** ((target_dbfs - 20 * np.log10(level + 1e-9)) / 20)
    g = min(g, 10 ** (max_gain_db / 20))
    y = x * g
    a = np.abs(y)
    over = a > knee
    y[over] = np.sign(y[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return np.clip(y * 32767, -32767, 32767).astype(np.int16)
