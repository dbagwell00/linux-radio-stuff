#!/usr/bin/env python3
"""Compare faster-whisper models on your own recordings: same clips, same settings.

Usage: whisper_bench.py transcripts/YYYY-MM-DD.jsonl MODEL [MODEL ...]
  e.g. whisper_bench.py transcripts/2026-01-31.jsonl small.en medium.en large-v3-turbo

Picks up to 32 random calls (>= 1.5 s, MP3 still on disk) from the transcript log,
prints the real-time factor per model, and writes the texts to whisper_bench.json
so you can compare quality side by side. Set WHISPER_PROMPT to test an initial prompt.
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
from faster_whisper import WhisperModel

if len(sys.argv) < 3:
    sys.exit(__doc__)
random.seed(7)
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs = [r for r in recs if r.get("mp3") and os.path.exists(r["mp3"]) and r["duration"] >= 1.5]
clips = random.sample(recs, min(32, len(recs)))
prompt = os.environ.get("WHISPER_PROMPT") or None
audio = []
for r in clips:
    x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "quiet", "-i", r["mp3"], "-f", "s16le", "-ac", "1",
                                      "-ar", "16000", "-"], capture_output=True).stdout, np.int16)
    audio.append(x.astype(np.float32) / 32768)
total = sum(len(a) for a in audio) / 16000
out = {"clips": [{"tg": r["tg"], "label": r.get("tg_label"), "dur": r["duration"]} for r in clips],
       "models": {}}
for name in sys.argv[2:]:
    t0 = time.time()
    m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=3)
    load = time.time() - t0
    texts, t1 = [], time.time()
    for a in audio:
        segs, _ = m.transcribe(a, language="en", beam_size=1, temperature=0.0, vad_filter=False,
                               condition_on_previous_text=False, initial_prompt=prompt)
        segs = [s for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2]
        texts.append(" ".join(s.text.strip() for s in segs).strip())
    took = time.time() - t1
    out["models"][name] = {"texts": texts, "secs": took, "audio_s": total, "load_s": load}
    print(f"{name}: {took:.0f}s for {total:.0f}s audio (RTF {took / total:.2f}), load {load:.0f}s",
          flush=True)
    del m
json.dump(out, open("whisper_bench.json", "w"), indent=1)
