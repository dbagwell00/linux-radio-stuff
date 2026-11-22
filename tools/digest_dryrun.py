#!/usr/bin/env python3
"""Print (don't send) a digest for the last N minutes of saved transcripts.

Usage: digest_dryrun.py [minutes] [transcriber.json]   (run from the transcriber's directory)
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radio"))
import edacs_telegram as et  # noqa: E402

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = dict(et.CFG)
if len(sys.argv) > 2:
    cfg.update(json.load(open(sys.argv[2])))
env = et.load_env(cfg["env_file"])
tz = ZoneInfo(cfg["timezone"])
t1 = time.time()
t0 = t1 - minutes * 60
day = datetime.now(tz).strftime("%Y-%m-%d")
recs = [json.loads(l) for l in open(f"{cfg['transcript_dir']}/{day}.jsonl")]
recs = [r for r in recs if t0 <= r["start"] < t1]
print(f"{len(recs)} transcripts in window\n")
msg = et.build_digest(cfg, env, recs, t0, t1, tz)
print(re.sub(r"</?[bi]>", "", msg))
