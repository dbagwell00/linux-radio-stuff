#!/usr/bin/env python3
"""Replay the last N minutes of transcripts through the incident tracker; print, don't send.

Usage: incident_dryrun.py [minutes] [transcriber.json]   (run from the transcriber's directory)
"""
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "radio"))
import edacs_telegram as et  # noqa: E402
import incidents  # noqa: E402

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 20
cfg = dict(et.CFG)
if len(sys.argv) > 2:
    cfg.update(json.load(open(sys.argv[2])))
cfg["incident_state_file"] = os.path.join(tempfile.mkdtemp(prefix="incident_dryrun_"), "state.json")
cfg["incident_audio"] = False
env = et.load_env(cfg["env_file"])
tz = ZoneInfo(cfg["timezone"])


class FakeTG:
    n = 0

    def _p(self, kind, text):
        print(f"--- {kind} ---\n" + re.sub(r"</?[bi]>", "", text) + "\n")

    def send(self, text, silent=False):
        self._p("SEND (silent)" if silent else "SEND", text)
        return True

    def send_get_id(self, text, silent=False):
        FakeTG.n += 1
        self._p(f"NEW MESSAGE #{FakeTG.n}", text)
        return FakeTG.n

    def edit(self, mid, text):
        self._p(f"EDIT #{mid}", text)
        return True


day = datetime.now(tz).strftime("%Y-%m-%d")
recs = [json.loads(l) for l in open(f"{cfg['transcript_dir']}/{day}.jsonl")]
t_end = time.time()
recs = sorted((r for r in recs if r["start"] >= t_end - minutes * 60), key=lambda r: r["start"])
print(f"replaying {len(recs)} transcripts in {cfg['incident_batch_s']}s batches\n")
tr = incidents.IncidentTracker(cfg, env, FakeTG(), tz, lambda rs: "(plain digest)")
batch = cfg["incident_batch_s"]
t = recs[0]["start"] if recs else t_end
while t < t_end:
    for r in recs:
        if t <= r["start"] < t + batch:
            tr.add(r)
    with tr.lock:
        pending = list(tr.pending)
        tr.pending = []
    if pending:
        tr._thread(pending)
    tr._close_idle(t + batch)   # simulated clock
    t += batch
print("=== still open ===")
for inc in tr.open.values():
    print(f"{inc.id} sev{inc.severity} {inc.type}: {inc.title} | {inc.location} | {inc.count} tx")
print("=== minor (would go to hourly round-up) ===")
for inc in tr.minor_done + [i for i in tr.open.values() if i.severity < 2]:
    print(f"  {inc.title} — {inc.location}")
