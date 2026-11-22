#!/usr/bin/env python3
"""radiowatch — live terminal dashboard for the EDACS → Whisper → Telegram pipeline.

Keys: /  filter   esc  clear filter   space  pause scroll   i  incidents   q  quit

Usage: radiowatch.py [radiowatch.json]   (or set RADIOWATCH_CONFIG)
"""
import json
import os
import sys
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import zlib
from collections import Counter, deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Sparkline, Static

CONF = {
    "base_dir": ".",                 # the transcriber's working directory
    "output_dir": "recordings",      # recorder output (shown in the disk panel)
    "llm_models_url": "http://127.0.0.1:8000/v1/models",
    "services": ["edacs-rec", "edacs-local", "edacs-telegram"],
    "voice_lcns": [],                # trunked voice LCNs to show; [] = read from recorder.json
    "recorder_config": "recorder.json",
    "timezone": "UTC",
    "subtitle": "EDACS · Whisper · Telegram",
}
_cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RADIOWATCH_CONFIG")
if _cfg_path:
    with open(_cfg_path) as _f:
        CONF.update(json.load(_f))
BASE = os.path.abspath(CONF["base_dir"])
TDIR = f"{BASE}/transcripts"
SPOOL = f"{BASE}/spool"
STATE = f"{BASE}/incidents_state.json"
SHARE = os.path.join(BASE, CONF["output_dir"])
LLM_MODELS = CONF["llm_models_url"]
SERVICES = CONF["services"]
LCNS = CONF["voice_lcns"]
if not LCNS:
    try:
        with open(os.path.join(BASE, CONF["recorder_config"])) as _f:
            _rc = json.load(_f)
        _cc = _rc.get("control", {}).get("freq")
        LCNS = sorted(int(k) for k, v in _rc.get("lcns", {}).items() if v != _cc)
    except (OSError, ValueError):
        LCNS = []
TZ = ZoneInfo(CONF["timezone"])
HIST_S = 2 * 3600

KEYWORDS = re.compile(
    r"\b(shots? fired|shooting|shot|gunshots?|guns?|weapons?|stab\w*|pursuit|officer down|11-99|"
    r"structure fire|working fire|fully involved|explosion|hazmat|fatal|doa|active shooter|"
    r"amber alert|missing (?:child|juvenile)|cpr in progress|not breathing|barricaded|hostage)\b",
    re.I)
PALETTE = ["#5fafff", "#87d787", "#ffaf5f", "#d787ff", "#5fd7d7", "#ffd75f",
           "#ff87af", "#afaf87", "#87afff", "#d7af87", "#87ffaf", "#af87d7"]
SEV = {1: ("·", "dim"), 2: ("●", "yellow"), 3: ("‼", "bold red")}

RE_WHISPER = re.compile(r"TG (\d+) ([\d.]+)s:? (\(no speech\) )?\[([\d.]+)s\]")
RE_LLM = re.compile(r"threaded (\d+) transmissions -> (\d+) incidents \((\d+) open\) in ([\d.]+)s, (\S+)/(\S+) tokens")
RE_START = re.compile(r"call start\s+LCN (\d+)\s+TG (\d+)\s+(.*?)\s+LID (\S*)")
RE_END = re.compile(r"call end\s+LCN (\d+)\s")
RE_REPEAT = re.compile(r"((?:\S+[,.]?\s+){1,3}?)\1{4,}")
RE_RECSTATS = re.compile(r"stats: (\{.*\}), active")


def tg_color(tg):
    return PALETTE[zlib.crc32(str(tg).encode()) % len(PALETTE)]


def short_label(r):
    lab = (r.get("tg_label") or "").split(" - ")[0].strip()
    return lab or f"TG {r.get('tg')}"


def bar(frac, width, color):
    frac = max(0.0, min(1.0, frac))
    full, rem = divmod(frac * width, 1)
    s = "█" * int(full) + ("▏▎▍▌▋▊▉"[int(rem * 7) - 1] if rem * 7 >= 1 else "")
    t = Text(s, style=color)
    t.append("·" * (width - len(s)), style="grey23")
    return t


def human(n):
    for u in ("B", "K", "M", "G", "T", "P"):
        if abs(n) < 1024:
            return f"{n:.0f}{u}" if u in "BK" else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}E"


def ago(ts):
    if not ts:
        return "never"
    s = time.time() - ts
    return f"{s:.0f}s ago" if s < 120 else f"{s / 60:.0f}m ago" if s < 7200 else f"{s / 3600:.1f}h ago"


def pct(vals, p):
    if not vals:
        return 0.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


class Data:
    """Everything the background threads collect; the UI reads it under the lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recs = deque()           # transcript records, last HIST_S
        self.whisper = deque()        # (ts, audio_s, took_s, no_speech)
        self.llm = deque()            # (ts, n_tx, n_inc, n_open, secs, ptok, ctok)
        self.events = deque(maxlen=6)  # (ts, unit, msg) warnings/errors
        self.on_air = {}              # lcn -> (since, tg, label, lid)
        self.rec_stats = {}
        self.today = {"calls": 0, "audio": 0.0, "words": 0, "day": None}
        self.slow = {}                # disk, llm reachability, services, mp3 bytes

    def prune(self):
        cut = time.time() - HIST_S
        for q in (self.recs, self.whisper, self.llm):
            while q and (q[0]["start"] if isinstance(q[0], dict) else q[0][0]) < cut:
                q.popleft()


class Watch(App):
    TITLE = "radiowatch"
    SUB_TITLE = CONF["subtitle"]
    CSS = """
    Screen { background: #0b0e14; }
    #main { height: 1fr; }
    #feedbox { width: 1fr; border: round #3a4560; border-title-color: #87afff; }
    #feed { background: #0b0e14; scrollbar-size-vertical: 1; }
    #filter { display: none; dock: bottom; border: tall #ffaf5f; }
    #filter.show { display: block; }
    #side { width: 58; scrollbar-size-vertical: 1; }
    .panel { height: auto; border: round #3a4560; border-title-color: #87afff;
             border-subtitle-color: #6c7a96; padding: 0 1; }
    Sparkline { height: 4; margin: 0 0 0 0; }
    #spark_calls > .sparkline--max-color { color: #5fd7d7; }
    #spark_calls > .sparkline--min-color { color: #1f3a4a; }
    #spark_lat > .sparkline--max-color { color: #ffaf5f; }
    #spark_lat > .sparkline--min-color { color: #3a2a1a; }
    #spark_lat { height: 3; }
    #incidents { height: 9; border: round #3a4560; border-title-color: #87afff; }
    #incidents.hide { display: none; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "filter", "Filter"),
        Binding("escape", "clear_filter", "Clear", show=False),
        Binding("space", "pause", "Pause"),
        Binding("i", "incidents", "Incidents"),
        Binding("c", "clear", "Clear feed"),
    ]

    def __init__(self):
        super().__init__()
        self.d = Data()
        self.stop = threading.Event()
        self.filter = ""
        self.paused = False
        self.feed_buf = deque(maxlen=3000)
        self.cpu_prev = None
        self.cg_prev = {}

    # ---------- layout ----------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="feedbox"):
                yield RichLog(id="feed", wrap=True, markup=False, max_lines=3000, auto_scroll=True)
                yield Input(id="filter", placeholder="filter: talkgroup, unit or words (esc clears)")
            with VerticalScroll(id="side"):
                with Vertical(id="vol", classes="panel"):
                    yield Static(id="vol_head")
                    yield Sparkline([0], id="spark_calls", summary_function=max)
                    yield Static(id="vol_axis")
                yield Static(id="onair", classes="panel")
                with Vertical(id="wsp", classes="panel"):
                    yield Static(id="wsp_stats")
                    yield Sparkline([0], id="spark_lat", summary_function=max)
                yield Static(id="llm", classes="panel")
                yield Static(id="tgs", classes="panel")
                yield Static(id="sys", classes="panel")
        yield DataTable(id="incidents", cursor_type="none", zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        titles = {"feedbox": "live transcripts", "vol": "calls / minute · last hour",
                  "onair": "on air", "wsp": "whisper", "llm": "incident LLM",
                  "tgs": "busiest talkgroups · last hour", "sys": socket.gethostname(),
                  "incidents": "open incidents"}
        for wid, t in titles.items():
            self.query_one(f"#{wid}").border_title = t
        tbl = self.query_one("#incidents", DataTable)
        tbl.add_columns("", "id", "incident", "location", "tx", "last", "channels")
        self.load_history()
        for target in (self.tail_transcripts, self.follow_journal, self.slow_poller):
            threading.Thread(target=target, daemon=True).start()
        self.set_interval(1.0, self.refresh_panels)
        self.set_interval(5.0, self.refresh_incidents)
        self.refresh_panels()
        self.refresh_incidents()

    # ---------- data sources ----------
    def _ingest(self, r):
        d = self.d
        day = datetime.fromtimestamp(r["start"], TZ).strftime("%Y-%m-%d")
        with d.lock:
            if r["start"] >= time.time() - HIST_S:
                d.recs.append(r)
            if day == datetime.now(TZ).strftime("%Y-%m-%d"):
                if d.today["day"] != day:
                    d.today = {"calls": 0, "audio": 0.0, "words": 0, "day": day}
                d.today["calls"] += 1
                d.today["audio"] += r.get("duration", 0)
                d.today["words"] += len(r.get("text", "").split())

    def load_history(self):
        now = datetime.now(TZ)
        paths = [f"{TDIR}/{(now - timedelta(days=1)):%Y-%m-%d}.jsonl", f"{TDIR}/{now:%Y-%m-%d}.jsonl"]
        recent = []
        for p in paths:
            try:
                with open(p) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        self._ingest(r)
                        recent.append(r)
            except FileNotFoundError:
                pass
        self.today_path = paths[1]
        self.today_pos = os.path.getsize(paths[1]) if os.path.exists(paths[1]) else 0
        for r in recent[-300:]:
            self.show(r)

    def tail_transcripts(self):
        path, f, buf = self.today_path, None, ""
        if os.path.exists(path):
            f = open(path)
            f.seek(self.today_pos)
        while not self.stop.is_set():
            chunk = f.read() if f else ""
            if chunk:
                buf += chunk
                *lines, buf = buf.split("\n")
                for line in lines:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    self._ingest(r)
                    self.call_from_thread(self.show, r)
                continue
            today = f"{TDIR}/{datetime.now(TZ):%Y-%m-%d}.jsonl"
            if today != path and os.path.exists(today):
                if f:
                    f.close()
                path, f, buf = today, open(today), ""
                continue
            if f is None and os.path.exists(path):
                f = open(path)
            time.sleep(0.4)

    def follow_journal(self):
        cmd = ["journalctl", "-f", "-o", "json", "--since", f"-{HIST_S}s", "--no-pager",
               "--output-fields=MESSAGE,_SYSTEMD_UNIT,PRIORITY"]
        for s in SERVICES:
            cmd += ["-u", s]
        while not self.stop.is_set():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    try:
                        self._journal(json.loads(line))
                    except Exception:
                        pass
            except Exception as e:
                with self.d.lock:
                    self.d.events.append((time.time(), "radiowatch", f"journalctl: {e}"))
            time.sleep(3)

    def _journal(self, j):
        msg = j.get("MESSAGE")
        if not isinstance(msg, str):
            return
        ts = int(j["__REALTIME_TIMESTAMP"]) / 1e6
        unit = (j.get("_SYSTEMD_UNIT") or "").replace(".service", "")
        d = self.d
        with d.lock:
            if (m := RE_WHISPER.search(msg)):
                d.whisper.append((ts, float(m[2]), float(m[4]), bool(m[3])))
            elif (m := RE_LLM.search(msg)):
                tok = lambda s: int(s) if s.isdigit() else 0  # noqa: E731
                d.llm.append((ts, int(m[1]), int(m[2]), int(m[3]), float(m[4]), tok(m[5]), tok(m[6])))
            elif (m := RE_START.search(msg)):
                d.on_air[int(m[1])] = (ts, m[2], m[3].split(" - ")[0], m[4])
            elif (m := RE_END.search(msg)):
                d.on_air.pop(int(m[1]), None)
            elif (m := RE_RECSTATS.search(msg)):
                try:
                    d.rec_stats = eval(m[1], {"__builtins__": {}})  # dict repr from our own logger
                except Exception:
                    pass
            if int(j.get("PRIORITY", 6)) <= 4 or " WARNING " in msg or " ERROR " in msg:
                d.events.append((ts, unit, re.sub(r"^\S+ \S+ (WARNING|ERROR) ", "", msg)))

    def slow_poller(self):
        sized = {}
        while not self.stop.is_set():
            out = {}
            for s in SERVICES:
                try:
                    o = subprocess.run(["systemctl", "show", s, "-p", "ActiveState",
                                        "-p", "ActiveEnterTimestampMonotonic", "-p", "NRestarts"],
                                       capture_output=True, text=True, timeout=5).stdout
                    kv = dict(line.split("=", 1) for line in o.splitlines() if "=" in line)
                    up = time.monotonic() - int(kv.get("ActiveEnterTimestampMonotonic", 0)) / 1e6
                    out[s] = (kv.get("ActiveState", "?"), up, kv.get("NRestarts", "0"))
                except Exception:
                    out[s] = ("?", 0, "?")
            t0 = time.time()
            try:
                urllib.request.urlopen(LLM_MODELS, timeout=4).read()
                out["llm"] = ("up", time.time() - t0)
            except Exception as e:
                out["llm"] = ("down", str(e)[:40])
            for name, path in (("root", "/"), ("share", SHARE)):
                res = {}
                th = threading.Thread(target=lambda: res.update(u=shutil.disk_usage(path)), daemon=True)
                th.start()
                th.join(6)
                out[name] = res.get("u")
            with self.d.lock:
                paths = [r.get("mp3") for r in self.d.recs if r.get("mp3")]
                day = self.d.today["day"]
            for p in paths:
                if p not in sized and day and f"/{day}/" in p:
                    try:
                        sized[p] = os.path.getsize(p)
                    except OSError:
                        sized[p] = 0
            out["mp3_today"] = sum(v for k, v in sized.items() if day and f"/{day}/" in k)
            with self.d.lock:
                self.d.slow = out
            self.stop.wait(15)

    # ---------- feed ----------
    def _match(self, r):
        if not self.filter:
            return True
        f = self.filter.lower()
        return any(f in str(x).lower() for x in (r.get("tg"), r.get("tg_label"), r.get("lid"), r.get("text")))

    def _render_rec(self, r):
        c = tg_color(r.get("tg"))
        text = RE_REPEAT.sub(lambda m: (m[1] * 3).rstrip() + f" … (×{len(m[0]) // len(m[1])}) ",
                             r.get("text", "")) or "(no speech)"
        body = Text(text, style="white" if r.get("text") else "dim italic")
        if self.filter:
            body.highlight_words([self.filter], "black on #ffaf5f", case_sensitive=False)
        body.highlight_regex(KEYWORDS, "bold #ff5f5f underline")
        if r.get("emergency"):
            body = Text("🚨 EMERGENCY  ", style="bold white on red") + body
        g = Table.grid(padding=(0, 1), expand=True)
        g.add_column(width=8, style="#6c7a96")
        g.add_column(width=1)
        g.add_column(width=20, no_wrap=True)
        g.add_column(width=5, justify="right", style="#6c7a96")
        g.add_column(width=5, justify="right", style="#4e5a72")
        g.add_column(ratio=1)
        g.add_row(datetime.fromtimestamp(r["start"], TZ).strftime("%H:%M:%S"),
                  Text("▌", style=c), Text(short_label(r)[:20], style=f"bold {c}"),
                  str(r.get("lid", "")), f"{r.get('duration', 0):.1f}s", body)
        return g

    def show(self, r):
        self.feed_buf.append(r)
        if self._match(r):
            self.query_one("#feed", RichLog).write(self._render_rec(r))

    def redraw_feed(self):
        log = self.query_one("#feed", RichLog)
        log.clear()
        for r in list(self.feed_buf)[-800:]:
            if self._match(r):
                log.write(self._render_rec(r))
        box = self.query_one("#feedbox")
        box.border_title = f"live transcripts · filter “{self.filter}”" if self.filter else "live transcripts"

    # ---------- panels ----------
    def refresh_panels(self):
        now = time.time()
        d = self.d
        with d.lock:
            d.prune()
            recs = list(d.recs)
            wsp = list(d.whisper)
            llm = list(d.llm)
            events = list(d.events)
            on_air = dict(d.on_air)
            today = dict(d.today)
            slow = dict(d.slow)
            rstats = dict(d.rec_stats)

        # volume sparkline: calls per minute, last 60 minutes
        hour = [r for r in recs if r["start"] >= now - 3600]
        bins = [0] * 60
        for r in hour:
            bins[min(59, int((r["start"] - (now - 3600)) // 60))] += 1
        self.query_one("#spark_calls", Sparkline).data = bins
        last5 = sum(bins[-5:]) / 5
        air = sum(r.get("duration", 0) for r in hour)
        util = air / (3600 * max(1, len(LCNS)))
        head = Text.assemble(("● ", "#5fd7d7"), (f"{bins[-1]}", "bold #5fd7d7"), " this min  ",
                             (f"{last5:.1f}", "bold"), "/min 5m avg  ", (f"{len(hour)}", "bold"), " in 1h  ",
                             ("peak ", "dim"), (f"{max(bins)}", "bold"))
        self.query_one("#vol_head", Static).update(head)
        self.query_one("#vol_axis", Static).update(Text.assemble(
            ("-60m", "dim"), " " * 22, ("-30m", "dim"), " " * 18, ("now", "dim"), "\n",
            ("airtime ", "dim"), bar(util, 20, "#5fd7d7"), f" {util * 100:.0f}% of {len(LCNS)} ch",
            "\n", ("today   ", "dim"), (f"{today['calls']:,}", "bold"), " calls ",
            (f"{today['audio'] / 3600:.1f}h", "bold"), " audio  ", (f"{today['words']:,}", "bold"), " words"))

        # on air
        t = Table.grid(padding=(0, 1))
        t.add_column(width=6)
        t.add_column(width=24, no_wrap=True)
        t.add_column(width=6, justify="right")
        t.add_column(width=12)
        lcn_air = Counter()
        for r in recs:
            if r["start"] >= now - 600:
                lcn_air[r.get("lcn")] += r.get("duration", 0)
        local_live = sorted(l for l in on_air if l >= 100 and now - on_air[l][0] < 200)
        for lcn in LCNS + local_live:
            if lcn in on_air and now - on_air[lcn][0] < 200:
                since, tg, lab, lid = on_air[lcn]
                c = tg_color(int(tg))
                t.add_row(Text(f"LCN {lcn}" if lcn < 100 else "VHF", style="bold"),
                          Text(f"▶ {lab}"[:24], style=f"bold {c}"),
                          Text(lid, style="dim"), Text(f"{now - since:4.0f}s", style="bold #87d787"))
            else:
                t.add_row(Text(f"LCN {lcn}", style="#4e5a72"), Text("idle", style="#4e5a72"), "",
                          bar(lcn_air[lcn] / 600, 12, "#3a4560"))
        self.query_one("#onair", Static).update(t)
        self.query_one("#onair").border_subtitle = "bars = airtime last 10m"

        # whisper
        w15 = [w for w in wsp if w[0] >= now - 900]
        took = [w[2] for w in w15]
        audio = sum(w[1] for w in w15)
        rtf = (sum(took) / audio) if audio else 0
        nos = sum(1 for w in w15 if w[3])
        try:
            queue = sum(1 for f in os.listdir(SPOOL) if f.endswith(".json"))
        except OSError:
            queue = -1
        qstyle = "bold #87d787" if queue <= 2 else "bold #ffd75f" if queue <= 20 else "bold #ff5f5f"
        lat_bins = [[] for _ in range(30)]
        for w in wsp:
            if w[0] >= now - 1800:
                lat_bins[min(29, int((w[0] - (now - 1800)) // 60))].append(w[2])
        self.query_one("#spark_lat", Sparkline).data = [max(b) if b else 0 for b in lat_bins]
        self.query_one("#wsp_stats", Static).update(Text.assemble(
            ("queue ", "dim"), (f"{queue}", qstyle), ("   15m: ", "dim"), (f"{len(w15)}", "bold"), " jobs  ",
            (f"{len(w15) / 15:.1f}", "bold"), "/min  no-speech ", (f"{(nos / len(w15) * 100) if w15 else 0:.0f}%", "bold"),
            "\n", ("latency ", "dim"), "avg ", (f"{(sum(took) / len(took)) if took else 0:.1f}s", "bold #ffaf5f"),
            "  p95 ", (f"{pct(took, .95):.1f}s", "bold #ffaf5f"), "  max ", (f"{max(took) if took else 0:.1f}s", "bold"),
            "  RTF ", (f"{rtf:.2f}", "bold"), ("\nsmall.en int8 · 2×3 threads · max latency/min, 30m", "dim")))

        # llm
        st = slow.get("llm")
        up = st and st[0] == "up"
        last = llm[-1] if llm else None
        l1h = [x for x in llm if x[0] >= now - 3600]
        lt = Text.assemble(("● ", "#87d787" if up else "#ff5f5f"),
                           ("LLM ", "bold"), ("reachable" if up else f"DOWN {st[1] if st else ''}",
                                                    "#87d787" if up else "bold #ff5f5f"))
        if up:
            lt.append(f" ({st[1] * 1000:.0f}ms)", "dim")
        if last:
            lt.append_text(Text.assemble(
                "\n", ("last pass ", "dim"), ago(last[0]), f": {last[1]} tx → {last[2]} incidents in ",
                (f"{last[4]:.1f}s", "bold"), f", {last[3]} open",
                "\n", ("1h: ", "dim"), f"{len(l1h)} passes, avg ",
                (f"{(sum(x[4] for x in l1h) / len(l1h)) if l1h else 0:.1f}s", "bold"),
                f", {sum(x[5] + x[6] for x in l1h):,} tokens"))
        else:
            lt.append("\nno incident passes in the last 2h", "dim")
        self.query_one("#llm", Static).update(lt)

        # busiest talkgroups
        cnt = Counter()
        lab = {}
        for r in hour:
            cnt[r.get("tg")] += 1
            lab[r.get("tg")] = short_label(r)
        tt = Table.grid(padding=(0, 1))
        tt.add_column(width=20, no_wrap=True)
        tt.add_column(width=22)
        tt.add_column(width=4, justify="right")
        top = cnt.most_common(8)
        mx = top[0][1] if top else 1
        for tg, n in top:
            c = tg_color(tg)
            tt.add_row(Text(lab[tg][:20], style=c), bar(n / mx, 22, c), Text(str(n), style="bold"))
        self.query_one("#tgs", Static).update(tt if top else Text("no calls yet", style="dim"))

        # system
        self.query_one("#sys", Static).update(self._sys_text(slow, events, rstats, today))

    def _sys_text(self, slow, events, rstats, today):
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle, total = v[3] + v[4], sum(v)
        cpu = 0.0
        if self.cpu_prev:
            dt = total - self.cpu_prev[1]
            cpu = 1 - (idle - self.cpu_prev[0]) / dt if dt else 0
        self.cpu_prev = (idle, total)
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, val = line.split(":")
                mem[k] = int(val.split()[0]) * 1024
        used = mem["MemTotal"] - mem["MemAvailable"]
        la = os.getloadavg()
        ncpu = os.cpu_count() or 1
        out = Text.assemble(("cpu  ", "dim"), bar(cpu, 24, "#87afff"), f" {cpu * 100:4.0f}%  ",
                            ("load ", "dim"), f"{la[0]:.2f} {la[1]:.2f} {la[2]:.2f}",
                            "\n", ("mem  ", "dim"), bar(used / mem["MemTotal"], 24, "#d787ff"),
                            f" {human(used)}/{human(mem['MemTotal'])}")
        for name, label in (("root", "disk "), ("share", "share")):
            u = slow.get(name)
            out.append("\n" + label + " ", "dim")
            if u:
                fr = u.used / u.total
                out.append_text(bar(fr, 24, "#ff5f5f" if fr > .9 else "#87d787"))
                out.append(f" {human(u.free)} free")
            elif slow:
                out.append("unreachable", "bold #ff5f5f")
        if slow.get("mp3_today"):
            out.append(f"\nmp3 today {human(slow['mp3_today'])}", "dim")
            if today["calls"]:
                out.append(f" · {today['words']:,} words transcribed", "dim")
        for s in SERVICES:
            stt = slow.get(s)
            cg = f"/sys/fs/cgroup/system.slice/{s}.service"
            try:
                with open(f"{cg}/cpu.stat") as f:
                    usec = int(f.readline().split()[1])
                with open(f"{cg}/memory.current") as f:
                    mcur = int(f.read())
            except OSError:
                usec, mcur = None, 0
            now = time.monotonic()
            core = 0.0
            if usec is not None and s in self.cg_prev:
                pu, pt = self.cg_prev[s]
                core = (usec - pu) / 1e6 / max(1e-3, now - pt)
            if usec is not None:
                self.cg_prev[s] = (usec, now)
            ok = stt and stt[0] == "active"
            out.append("\n")
            out.append("● " if ok else "✖ ", "#87d787" if ok else "bold #ff5f5f")
            out.append(f"{s:<15}", "bold")
            out.append_text(bar(core / ncpu, 10, "#87afff"))
            out.append(f" {core:4.2f} core {human(mcur):>6}")
            if stt:
                upt = stt[1]
                out.append(f"  up {upt / 3600:.0f}h" if upt > 3600 else f"  up {upt / 60:.0f}m", "dim")
                if stt[2] not in ("0", "?"):
                    out.append(f" ↻{stt[2]}", "#ffd75f")
        if rstats:
            out.append(f"\ngrants {rstats.get('grants', 0):,} · calls {rstats.get('calls', 0):,}"
                       f" · unmapped LCN {rstats.get('unknown_lcn', 0)}", "dim")
        for ts, unit, msg in events[-3:]:
            if time.time() - ts < 3600:
                out.append(f"\n⚠ {datetime.fromtimestamp(ts, TZ):%H:%M} {unit}: {msg}"[:56], "#ffd75f")
        return out

    def refresh_incidents(self):
        tbl = self.query_one("#incidents", DataTable)
        try:
            with open(STATE) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return
        op = st.get("open", {})
        incs = list(op.values()) if isinstance(op, dict) else op
        incs.sort(key=lambda i: (-i.get("severity", 1), -i.get("last", 0)))
        tbl.clear()
        for i in incs:
            sym, sty = SEV.get(i.get("severity", 1), SEV[1])
            tbl.add_row(Text(sym, style=sty), Text(i.get("id", ""), style="dim"),
                        Text(i.get("title", ""), style="bold" if i.get("severity", 1) >= 2 else ""),
                        Text(i.get("location", "") or "", style="dim"), str(i.get("count", "")),
                        ago(i.get("last")), ", ".join(v.split(" - ")[0] for v in i.get("tgs", {}).values())[:40])
        notable = sum(1 for i in incs if i.get("severity", 1) >= 2)
        tbl.border_title = f"open incidents · {notable} notable · {len(incs) - notable} minor"

    # ---------- actions ----------
    def action_filter(self):
        inp = self.query_one("#filter", Input)
        inp.add_class("show")
        inp.focus()

    def on_input_changed(self, ev: Input.Changed):
        self.filter = ev.value.strip()
        self.redraw_feed()

    def on_input_submitted(self, ev: Input.Submitted):
        self.query_one("#feed").focus()

    def action_clear_filter(self):
        inp = self.query_one("#filter", Input)
        inp.value = ""
        inp.remove_class("show")
        self.filter = ""
        self.redraw_feed()
        self.query_one("#feed").focus()

    def action_pause(self):
        log = self.query_one("#feed", RichLog)
        self.paused = not self.paused
        log.auto_scroll = not self.paused
        if not self.paused:
            log.scroll_end(animate=False)
        self.sub_title = "⏸ PAUSED (space resumes)" if self.paused else Watch.SUB_TITLE

    def action_incidents(self):
        self.query_one("#incidents").toggle_class("hide")

    def action_clear(self):
        self.query_one("#feed", RichLog).clear()

    def on_unmount(self):
        self.stop.set()


if __name__ == "__main__":
    Watch().run()
