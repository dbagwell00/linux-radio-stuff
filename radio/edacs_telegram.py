#!/usr/bin/env python3
"""
edacs_telegram - voice -> text -> (digest | alert) -> Telegram.

  * Watches the recorder's spool dir (<id>.json + <id>.pcm, 16 kHz int16) and
    transcribes each call with faster-whisper on CPU.
  * Every transcript is appended to transcripts/YYYY-MM-DD.jsonl.
  * mode "incidents" (default): an OpenAI-compatible LLM threads transcripts
    into incidents; each notable one is one Telegram message edited in place
    (optionally with the joined call audio), minor ones go to an hourly
    round-up. See incidents.py.
  * mode "digest": a per-talkgroup LLM summary every `digest_minutes`.
  * mode "per_call": every transcript is sent as its own message.
  * In incidents/digest mode, ALERTS go out immediately for the EDACS
    emergency flag or a keyword match.
  * If the LLM is unreachable a plain digest (counts + longest
    transmissions) is sent instead, so nothing is lost.

Usage: edacs_telegram.py [transcriber.json]   (its keys override CFG below)
"""
import html
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from faster_whisper import WhisperModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import incidents  # noqa: E402

log = logging.getLogger("edacs_telegram")

CFG = {
    # relative paths resolve against the working directory (systemd WorkingDirectory)
    "spool_dir": "spool",
    "transcript_dir": "transcripts",
    "env_file": "telegram.env",     # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, optional OPENAI_API_KEY
    "timezone": "UTC",
    # where the radio traffic is, e.g. "Springfield and Shelby County, Ohio"; used in LLM prompts
    "area": "the local area",
    # whisper
    "model": "small.en",
    "cpu_threads": 3,               # per worker
    "workers": 2,
    "beam_size": 1,
    "min_duration_s": 1.5,          # skip key-ups
    # Whisper initial prompt: a sentence with your place names, highways and jargon helps
    # spelling. No bare acronyms: on near-silent audio Whisper loops them ("ABC, ABC, ...").
    "initial_prompt": "Police, fire and EMS radio dispatch. Copy, en route, 10-4.",
    # prompt for conventional channels (local_rec.py jobs have source "local-*"); null = same
    "initial_prompt_local": None,
    "max_backlog": 200,
    # delivery
    "mode": "incidents",            # incidents | digest | per_call
    "incident_batch_s": 120,        # how often new transcripts are threaded
    "incident_idle_close_s": 900,   # close an incident after this much silence
    "incident_min_severity_post": 2,  # lower severities go to the hourly round-up
    "incident_roundup_s": 3600,
    "incident_fallback_after_s": 600,  # LLM down this long -> plain digest
    "incident_state_file": "incidents_state.json",
    "incident_audio": True,         # post incidents as playable audio (calls joined in order)
    "incident_audio_dir": "incident_audio",
    "incident_audio_max_s": 1200,   # keep only the most recent 20 min of calls in the clip
    "incident_audio_gap_s": 0.7,    # silence between calls
    # incident maps (geomap.py): a reply album with a close-up map and one showing the incident
    # relative to home_latlon; the caption's 📍 line gets "N mi DIR of home". Uses the free
    # US Census geocoder and OpenStreetMap Nominatim + tiles: keep the volume low.
    "incident_maps": False,
    "home_latlon": None,            # [lat, lon]; needed for the second map and the radius
    "incident_map_radius_mi": 10,   # maps only within this distance of home (0 = everywhere)
    "geo_region": None,             # [south, west, north, east] your coverage area
    "geo_state": "",                # e.g. "Nevada"
    "geo_interstates": [],          # numbers a bare "Highway N" means I-N
    "geo_us_routes": [],            # ... means US-N (any other number: SR N)
    "geo_force_ipv4": False,
    "geocode_cache": "geocache.json",
    "map_tile_dir": "tiles",
    "digest_minutes": 10,
    "digest_grace_s": 30,           # wait for late transcripts before closing a window
    "talkgroups": [],               # empty = all; else only these TGs
    "ignore_talkgroups": [],        # never transcribed / reported
    "alert_emergency": True,
    "alert_keywords": [
        r"shots? fired", r"shooting", r"\bshot\b", r"gunshot", r"\bgun\b", r"weapon",
        r"stab(bed|bing)?", r"pursuit", r"officer down", r"11-99", r"structure fire",
        r"working fire", r"fully involved", r"explosion", r"hazmat", r"fatal(ity)?",
        r"\bdoa\b", r"active shooter", r"amber alert", r"missing (child|juvenile)",
        r"cpr in progress", r"not breathing", r"barricaded", r"hostage"],
    "alert_talkgroups": [],         # every call on these TGs alerts immediately
    # LLM (OpenAI-compatible chat completions)
    "llm_url": "http://127.0.0.1:8000/v1/chat/completions",
    "llm_model": "",                # as listed by the server's /v1/models
    "llm_disable_thinking": False,  # Qwen3-style models: ask for no <think> block
    "llm_timeout_s": 300,
    "llm_max_tokens": 4000,
}

JUNK = {"", "you", "thank you.", "thanks for watching!", "bye.", "thank you",
        "thanks for watching.", ".", "okay.", "so", "thank you for watching.",
        "thank you for watching!", "thank you for watching", "thanks for watching"}
HALLUCINATION = re.compile(r"thank(s| you) for watching|subscribe|subtitles by", re.I)


REPEAT_RUN = re.compile(r"\b([\w'-]+)(?:[,.!?]?\s+\1\b){3,}", re.I)


def strip_loops(text):
    """Drop Whisper repetition loops ("ABC, ABC, ABC, ..."); keep the rest of the line."""
    if not text:
        return text
    loops = sum(len(m.group(0)) for m in REPEAT_RUN.finditer(text))
    if loops > 0.4 * len(text):
        return ""
    return REPEAT_RUN.sub(lambda m: m.group(1), text).strip()


def _norm(t):
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()


def echoes_prompt(text, prompt):
    """Whisper on near-silence often just repeats (part of) its initial prompt."""
    w, p = _norm(text), set(_norm(prompt))
    return len(w) >= 3 and sum(x in p for x in w) / len(w) >= 0.8

SYSTEM_PROMPT = """You summarize public-safety radio traffic for a resident of {area}.
Input: Whisper transcripts of trunked-radio transmissions from the last few minutes, one per line:
  HH:MM:SS [TG <number> <talkgroup name>] unit <radio id>: <text>
Transcripts are noisy: numbers, names and street names may be misheard.

Produce, per talkgroup, a few short bullets describing what actually happened:
- Merge related transmissions into one incident (a traffic stop, a crash, a call for service).
- Keep concrete facts: incident type, location (street, highway, mile marker, exit, business), vehicle
  descriptions, injuries, units/resources requested (tow, medical, fire), outcome.
- Keep 10-codes as spoken (e.g. "10-51") unless you are sure of their meaning.
- Skip pure acknowledgements, radio checks, status changes and garbled fragments.
- Do not invent details. If unsure, say so briefly ("unclear").
- A talkgroup with only routine traffic gets an empty bullet list.
- At most 4 bullets per talkgroup, each under 25 words.

Reply with JSON only:
{"headline": "<one sentence on the most significant thing in this window, or empty>",
 "groups": [{"tg": <number>, "bullets": ["...", "..."]}]}"""


ENV_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY")


def load_env(path):
    """Secrets from a KEY=value file; real environment variables take precedence."""
    env = {}
    if path and os.path.exists(path):
        if os.stat(path).st_mode & 0o077:
            log.warning("%s is readable by other users; chmod 600 it", path)
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)})
    return env


def prompt_for(template, cfg):
    return template.replace("{area}", cfg.get("area") or "the local area")


def llm_request(cfg, env, system, user, max_tokens, temperature):
    """POST an OpenAI-compatible chat completion; returns (json_dict, usage).

    Transcripts are untrusted input (anyone can key up a radio), so the reply is
    only ever parsed as JSON, and every field is HTML-escaped before it is sent."""
    body = {
        "model": cfg["llm_model"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if cfg.get("llm_disable_thinking"):
        body["messages"][1]["content"] += "\n/no_think"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if env.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {env['OPENAI_API_KEY']}"

    def call(b):
        req = urllib.request.Request(cfg["llm_url"], data=json.dumps(b).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=cfg["llm_timeout_s"]) as resp:
            return json.loads(resp.read())

    try:
        out = call(body)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        body.pop("response_format")            # server without JSON mode
        body.pop("chat_template_kwargs", None)
        out = call(body)
    content = out["choices"][0]["message"]["content"] or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in LLM reply: {content[:200]!r}")
    return json.loads(m.group(0)), out.get("usage", {})


# --------------------------------------------------------------------------
class Telegram:
    def __init__(self, token, chat_id):
        self.base = f"https://api.telegram.org/bot{token}/"
        self.chat_id = chat_id
        self.next_ok = 0.0
        self.lock = threading.Lock()

    def send(self, text, silent=False):
        """Send a message (split if > 4096). Returns True on success."""
        ok = True
        for p in split_message(text):
            ok &= self._call("sendMessage", {"text": p}, silent) is not None
        return ok

    def send_get_id(self, text, silent=False):
        """Send one message (<= 4096 chars) and return its message_id, or None."""
        r = self._call("sendMessage", {"text": text[:4096]}, silent)
        return r["result"]["message_id"] if r else None

    def edit(self, message_id, text):
        return self._call("editMessageText",
                          {"message_id": message_id, "text": text[:4096]}) is not None

    # ---- audio (incident recordings): one playable message, caption = incident text
    @staticmethod
    def _audio_name(title):
        safe = re.sub(r"[^A-Za-z0-9 ._-]+", "", title).strip()[:60] or "radio"
        return safe + ".mp3"

    def send_audio(self, path, caption, title, performer, silent=False):
        with open(path, "rb") as f:
            data = f.read()
        r = self._call("sendAudio", {"caption": caption, "title": title[:64],
                                     "performer": performer[:64]}, silent,
                       files={"audio": (self._audio_name(title), data, "audio/mpeg")})
        return r["result"]["message_id"] if r and "result" in r else None

    def edit_audio(self, message_id, path, caption, title, performer):
        with open(path, "rb") as f:
            data = f.read()
        media = {"type": "audio", "media": "attach://clip", "caption": caption,
                 "parse_mode": "HTML", "title": title[:64], "performer": performer[:64]}
        return self._call("editMessageMedia",
                          {"message_id": message_id, "media": json.dumps(media)},
                          files={"clip": (self._audio_name(title), data, "audio/mpeg")}
                          ) is not None

    def edit_caption(self, message_id, caption):
        return self._call("editMessageCaption",
                          {"message_id": message_id, "caption": caption}) is not None

    # ---- photos (incident maps): an album replying to the incident message
    def send_photos(self, photos, reply_to=None, silent=True):
        """photos = [(png bytes, caption or "")]; returns the album's message ids or None."""
        media, files = [], {}
        for n, (data, cap) in enumerate(photos):
            m = {"type": "photo", "media": f"attach://p{n}"}
            if cap:
                m.update(caption=cap, parse_mode="HTML")
            media.append(m)
            files[f"p{n}"] = (f"map{n}.png", data, "image/png")
        f = {"media": json.dumps(media)}
        if reply_to:
            f["reply_parameters"] = json.dumps({"message_id": reply_to,
                                                "allow_sending_without_reply": True})
        r = self._call("sendMediaGroup", f, silent, files=files)
        return [m["message_id"] for m in r["result"]] if r and "result" in r else None

    def edit_photo(self, message_id, data, caption=""):
        m = {"type": "photo", "media": "attach://p", "caption": caption, "parse_mode": "HTML"}
        return self._call("editMessageMedia", {"message_id": message_id, "media": json.dumps(m)},
                          files={"p": ("map.png", data, "image/png")}) is not None

    @staticmethod
    def _multipart(params, files):
        b = uuid.uuid4().hex
        body = bytearray()
        for k, v in params.items():
            body += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                     f'{v}\r\n').encode()
        for k, (fname, data, ctype) in files.items():
            body += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"; '
                     f'filename="{fname}"\r\nContent-Type: {ctype}\r\n\r\n').encode()
            body += data + b"\r\n"
        body += f"--{b}--\r\n".encode()
        return bytes(body), f"multipart/form-data; boundary={b}"

    def _call(self, method, fields, silent=False, files=None):
        with self.lock:
            wait = self.next_ok - time.time()
            if wait > 0:
                time.sleep(wait)
            params = {"chat_id": self.chat_id, "parse_mode": "HTML",
                      "disable_web_page_preview": "true", **fields}
            if silent:
                params["disable_notification"] = "true"
            if files:
                data, ctype = self._multipart(params, files)
                headers, timeout = {"Content-Type": ctype}, 120
            else:
                data = urllib.parse.urlencode(params).encode()
                headers, timeout = {}, 20
            for attempt in range(5):
                try:
                    req = urllib.request.Request(self.base + method, data=data, headers=headers)
                    with urllib.request.urlopen(req, timeout=timeout) as r:
                        out = json.loads(r.read())
                    self.next_ok = time.time() + 1.05
                    return out
                except urllib.error.HTTPError as e:
                    body = e.read().decode(errors="replace")
                    if e.code == 429:
                        try:
                            ra = json.loads(body)["parameters"]["retry_after"]
                        except Exception:
                            ra = 5
                        log.warning("telegram rate limited, waiting %ss", ra)
                        time.sleep(ra + 0.5)
                        continue
                    if "message is not modified" in body:
                        return {"ok": True}
                    log.error("telegram %s HTTP %s: %s", method, e.code, body[:300])
                    return None
                except Exception as e:
                    log.warning("telegram %s failed (%s), retrying", method, e)
                    time.sleep(2 * (attempt + 1))
            return None


def split_message(text, limit=4000):
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for block in text.split("\n"):
        if len(cur) + len(block) + 1 > limit:
            out.append(cur)
            cur = ""
        cur += block + "\n"
    if cur.strip():
        out.append(cur)
    return out


# --------------------------------------------------------------------------
def llm_digest(cfg, env, recs, tz):
    """Ask the LLM for per-TG bullets. Returns dict or raises."""
    lines = []
    for r in recs:
        t = datetime.fromtimestamp(r["start"], tz)
        lines.append(f"{t:%H:%M:%S} [TG {r['tg']} {r.get('tg_label') or ''}] "
                     f"unit {r.get('lid') or '?'}: {r['text']}")
    t0 = time.time()
    d, u = llm_request(cfg, env, prompt_for(SYSTEM_PROMPT, cfg), "\n".join(lines),
                       cfg["llm_max_tokens"], 0.3)
    log.info("llm digest: %d lines -> %d groups in %.1fs (%s prompt / %s completion tokens)",
             len(lines), len(d.get("groups", [])), time.time() - t0,
             u.get("prompt_tokens"), u.get("completion_tokens"))
    return d


def build_digest(cfg, env, recs, t0, t1, tz, use_llm=True):
    by_tg = defaultdict(list)
    for r in recs:
        by_tg[r["tg"]].append(r)
    order = sorted(by_tg, key=lambda tg: -len(by_tg[tg]))
    head = (f"🗒 <b>{datetime.fromtimestamp(t0, tz):%H:%M}–{datetime.fromtimestamp(t1, tz):%H:%M}</b>"
            f" · {len(recs)} calls · {len(by_tg)} talkgroups")

    summary, llm_ok = {}, False
    try:
        if not use_llm:
            raise RuntimeError("LLM skipped")
        d = llm_digest(cfg, env, recs, tz)
        llm_ok = True
        for g in d.get("groups", []):
            try:
                summary[int(g.get("tg"))] = [str(b) for b in g.get("bullets", []) if str(b).strip()]
            except (TypeError, ValueError):
                pass
        if d.get("headline"):
            head += f"\n<b>{html.escape(str(d['headline']))}</b>"
    except Exception as e:
        log.warning("llm digest failed (%s); sending plain digest", e)
        head += "\n<i>(summarizer offline — longest transmissions shown)</i>"

    blocks, routine = [], []
    for tg in order:
        rs = by_tg[tg]
        name = html.escape(rs[0].get("tg_label") or f"TG {tg}")
        units = []
        for r in rs:
            if r.get("lid") and r["lid"] not in units:
                units.append(r["lid"])
        u = ", ".join(html.escape(x) for x in units[:4]) + (f" +{len(units) - 4}" if len(units) > 4 else "")
        if llm_ok:
            bullets = summary.get(tg, [])
        else:
            longest = sorted(rs, key=lambda r: -len(r["text"]))[:3]
            bullets = [r["text"][:200] for r in longest if len(r["text"]) > 25]
        if not bullets:
            routine.append(f"{name} ({len(rs)})")
            continue
        blk = f"\n<b>{name}</b> <i>({tg})</i> · {len(rs)} calls" + (f" · units {u}" if u else "")
        blk += "".join(f"\n• {html.escape(b)}" for b in bullets)
        blocks.append(blk)
    msg = head + "\n" + "\n".join(blocks)
    if routine:
        msg += "\n\n<i>Routine: " + ", ".join(routine) + "</i>"
    return msg


def fmt_call(r, tz, prefix=""):
    t = datetime.fromtimestamp(r["start"], tz)
    label = r.get("tg_label") or f"TG {r['tg']}"
    return (f"{prefix}<b>{html.escape(label)}</b> <i>({r['tg']})</i>\n"
            f"🕑 {t:%H:%M:%S} · unit {html.escape(r.get('lid') or '?')} · {r['duration']:.0f}s\n"
            f"{html.escape(r['text'])}")


def jobs(spool):
    try:
        return sorted(n for n in os.listdir(spool)
                      if n.endswith(".json") and not n.startswith("."))
    except FileNotFoundError:
        return []


def drop(spool, name):
    for ext in (".json", ".pcm"):
        try:
            os.remove(os.path.join(spool, name[:-5] + ext))
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    cfg = dict(CFG)
    if len(sys.argv) > 1:
        cfg.update(json.load(open(sys.argv[1])))
    env = load_env(cfg["env_file"])
    missing = [k for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not env.get(k)]
    if missing:
        sys.exit(f"missing {', '.join(missing)} (set them in {cfg['env_file']} or the environment)")
    if not cfg["llm_model"] and cfg["mode"] != "per_call":
        log.warning("llm_model is empty; set it to a model your server lists at /v1/models")
    tg = Telegram(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"])
    tz = ZoneInfo(cfg["timezone"])
    spool = cfg["spool_dir"]
    os.makedirs(spool, exist_ok=True)
    os.makedirs(cfg["transcript_dir"], exist_ok=True)
    allow = set(cfg["talkgroups"])
    deny = set(cfg["ignore_talkgroups"])
    alert_tgs = set(cfg["alert_talkgroups"])
    kw = re.compile("|".join(cfg["alert_keywords"]), re.I) if cfg["alert_keywords"] else None

    workers = cfg["workers"]
    log.info("loading whisper model %s (%d workers x %d threads); mode=%s",
             cfg["model"], workers, cfg["cpu_threads"], cfg["mode"])
    model = WhisperModel(cfg["model"], device="cpu", compute_type="int8",
                         cpu_threads=cfg["cpu_threads"], num_workers=workers)
    log.info("ready; watching %s", spool)

    lock = threading.Lock()
    window = []                       # transcripts waiting for the next digest
    sendq = queue.Queue()             # (start_time, message) to send now
    work = queue.Queue()
    claimed = set()

    def record(r):
        day = datetime.fromtimestamp(r["start"], tz).strftime("%Y-%m-%d")
        with lock:
            with open(os.path.join(cfg["transcript_dir"], day + ".jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")

    def worker():
        while True:
            n = work.get()
            try:
                meta = json.load(open(os.path.join(spool, n)))
                if (meta["duration"] < cfg["min_duration_s"]
                        or (allow and meta["tg"] not in allow) or meta["tg"] in deny):
                    continue
                pcm = os.path.join(spool, n[:-5] + ".pcm")
                audio = np.fromfile(pcm, dtype=np.int16).astype(np.float32) / 32768
                t0 = time.time()
                prompt = cfg["initial_prompt"]
                if str(meta.get("source", "")).startswith("local"):
                    prompt = cfg.get("initial_prompt_local") or prompt
                segs, _ = model.transcribe(
                    audio, language="en", beam_size=cfg["beam_size"],
                    vad_filter=False, condition_on_previous_text=False, temperature=0.0,
                    initial_prompt=prompt)
                segs = [s for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2
                        and s.compression_ratio < 2.4]
                text = strip_loops(" ".join(s.text.strip() for s in segs).strip())
                took = time.time() - t0
                emerg = bool(meta.get("emergency"))
                if HALLUCINATION.search(text) and len(text) < 60:
                    text = ""
                if echoes_prompt(text, prompt):
                    text = ""
                if text.lower() in JUNK and not emerg:
                    log.info("TG %s %.1fs: (no speech) [%.1fs]", meta["tg"], meta["duration"], took)
                    continue
                r = {k: meta.get(k) for k in ("start", "tg", "tg_label", "lid", "lcn",
                                               "duration", "mp3", "emergency")}
                r["text"] = text
                record(r)
                hit = kw.search(text) if kw else None
                why = ("EMERGENCY" if emerg and cfg["alert_emergency"] else
                       f"“{hit.group(0)}”" if hit else
                       "watched talkgroup" if meta["tg"] in alert_tgs else None)
                log.info("TG %s %.1fs [%.1fs]%s: %s", meta["tg"], meta["duration"], took,
                         f" ALERT({why})" if why else "", text)
                if cfg["mode"] == "per_call":
                    sendq.put((r["start"], fmt_call(r, tz)))
                else:
                    if why:
                        sendq.put((r["start"], fmt_call(
                            r, tz, prefix=f"🚨 <b>{html.escape(why)}</b> · ")))
                    if tracker is not None:
                        tracker.add(r)
                    else:
                        with lock:
                            window.append(r)
            except Exception:
                log.exception("job %s failed", n)
            finally:
                drop(spool, n)
                with lock:
                    claimed.discard(n)

    def digester():
        period = cfg["digest_minutes"] * 60
        start = (time.time() // period) * period
        # restart-safe: reload this window's transcripts from today's log
        day = datetime.fromtimestamp(start, tz).strftime("%Y-%m-%d")
        try:
            with open(os.path.join(cfg["transcript_dir"], day + ".jsonl")) as f:
                old = [json.loads(l) for l in f if l.strip()]
            old = [r for r in old if r.get("start", 0) >= start]
            with lock:
                seen = {(r["start"], r.get("lcn")) for r in window}
                window.extend(r for r in old if (r["start"], r.get("lcn")) not in seen)
            if old:
                log.info("reloaded %d transcripts for the current digest window", len(old))
        except FileNotFoundError:
            pass
        while True:
            end = start + period
            time.sleep(max(0, end + cfg["digest_grace_s"] - time.time()))
            with lock:
                recs = sorted((r for r in window if r["start"] < end), key=lambda r: r["start"])
                window[:] = [r for r in window if r["start"] >= end]
            if recs:
                try:
                    sendq.put((end, build_digest(cfg, env, recs, start, end, tz)))
                except Exception:
                    log.exception("digest build failed")
            else:
                log.info("digest window empty")
            start = end

    tracker = None
    if cfg["mode"] == "incidents":
        def plain(recs):
            return build_digest(cfg, env, recs, recs[0]["start"], recs[-1]["start"] + 1, tz,
                                use_llm=False)
        tracker = incidents.IncidentTracker(cfg, env, tg, tz, plain)

    for _ in range(workers):
        threading.Thread(target=worker, daemon=True, name="whisper").start()
    if cfg["mode"] == "digest":
        threading.Thread(target=digester, daemon=True, name="digest").start()
    elif cfg["mode"] == "incidents":
        threading.Thread(target=tracker.run, daemon=True, name="incidents").start()

    def sender():
        while True:
            _, msg = sendq.get()
            tg.send(msg)
    threading.Thread(target=sender, daemon=True, name="telegram").start()

    last_warn = 0
    while True:
        names = [n for n in jobs(spool) if n not in claimed]
        if len(names) > cfg["max_backlog"]:
            for n in names[:-cfg["max_backlog"]]:
                drop(spool, n)
            names = names[-cfg["max_backlog"]:]
            log.warning("backlog over %d, dropped oldest", cfg["max_backlog"])
        for n in names:
            with lock:
                claimed.add(n)
            work.put(n)
        if work.qsize() > 20 and time.time() - last_warn > 60:
            last_warn = time.time()
            log.warning("transcription backlog: %d calls", work.qsize())
        time.sleep(0.3)


if __name__ == "__main__":
    main()
