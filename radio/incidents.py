"""
Incident threading for edacs_telegram.

Every `batch_s` the new transcripts plus the currently open incidents are sent
to the LLM, which assigns each transmission to an incident (existing or new)
or marks it routine, and returns updated incident state. Each notable incident
is one Telegram message, edited in place as it develops; minor incidents are
collected into an hourly round-up.
"""
import html
import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime

import numpy as np

import geomap
from audio_util import level_speech

log = logging.getLogger("edacs_telegram.incidents")


SYSTEM_PROMPT = """You track incidents on public-safety trunked radio around {area}.
Transcripts come from Whisper and are noisy: numbers, names and streets may be misheard.

You get OPEN INCIDENTS (already being tracked) and NEW TRANSMISSIONS (numbered).
Assign every new transmission to exactly one of:
  - an existing incident (by its id), when it continues that event (same units, location, subject);
  - a new incident (id "new"), when it starts a distinct event;
  - "routine": acknowledgements ("copy", "10-4"), radio checks, unit status changes, garbled fragments,
    shift/admin chatter that is not part of any event.
One talkgroup often carries several unrelated incidents at once - separate them by content, units and
location. An incident may span talkgroups (e.g. dispatch + tactical channel).

For every incident that received new transmissions (or that you now consider finished), return its FULL
updated state:
  type: crash | traffic | hazard | fire | medical | crime | disturbance | search | weather | utility | other
  title: <= 8 words, specific ("Jeep off roadway, I-80 WB Vista")
  location: best known location, or ""
  summary: 1-2 short sentences, cumulative, terse, newspaper style: what happened, where, injuries,
           resources requested, outcome. No filler ("units are coordinating", "indicating...",
           "acknowledges"). Don't narrate unit numbers or radio procedure. Keep 10-codes as spoken
           unless sure. Never invent details; write "unclear" if needed.
  severity - be conservative, most traffic is 1:
            1 minor: plate/RO/license/warrant checks, traffic stops, units en route with no detail,
              disabled or stuck vehicles, chain-up/snow issues, maintenance, admin, unclear codes
            2 notable: any crash, disturbance/domestic/fight, medical call, theft/burglary in
              progress, arrest, small fire, road blocked or lane closure, missing adult
            3 major: injury or fatal crash, structure/wild fire, weapons or shots, pursuit,
              officer needs help, missing child, full road closure, hazmat
            If you can't tell what the event is, it is 1.
  status: "active" or "resolved" (cleared, handled, units back in service)
  lines: indices of the NEW transmissions assigned to it

Reply with JSON only:
{"incidents": [{"id": "I12" | "new", "type": "...", "title": "...", "location": "...",
                "summary": "...", "severity": 1, "status": "active", "lines": [0, 3]}],
 "routine": [1, 2]}"""

EMOJI = {"crash": "💥", "traffic": "🚓", "hazard": "⚠️", "fire": "🔥", "medical": "🚑",
         "crime": "🚨", "disturbance": "🗣", "search": "🔎", "weather": "🌨", "utility": "⚡",
         "other": "📻"}


class LLMError(Exception):
    pass


def chat(cfg, env, system, user, max_tokens):
    from edacs_telegram import llm_request, prompt_for
    try:
        return llm_request(cfg, env, prompt_for(system, cfg), user, max_tokens, 0.2)
    except Exception as e:
        raise LLMError(str(e)) from e


class Incident:
    def __init__(self, iid):
        self.id = iid
        self.type, self.title, self.location, self.summary = "other", "", "", ""
        self.severity, self.status = 1, "active"
        self.first = self.last = 0.0
        self.tgs = {}          # tg -> label
        self.units = []
        self.count = 0
        self.recent = []       # last few transcript lines (context for the LLM)
        self.msg_id = None     # Telegram message id once posted
        self.msg_kind = None   # "audio" (playable, caption = text) or "text"
        self.clips = []        # [mp3 path, start, duration] of every transmission
        self.audio_n = 0       # clips included in the posted audio
        self.closed = False
        self.posted_text = ""
        self.geo = None        # geomap.Geo dict for `location`, or None
        self.geo_for = ""      # the location string `geo` was looked up for
        self.map_ids = None    # [close-up, vs-home] photo message ids
        self.map_key = None    # "lat,lon" the posted maps show

    def to_json(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_json(cls, d):
        i = cls(d["id"])
        i.__dict__.update(d)
        i.tgs = {int(k): v for k, v in i.tgs.items()}
        return i


class IncidentTracker:
    def __init__(self, cfg, env, telegram, tz, fallback_digest):
        self.cfg, self.env, self.tg, self.tz = cfg, env, telegram, tz
        self.fallback_digest = fallback_digest          # callable(recs) -> message
        self.state_file = cfg["incident_state_file"]
        self.open = {}                                  # id -> Incident
        self.minor_done = []                            # closed severity-1 incidents
        self.next_id = 1
        self.pending = []                               # transcripts not yet threaded
        self.lock = threading.Lock()
        self.last_roundup = time.time()
        self.home = cfg.get("home_latlon")
        geomap.configure(cache_file=cfg.get("geocode_cache"), tile_dir=cfg.get("map_tile_dir"),
                         region=cfg.get("geo_region"), state=cfg.get("geo_state"),
                         interstates=cfg.get("geo_interstates"), us_routes=cfg.get("geo_us_routes"),
                         force_ipv4=cfg.get("geo_force_ipv4"))
        self._load()

    # ---- persistence
    def _load(self):
        try:
            d = json.load(open(self.state_file))
            self.next_id = d["next_id"]
            self.open = {i["id"]: Incident.from_json(i) for i in d["open"]}
            self.minor_done = [Incident.from_json(i) for i in d.get("minor_done", [])]
            self.pending = d.get("pending", [])
            self.last_roundup = d.get("last_roundup", time.time())
            log.info("restored %d open incidents, %d pending transcripts",
                     len(self.open), len(self.pending))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("could not restore incident state; starting fresh")

    def _save(self):
        d = {"next_id": self.next_id, "open": [i.to_json() for i in self.open.values()],
             "minor_done": [i.to_json() for i in self.minor_done],
             "pending": self.pending, "last_roundup": self.last_roundup}
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.state_file)

    # ---- input
    def add(self, rec):
        with self.lock:
            self.pending.append(rec)

    # ---- main loop
    def run(self):
        while True:
            time.sleep(self.cfg["incident_batch_s"])
            try:
                self.step()
            except Exception:
                log.exception("incident step failed")

    def step(self):
        with self.lock:
            recs = sorted(self.pending, key=lambda r: r["start"])
            self.pending = []
        now = time.time()
        if recs:
            try:
                self._thread(recs)
            except LLMError as e:
                # keep them for the next try; give up after fallback_after_s
                stale = [r for r in recs if now - r["start"] > self.cfg["incident_fallback_after_s"]]
                fresh = [r for r in recs if r not in stale]
                log.warning("LLM unavailable (%s); %d queued, %d sent as plain digest",
                            e, len(fresh), len(stale))
                with self.lock:
                    self.pending = fresh + self.pending
                if stale:
                    self.tg.send(self.fallback_digest(stale))
        self._close_idle(now)
        if now - self.last_roundup >= self.cfg["incident_roundup_s"]:
            self._roundup(now)
        self._save()

    def _fmt_line(self, r):
        t = datetime.fromtimestamp(r["start"], self.tz)
        return (f"{t:%H:%M:%S} TG {r['tg']} {r.get('tg_label') or ''} "
                f"unit {r.get('lid') or '?'}: {r['text']}")

    def _thread(self, recs):
        opened = sorted(self.open.values(), key=lambda i: -i.last)[:40]
        ctx = []
        for i in opened:
            t = datetime.fromtimestamp(i.last, self.tz)
            ctx.append(f"{i.id} | TG {','.join(map(str, i.tgs))} | {i.type} | sev {i.severity} | "
                       f"last {t:%H:%M} | {i.title} | {i.location} | {i.summary}"
                       + "".join(f"\n    > {x}" for x in i.recent[-3:]))
        lines = [f"[{n}] {self._fmt_line(r)}" for n, r in enumerate(recs)]
        user = ("OPEN INCIDENTS:\n" + ("\n".join(ctx) if ctx else "(none)") +
                "\n\nNEW TRANSMISSIONS:\n" + "\n".join(lines))
        t0 = time.time()
        d, usage = chat(self.cfg, self.env, SYSTEM_PROMPT, user, self.cfg["llm_max_tokens"])
        assigned = set()
        touched = []
        for item in d.get("incidents", []):
            idx = [n for n in item.get("lines", []) if isinstance(n, int) and 0 <= n < len(recs)
                   and n not in assigned]
            iid = str(item.get("id", "new"))
            inc = self.open.get(iid)
            if inc is None:
                if not idx:
                    continue
                inc = Incident(f"I{self.next_id}")
                self.next_id += 1
                inc.first = recs[idx[0]]["start"]
                self.open[inc.id] = inc
            assigned.update(idx)
            for n in idx:
                r = recs[n]
                inc.last = max(inc.last, r["start"])
                inc.count += 1
                inc.tgs[r["tg"]] = r.get("tg_label") or ""
                if r.get("lid") and r["lid"] not in inc.units:
                    inc.units.append(r["lid"])
                inc.recent = (inc.recent + [self._fmt_line(r)])[-6:]
                if r.get("mp3"):
                    inc.clips.append([r["mp3"], r["start"], r.get("duration") or 0])
            for k in ("type", "title", "location", "summary", "status"):
                if item.get(k):
                    setattr(inc, k, str(item[k]))
            try:
                inc.severity = max(1, min(3, int(item.get("severity", inc.severity))))
            except (TypeError, ValueError):
                pass
            touched.append(inc)
        log.info("threaded %d transmissions -> %d incidents (%d open) in %.1fs, %s/%s tokens",
                 len(recs), len(touched), len(self.open), time.time() - t0,
                 usage.get("prompt_tokens"), usage.get("completion_tokens"))
        for inc in touched:
            if inc.status == "resolved":
                self._close(inc)
            else:
                self._publish(inc)

    # ---- output
    def _render(self, inc, limit=4096):
        t1 = datetime.fromtimestamp(inc.first, self.tz)
        t2 = datetime.fromtimestamp(inc.last, self.tz)
        span = f"{t1:%H:%M}–{t2:%H:%M}" if inc.last - inc.first >= 60 else f"{t1:%H:%M}"
        chans = ", ".join(f"{html.escape(l or 'TG ' + str(t))}" for t, l in inc.tgs.items())
        u = ", ".join(html.escape(x) for x in inc.units[:5]) + (f" +{len(inc.units) - 5}" if len(inc.units) > 5 else "")
        state = "✅ resolved" if inc.closed else "🟢 active"
        sev = {1: "", 2: "", 3: "‼️ "}[inc.severity]
        txt = (f"{sev}{EMOJI.get(inc.type, '📻')} <b>{html.escape(inc.title or inc.type)}</b>\n"
               + (f"📍 {html.escape(inc.location)}{self._rel(inc)}\n" if inc.location else "")
               + f"{html.escape(inc.summary)}\n"
               f"<i>{chans} · {span} · {inc.count} tx" + (f" · units {u}" if u else "")
               + f" · {state}</i>")
        over = len(re.sub(r"<[^>]+>", "", txt)) - limit
        if over > 0:                                  # captions are capped at 1024 chars
            summ = html.escape(inc.summary)
            txt = txt.replace(summ, summ[:max(0, len(summ) - over - 2)].rsplit(" ", 1)[0] + " …", 1)
        return txt

    def _build_audio(self, inc):
        """Join the incident's calls (oldest first) into one MP3; returns (path, n) or (None, 0)."""
        clips = sorted((c for c in inc.clips if c[0] and os.path.exists(c[0])), key=lambda c: c[1])
        keep, total = [], 0.0
        for c in reversed(clips):                      # newest back to the duration cap
            if total + c[2] > self.cfg["incident_audio_max_s"] and keep:
                break
            keep.append(c)
            total += c[2]
        keep.reverse()
        if not keep:
            return None, 0
        os.makedirs(self.cfg["incident_audio_dir"], exist_ok=True)
        out = os.path.join(self.cfg["incident_audio_dir"], f"{inc.id}.mp3")
        gap = np.zeros(int(self.cfg["incident_audio_gap_s"] * 16000), np.int16)
        parts = []
        try:
            for c in keep:                                     # decode, level each call
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", c[0], "-f", "s16le",
                                      "-ac", "1", "-ar", "16000", "-"],
                                     capture_output=True, check=True, timeout=30).stdout
                pcm = np.frombuffer(raw, np.int16)
                if len(pcm):
                    parts += [level_speech(pcm, 16000), gap]
            if not parts:
                return None, 0
            subprocess.run(["lame", "--quiet", "-r", "-s", "16", "--bitwidth", "16", "--signed",
                            "--little-endian", "-m", "m", "-b", "32", "-", out + ".tmp"],
                           input=np.concatenate(parts).tobytes(), check=True, timeout=120)
            os.replace(out + ".tmp", out)
        except Exception as e:
            err = getattr(e, "stderr", b"") or b""
            log.warning("audio for %s failed: %s %s", inc.id, e, err.decode(errors="replace")[-300:])
            return None, 0
        return out, len(inc.clips)

    def _audio_tags(self, inc):
        title = f"{inc.title or inc.type} ({inc.id})"
        performer = " / ".join(l or f"TG {t}" for t, l in inc.tgs.items()) or "radio"
        return title, performer

    def _rel(self, inc):
        if not (inc.geo and self.home):
            return ""
        return f" <i>({'≈ ' if inc.geo['approx'] else ''}{geomap.relation(inc.geo, self.home)})</i>"

    def _geocode(self, inc):
        if not self.cfg.get("incident_maps") or inc.location == inc.geo_for:
            return
        inc.geo_for = inc.location
        try:
            inc.geo = geomap.geocode(inc.location) if inc.location else None
        except Exception:
            log.exception("geocode for %s failed", inc.id)
            inc.geo = None

    def _publish_maps(self, inc):
        """Album of 2 maps replying to the incident message; edited in place if the pin moves."""
        g = inc.geo
        if not (self.cfg.get("incident_maps") and g and inc.msg_id is not None):
            return
        key = f"{g['lat']:.5f},{g['lon']:.5f}"
        if key == inc.map_key:
            return
        radius = self.cfg.get("incident_map_radius_mi")
        if radius and self.home and not inc.map_ids and \
                geomap.distance_mi(self.home[0], self.home[1], g["lat"], g["lon"]) > radius:
            return                                    # far away: distance in caption only
        try:
            close = geomap.render_close(g, inc.location)
            rel = geomap.render_home(g, self.home, inc.location) if self.home else None
        except Exception:
            log.exception("map render for %s failed", inc.id)
            return
        cap = f"🗺 {html.escape(inc.title or inc.type)} ({inc.id})"
        if inc.map_ids:
            ok = self.tg.edit_photo(inc.map_ids[0], close, cap)
            if rel and len(inc.map_ids) > 1:
                ok &= self.tg.edit_photo(inc.map_ids[1], rel)
        else:
            ids = self.tg.send_photos([(close, cap)] + ([(rel, "")] if rel else []),
                                      reply_to=inc.msg_id, silent=True)
            ok = bool(ids)
            if ids:
                inc.map_ids = ids
        if ok:
            inc.map_key = key
            log.info("maps for %s: %s (%s)", inc.id, inc.location, g.get("how"))

    def _publish(self, inc):
        if inc.severity < self.cfg["incident_min_severity_post"]:
            return                                    # minor: hourly round-up only
        self._geocode(inc)
        self._publish_msg(inc)
        self._publish_maps(inc)

    def _publish_msg(self, inc):
        audio = self.cfg.get("incident_audio") and inc.msg_kind != "text" and \
            (inc.msg_id is None or inc.msg_kind == "audio")
        txt = self._render(inc, limit=1000 if audio else 4096)
        if audio:
            title, perf = self._audio_tags(inc)
            if inc.msg_id is None or len(inc.clips) != inc.audio_n:
                path, n = self._build_audio(inc)
                if path and inc.msg_id is None:
                    inc.msg_id = self.tg.send_audio(path, txt, title, perf)
                    if inc.msg_id is not None:
                        inc.msg_kind, inc.audio_n, inc.posted_text = "audio", n, txt
                        return
                elif path:
                    if self.tg.edit_audio(inc.msg_id, path, txt, title, perf):
                        inc.audio_n, inc.posted_text = n, txt
                    return
            if inc.msg_id is not None:
                if txt != inc.posted_text and self.tg.edit_caption(inc.msg_id, txt):
                    inc.posted_text = txt
                return
            txt = self._render(inc)                   # no audio available: plain text message
        if inc.msg_id is None:
            inc.msg_id = self.tg.send_get_id(txt)
            inc.msg_kind = "text"
        elif txt != inc.posted_text:
            self.tg.edit(inc.msg_id, txt)
        inc.posted_text = txt

    def _close(self, inc):
        inc.closed = True
        self.open.pop(inc.id, None)
        if inc.severity < self.cfg["incident_min_severity_post"]:
            self.minor_done.append(inc)
        else:
            self._publish(inc)
        log.info("incident %s closed (sev %d, %d tx): %s", inc.id, inc.severity, inc.count, inc.title)

    def _close_idle(self, now):
        for inc in list(self.open.values()):
            if now - inc.last > self.cfg["incident_idle_close_s"]:
                self._close(inc)

    def _roundup(self, now):
        self.last_roundup = now
        items = sorted(self.minor_done, key=lambda i: i.first)
        self.minor_done = []
        if not items:
            return
        lines = []
        for inc in items:
            t = datetime.fromtimestamp(inc.first, self.tz)
            chan = next(iter(inc.tgs.values()), "") or ""
            lines.append(f"• {t:%H:%M} {EMOJI.get(inc.type, '📻')} {html.escape(inc.title)}"
                         + (f" — {html.escape(inc.location)}" if inc.location else "")
                         + (f" <i>({html.escape(chan)})</i>" if chan else ""))
        self.tg.send(f"🗒 <b>Minor activity</b> · {len(items)} items\n" + "\n".join(lines),
                     silent=True)
