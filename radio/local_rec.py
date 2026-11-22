#!/usr/bin/env python3
"""
local_rec - recorder for conventional (non-trunked) analog FM and P25 channels.

One RTL-SDR (via rtl_tcp, bias-T on) streams a 2.4 MHz slice; an overlap-save
FFT channelizer pulls out every configured channel at 48 kHz.

  * mode "fm"    : power squelch -> FM demod -> MP3 + transcription job
  * mode "p25"   : raw discriminator audio is served on a local TCP port and
                   dsd-fme (-i tcp:...) decodes it; its per-call WAVs become
                   MP3 + transcription jobs, tagged with the P25 source radio ID
  * mode "mixed" : both, for channels that carry P25 and analog traffic. If the
                   channel has a "tone" (CTCSS Hz), the analog side only keeps
                   calls carrying it, so P25 bursts aren't recorded twice as noise.

An SDR entry with "direct" hands the whole dongle to dsd-fme on one P25
channel; dsd-fme's own demodulator decodes noticeably better than the tap.

Calls land in the same output tree / spool as edacs_rec, so the transcriber
picks them up unchanged. The "talkgroup" of a
conventional channel is its frequency in kHz (e.g. 159210).
"""
import glob
import json
import logging
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edacs_rec import TapServer, connect_rtl_tcp, level_speech, safe, start_rtl_tcp  # noqa: E402

log = logging.getLogger("local_rec")

FS = 2_400_000
DEC = 50                 # -> 48 kHz channels (what dsd-fme wants on tcp input)
CH_FS = FS // DEC
N = 76_800
K = N // DEC             # 1536
OVL = 1_200
STEP = N - OVL
DISCARD = OVL // DEC     # 24
AUDIO_FS = 16_000
TONE_FS = 2_000          # CTCSS analysis rate
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TAP_REC = os.environ.get("TAP_REC")   # debug: append raw tap audio here


# --------------------------------------------------------------------------
class Channelizer(threading.Thread):
    def __init__(self, sdr_cfg, chans, bias):
        super().__init__(daemon=True, name=f"sdr-{sdr_cfg['serial']}")
        self.serial = sdr_cfg["serial"]
        self.center = sdr_cfg["center_hz"]
        self.gain = sdr_cfg.get("gain", 0)
        self.ppm = sdr_cfg.get("ppm", 0)
        self.port = sdr_cfg.get("port", 1250)
        self.bias = bias
        self.chans = chans
        h = sps.firwin(OVL + 1, sdr_cfg.get("filter_hz", 7_000), fs=FS)
        self.Hf = np.fft.fft(h, N).astype(np.complex64)
        self.hidx = (np.fft.fftfreq(K, 1 / K).astype(int)) % N
        for c in chans:
            off = c.freq - self.center
            if abs(off) > FS / 2 * 0.92:
                raise ValueError(f"{c.name} {c.freq} outside window of {self.serial}")
            c.k0 = int(round(off / (FS / N)))
            c.idx = (c.k0 + np.fft.fftfreq(K, 1 / K).astype(int)) % N
        self.block = 0
        self.proc = None
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self._stream()
            except Exception as e:
                log.warning("sdr %s: %s", self.serial, e)
            if not self.stop.is_set():
                time.sleep(3)

    def _stream(self):
        log.info("sdr %s: center %.4f MHz gain %s bias-T %s, %d channels", self.serial,
                 self.center / 1e6, self.gain, self.bias, len(self.chans))
        self.proc = start_rtl_tcp(self.serial, self.port, self.center, FS, self.gain,
                                  self.ppm, self.bias)
        sock = None
        buf = np.zeros(N, dtype=np.complex64)
        need = STEP * 2
        raw = bytearray()
        try:
            sock = connect_rtl_tcp(self.proc, self.port)
            while not self.stop.is_set():
                chunk = sock.recv(min(1 << 18, need - len(raw)))
                if not chunk:
                    raise IOError("rtl_tcp closed the stream")
                raw += chunk
                if len(raw) < need:
                    continue
                iq = np.frombuffer(bytes(raw), dtype=np.uint8).astype(np.float32)
                raw.clear()
                iq = (iq - 127.4) / 128.0
                buf[:OVL] = buf[STEP:]
                buf[OVL:] = iq[0::2] + 1j * iq[1::2]
                X = np.fft.fft(buf)
                ph = (self.block * STEP) % N
                for c in self.chans:
                    y = np.fft.ifft(X[c.idx] * self.Hf[self.hidx])[DISCARD:]
                    y *= np.exp(-2j * np.pi * c.k0 * ph / N)
                    c.feed(y.astype(np.complex64))
                self.block += 1
        finally:
            if sock:
                sock.close()
            self.proc.kill()
            self.proc.wait()


# --------------------------------------------------------------------------
class Chan:
    def __init__(self, cfg, lcn, writer, defaults):
        self.name = cfg["name"]
        self.freq = int(cfg["freq"])
        self.mode = cfg.get("mode", "fm")
        self.tone = cfg.get("tone")
        self.tg = int(round(self.freq / 1000))
        self.lcn = lcn
        self.writer = writer
        self.analog = self.mode in ("fm", "mixed")
        self.digital = self.mode in ("p25", "mixed")
        self.prev = np.complex64(1)
        self.open_db = cfg.get("squelch_open_db", defaults.get("squelch_open_db", 10))
        self.hold_db = cfg.get("squelch_hold_db", defaults.get("squelch_hold_db", 6))
        self.hang = defaults.get("hangtime_s", 1.2)
        self.max_dur = defaults.get("max_duration_s", 120)
        self.floor = None
        self.above = 0
        self.call = None
        self.preroll = deque(maxlen=6)
        tau = defaults.get("deemphasis_tau_s", 750e-6)
        a = np.exp(-1.0 / (CH_FS * tau))
        self.de = ([1 - a], [1, -a], np.zeros(1))
        self.bp = sps.butter(4, [300, 3400], btype="bandpass", fs=CH_FS, output="sos")
        self.bp_zi = np.zeros((self.bp.shape[0], 2))
        self.audio_gain = CH_FS / (2 * np.pi * 2_500) * 8000   # +/-2.5 kHz NFM
        # noise ("quieting") squelch: hiss above 6 kHz in the discriminator output
        # collapses when a real FM carrier is present; front-end overload doesn't do that
        self.hf_sos = sps.butter(4, 6_000, btype="highpass", fs=CH_FS, output="sos")
        self.hf_zi = np.zeros((self.hf_sos.shape[0], 2))
        self.hf_ref = None
        self.quiet_db = cfg.get("quiet_db", defaults.get("quiet_db", 8))
        self.quiet_hold_db = defaults.get("quiet_hold_db", 4)
        self.tap = None
        self.tap_port = None
        self.disc_scale = CH_FS / (2 * np.pi) * 3.0           # Hz -> int16 counts
        self.hp_b, self.hp_a = sps.butter(1, 5, "highpass", fs=CH_FS)   # strip ppm DC offset
        self.hp_zi = np.zeros(1)
        self.power_db = -99.0

    # called from the channelizer thread, ~32 blocks/s
    def feed(self, y):
        z = np.concatenate(([self.prev], y))
        self.prev = y[-1]
        disc = np.angle(z[1:] * np.conj(z[:-1])).astype(np.float32)
        if self.tap:
            d48, self.hp_zi = sps.lfilter(self.hp_b, self.hp_a, disc * self.disc_scale,
                                          zi=self.hp_zi)
            pcm48 = np.clip(d48, -32767, 32767).astype(np.int16)
            self.tap.send(pcm48)
            if TAP_REC:
                if not hasattr(self, "_rec"):
                    self._rec = open(os.path.join(TAP_REC, f"tap_{self.tg}.s16"), "ab")
                self._rec.write(pcm48.tobytes())
        p = 10 * np.log10(float(np.mean(np.abs(y) ** 2)) + 1e-12)
        self.power_db = p
        self.peak_db = max(getattr(self, "peak_db", -99.0), p)
        if not self.analog:
            if self.floor is None or p < self.floor:
                self.floor = p if self.floor is None else 0.8 * self.floor + 0.2 * p
            elif p < self.floor + 3:
                self.floor = 0.998 * self.floor + 0.002 * p
            return
        if self.floor is None:
            self.floor = p
        if p < self.floor:
            self.floor = 0.8 * self.floor + 0.2 * p
        elif p < self.floor + 3:
            self.floor = 0.998 * self.floor + 0.002 * p
        d, zi = sps.lfilter(self.de[0], self.de[1], disc, zi=self.de[2])
        self.de = (self.de[0], self.de[1], zi)
        a, self.bp_zi = sps.sosfilt(self.bp, d, zi=self.bp_zi)
        pcm = np.clip(a[::CH_FS // AUDIO_FS] * self.audio_gain, -32767, 32767).astype(np.int16)
        tone = disc.reshape(-1, CH_FS // TONE_FS).mean(axis=1) if self.tone else None
        hp, self.hf_zi = sps.sosfilt(self.hf_sos, disc, zi=self.hf_zi)
        hf = 10 * np.log10(float(np.mean(hp * hp)) + 1e-12)
        if self.hf_ref is None:
            self.hf_ref = hf
        elif hf > self.hf_ref:
            self.hf_ref = 0.9 * self.hf_ref + 0.1 * hf
        else:
            self.hf_ref = 0.9995 * self.hf_ref + 0.0005 * hf
        quiet = self.hf_ref - hf                    # dB of quieting vs pure noise
        self.quiet_peak = max(getattr(self, "quiet_peak", 0.0), quiet)
        now = time.time()
        if self.call is None:
            self.preroll.append((pcm, tone))
            self.above = (self.above + 1 if p > self.floor + self.open_db
                          and quiet > self.quiet_db else 0)
            if self.above >= 2:
                self.call = {"start": now - len(self.preroll) * len(y) / CH_FS,
                             "last": now, "pcm": [x[0] for x in self.preroll],
                             "tone": [x[1] for x in self.preroll if x[1] is not None]}
                self.preroll.clear()
                log.info("call start  LCN %-3d TG %-6s %-40s LID -", self.lcn, self.tg, self.name)
            return
        c = self.call
        c["pcm"].append(pcm)
        if tone is not None:
            c["tone"].append(tone)
        if p > self.floor + self.hold_db and quiet > self.quiet_hold_db:
            c["last"] = now
        if now - c["last"] > self.hang or now - c["start"] > self.max_dur:
            self.call = None
            self.above = 0
            keep = len(c["pcm"]) - int((now - c["last"]) * CH_FS / len(y)) + 3
            pcm = np.concatenate(c["pcm"][:max(1, keep)])
            if self.tone and not self._has_tone(c["tone"]):
                log.info("call end    LCN %-3d TG %-6s (no %.1f Hz tone - P25/interference, dropped)",
                         self.lcn, self.tg, self.tone)
                return
            self.writer.put(("fm", self, c["start"], pcm, ""))

    def _has_tone(self, chunks):
        if not chunks:
            return False
        x = np.concatenate(chunks)
        if len(x) < TONE_FS * 0.4:
            return False
        x = x - x.mean()
        nfft = 1 << int(np.ceil(np.log2(len(x) * 4)))
        P = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
        f = np.fft.rfftfreq(nfft, 1 / TONE_FS)
        band = P[(f > 60) & (f < 260)]
        peak = P[np.abs(f - self.tone) < 1.5].max()
        return 10 * np.log10(peak / (np.median(band) + 1e-12)) > 15


# --------------------------------------------------------------------------
class P25Decoder(threading.Thread):
    """Runs dsd-fme on a channel's tap; turns its per-call WAVs into jobs."""

    SRC = re.compile(r"\bSRC(?:_ID)?[:= ]+(\d+)", re.I)

    def __init__(self, chan, wav_root, writer, rtl=None):
        super().__init__(daemon=True, name=f"p25-{chan.tg}")
        self.chan, self.writer = chan, writer
        self.rtl = rtl          # dict -> dsd-fme drives this dongle itself (better demod)
        self.dir = os.path.join(wav_root, str(chan.tg))
        os.makedirs(self.dir, exist_ok=True)
        self.last_src = ""
        self.last_src_t = 0

    def run(self):
        threading.Thread(target=self._watch, daemon=True).start()
        while True:
            tcp = None
            if self.rtl:
                r = self.rtl
                try:
                    tcp = start_rtl_tcp(r["serial"], r["port"], self.chan.freq, 1_536_000,
                                        r.get("gain", 0), r.get("ppm", 0), r.get("bias", True))
                    connect_rtl_tcp(tcp, r["port"]).close()
                    time.sleep(0.5)
                except Exception as e:
                    log.warning("p25 %s: rtl_tcp failed: %s", self.chan.name, e)
                    if tcp:
                        tcp.kill()
                    time.sleep(3)
                    continue
                gain = int(round(float(r.get("gain", 0))))
                inp = (f"rtltcp:127.0.0.1:{r['port']}:{self.chan.freq / 1e6:.4f}M:{gain}:"
                       f"{r.get('ppm', 0)}:12")
            else:
                inp = f"tcp:127.0.0.1:{self.chan.tap_port}"
            cmd = ["dsd-fme", "-f1", "-i", inp, "-o", "null", "-7", self.dir, "-P"]
            log.info("p25 %s: %s", self.chan.name, " ".join(cmd))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", bufsize=1, cwd=self.dir)
            last_log = 0
            dbg = (open(os.path.join(os.environ["P25_LOG"], f"dsd_{self.chan.tg}.log"), "a")
                   if os.environ.get("P25_LOG") else None)
            for line in p.stdout:
                line = ANSI.sub("", line)
                if dbg:
                    dbg.write(f"{time.time():.1f} {line}")
                    dbg.flush()
                if "Sync:" in line and time.time() - last_log > 5:
                    last_log = time.time()
                    log.info("p25 %s: %s", self.chan.tg, line.strip()[:150])
                m = self.SRC.search(line)
                if m and m[1] != "0":
                    self.last_src, self.last_src_t = m[1], time.time()
            p.wait()
            if tcp:
                tcp.kill()
                tcp.wait()
            log.warning("p25 %s: dsd-fme exited (%s); restarting", self.chan.name, p.returncode)
            time.sleep(3)

    def _watch(self):
        while True:
            time.sleep(2)
            for path in glob.glob(os.path.join(self.dir, "**", "*.wav"), recursive=True):
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if time.time() - st.st_mtime < 4:
                    continue                      # still being written
                try:
                    self._job(path, st)
                except Exception:
                    log.exception("p25 wav %s", path)
                os.remove(path)

    def _job(self, path, st):
        with wave.open(path) as w:
            rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
            x = np.frombuffer(w.readframes(n), dtype=np.int16)
        if ch > 1:
            x = x.reshape(-1, ch)[:, 0]
        if not len(x):
            return
        g = np.gcd(rate, AUDIO_FS)
        pcm = sps.resample_poly(x.astype(np.float32), AUDIO_FS // g, rate // g)
        pcm = np.clip(pcm, -32767, 32767).astype(np.int16)
        m = self.SRC.search(os.path.basename(path)) or re.search(r"_SRC_?(\d+)", path, re.I)
        src = m[1] if m else (self.last_src if time.time() - self.last_src_t < 30 else "")
        start = st.st_mtime - len(x) / rate
        self.writer.put(("p25", self.chan, start, pcm, src))


# --------------------------------------------------------------------------
class Writer(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True, name="writer")
        self.cfg = cfg
        self.q = queue.Queue()
        self.tz = ZoneInfo(cfg.get("timezone", "UTC"))
        self.min_dur = cfg.get("min_duration_s", 0.8)
        self.calls = 0

    def put(self, item):
        self.q.put(item)

    def run(self):
        while True:
            item = self.q.get()
            try:
                self._write(*item)
            except Exception:
                log.exception("write failed")

    def _write(self, kind, ch, start, pcm, src):
        dur = len(pcm) / AUDIO_FS
        if dur < (self.cfg.get("min_duration_p25_s", 0.5) if kind == "p25" else self.min_dur):
            log.info("call end    LCN %-3d TG %-6s (%.1fs, too short - dropped)", ch.lcn, ch.tg, dur)
            return
        pcm = level_speech(pcm, AUDIO_FS)
        ts = datetime.fromtimestamp(start, self.tz)
        tg_part = safe(f"{ch.tg}-{ch.name}")
        name = f"{ts:%Y-%m-%d_%H%M%S}_{tg_part}_{safe(src + '-')}_{self.cfg['site_id']}_{self.cfg['system_id']}"
        d = os.path.join(self.cfg["output_dir"], f"{ts:%Y-%m-%d}", tg_part)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name + ".mp3")
        n = 2
        while os.path.exists(path):
            path = os.path.join(d, f"{name}-{n}.mp3")
            n += 1
        tmp = os.path.join(d, "." + os.path.basename(path) + ".part")
        subprocess.run(["lame", "--quiet", "-r", "-s", "16", "--bitwidth", "16", "--signed",
                        "--little-endian", "-m", "m", "-b", "32",
                        "--tt", f"{ch.tg} {ch.name}", "--ta", src or "unknown",
                        "--tl", self.cfg.get("album", "conventional"), "--ty", f"{ts:%Y}",
                        "--tc", f"{ch.freq / 1e6:.4f} MHz {kind.upper()}", "--add-id3v2",
                        "-", tmp], input=pcm.tobytes(), check=True)
        os.replace(tmp, path)
        self.calls += 1
        spool = self.cfg.get("spool_dir")
        if spool:
            job = f"{start:.3f}_{ch.lcn}"
            pcm.tofile(os.path.join(spool, job + ".pcm"))
            meta = {"start": start, "tg": ch.tg, "tg_label": ch.name, "lid": src,
                    "lid_label": "", "lcn": ch.lcn, "duration": dur, "rate": AUDIO_FS,
                    "mp3": path, "emergency": False, "source": f"local-{kind}",
                    "freq": ch.freq}
            with open(os.path.join(spool, "." + job + ".json"), "w") as f:
                json.dump(meta, f)
            os.replace(os.path.join(spool, "." + job + ".json"), os.path.join(spool, job + ".json"))
        log.info("call end    LCN %-3d TG %-6s %.1fs %s -> %s", ch.lcn, ch.tg, dur,
                 f"SRC {src}" if src else "", os.path.relpath(path, self.cfg["output_dir"]))


# --------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "local.json"))
    writer = Writer(cfg)
    writer.start()
    sdrs, decoders = [], []
    lcn = cfg.get("first_lcn", 100)
    tap_port = cfg.get("first_tap_port", 7360)
    for s in cfg["sdrs"]:
        if s.get("direct"):
            # whole dongle handed to dsd-fme on one P25 channel
            ch = Chan(dict(s["direct"], mode="p25"), lcn, writer, cfg)
            lcn += 1
            decoders.append(P25Decoder(ch, cfg["wav_dir"], writer, rtl={
                "serial": s["serial"], "port": s.get("port", 1250), "gain": s.get("gain", 0),
                "ppm": s.get("ppm", 0), "bias": s.get("bias_tee", cfg.get("bias_tee", True))}))
            log.info("sdr %s: direct P25 on %s %.4f MHz", s["serial"], ch.name, ch.freq / 1e6)
            continue
        chans = []
        for c in s["channels"]:
            ch = Chan(c, lcn, writer, cfg)
            lcn += 1
            if ch.digital:
                ch.tap_port = tap_port
                ch.tap = TapServer(tap_port)
                ch.tap.start()
                tap_port += 1
                decoders.append(P25Decoder(ch, cfg["wav_dir"], writer))
            chans.append(ch)
        sdr = Channelizer(s, chans, s.get("bias_tee", cfg.get("bias_tee", True)))
        sdrs.append(sdr)
        sdr.start()
    for d in decoders:
        d.start()

    def bye(*_):
        for s in sdrs:
            s.stop.set()
            if s.proc:
                s.proc.kill()
        os._exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    t = time.time()
    while True:
        time.sleep(1)
        if time.time() - t > cfg.get("stats_s", 300):
            t = time.time()
            fl = "  ".join(f"{c.tg}:{c.floor:.0f}/+{c.peak_db - c.floor:.0f}"
                           + (f"/q{getattr(c, 'quiet_peak', 0):.0f}" if c.analog else "")
                           for s in sdrs for c in s.chans if c.floor is not None)
            for s in sdrs:
                for c in s.chans:
                    c.peak_db = -99.0
                    c.quiet_peak = 0.0
            log.info("stats: calls %d; floor/peak-over-floor dB %s", writer.calls, fl)


if __name__ == "__main__":
    main()
