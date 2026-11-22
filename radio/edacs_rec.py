#!/usr/bin/env python3
"""
edacs_rec - multi-call EDACS (analog voice) trunk recorder.

  * dsd-fme decodes the control channel and prints channel grants. The CC is
    either on its own RTL-SDR or (cc_tap) carved out of a voice SDR's slice.
  * One or more "voice" RTL-SDRs stream a fixed 2.4 MHz slice each; an
    overlap-save FFT channelizer pulls every configured voice LCN out of
    that slice continuously and FM-demodulates it.
  * Grants gate the audio into per-transmission MP3s, named like the old
    UniTrunker/Trunking Recorder files so existing tools can read them:
      <out>/YYYY-MM-DD/<TG>-<label>/YYYY-MM-DD_HHMMSS_<TG>-<label>_<LID>-_<site>_<sys>.mp3
"""
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
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import level_speech  # noqa: E402,F401  (re-exported for local_rec)

log = logging.getLogger("edacs_rec")

FS = 2_400_000          # SDR sample rate
DEC = 75                # FS / DEC = 32 kHz channel rate
CH_FS = FS // DEC       # 32000
N = 76_800              # FFT size (DEC * 1024)
K = N // DEC            # 1024 output bins per channel
OVL = 1_200             # overlap (filter taps - 1), multiple of DEC
STEP = N - OVL          # new input samples per block (75600)
DISCARD = OVL // DEC    # 16 decimated samples discarded per block
AUDIO_FS = 16_000

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GRANT = re.compile(
    r"Voice Group Channel (Assignment|Update) :: (\w+) Group \[(\d+)\]"
    r"(?: LID \[(\d+)\])? LCN \[(\d+)\]")


# --------------------------------------------------------------------------
# RTL-SDR access via rtl_tcp. rtl_tcp -T is the only stock tool that holds the
# bias-T on for as long as the device is open (needed to power an inline LNA);
# rtl_sdr has no bias option and dsd-fme's rtl: input can't enable it.
# --------------------------------------------------------------------------
def rtl_index(serial):
    """Device index for a serial, without opening any device.

    Don't pass serials straight to rtl_* -d: they treat an all-digit serial such
    as "00000001" as an index."""
    import ctypes
    lib = ctypes.CDLL("librtlsdr.so")
    lib.rtlsdr_get_index_by_serial.argtypes = [ctypes.c_char_p]
    idx = lib.rtlsdr_get_index_by_serial(serial.encode())
    if idx < 0:
        raise IOError(f"RTL-SDR serial {serial} not found ({idx})")
    return idx


def start_rtl_tcp(serial, port, freq, rate, gain, ppm, bias):
    cmd = ["rtl_tcp", "-a", "127.0.0.1", "-p", str(port), "-d", str(rtl_index(serial)),
           "-f", str(int(freq)), "-s", str(rate), "-g", str(gain), "-P", str(ppm)]
    if bias:
        cmd.append("-T")
    log.info("rtl_tcp for %s: %s", serial, " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def connect_rtl_tcp(proc, port, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise IOError(f"rtl_tcp exited ({proc.returncode})")
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            hdr = b""
            while len(hdr) < 12:
                c = s.recv(12 - len(hdr))
                if not c:
                    raise IOError("rtl_tcp closed during header")
                hdr += c
            if not hdr.startswith(b"RTL0"):
                raise IOError(f"bad rtl_tcp header {hdr!r}")
            s.settimeout(10)
            return s
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.3)
    raise IOError("rtl_tcp did not come up")


class TapServer(threading.Thread):
    """Serves int16 48 kHz discriminator audio to one TCP client (dsd-fme -i tcp:...)."""

    def __init__(self, port):
        super().__init__(daemon=True, name=f"tap-{port}")
        self.port = port
        self.client = None
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(1)

    def run(self):
        while True:
            c, _ = self.srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c.settimeout(2)
            old, self.client = self.client, c
            if old:
                old.close()

    def send(self, pcm):
        c = self.client
        if c is None:
            return
        try:
            c.sendall(pcm.tobytes())
        except OSError:
            self.client = None
            c.close()


TAP_DEC = 50                          # 2.4 MS/s / 50 = 48 kHz for dsd-fme
TAP_K = N // TAP_DEC
TAP_DISCARD = OVL // TAP_DEC


class Tap:
    """A 48 kHz raw-discriminator side channel (e.g. the control channel)."""

    def __init__(self, freq, center, server, cutoff=12_000):
        self.freq, self.server = freq, server
        self.k0 = int(round((freq - center) / (FS / N)))
        self.idx = (self.k0 + np.fft.fftfreq(TAP_K, 1 / TAP_K).astype(int)) % N
        h = sps.firwin(OVL + 1, cutoff, fs=FS)
        self.Hf = np.fft.fft(h, N).astype(np.complex64)[
            (np.fft.fftfreq(TAP_K, 1 / TAP_K).astype(int)) % N]
        self.prev = np.complex64(1)
        self.scale = (FS / TAP_DEC) / (2 * np.pi) * 3.0      # Hz -> int16 counts
        # tuner ppm error shows up as a DC offset after FM demod (a few kHz at 800 MHz);
        # dsd-fme's FSK slicers hate that, so strip it with a 5 Hz high-pass
        self.hp_b, self.hp_a = sps.butter(1, 5, "highpass", fs=FS / TAP_DEC)
        self.hp_zi = np.zeros(1)

    def process(self, X, ph):
        y = np.fft.ifft(X[self.idx] * self.Hf)[TAP_DISCARD:]
        y = (y * np.exp(-2j * np.pi * self.k0 * ph / N)).astype(np.complex64)
        z = np.concatenate(([self.prev], y))
        self.prev = y[-1]
        d = np.angle(z[1:] * np.conj(z[:-1])) * self.scale
        d, self.hp_zi = sps.lfilter(self.hp_b, self.hp_a, d, zi=self.hp_zi)
        pcm = np.clip(d, -32767, 32767).astype(np.int16)
        self.server.send(pcm)
        if os.environ.get("TAP_REC"):
            if not hasattr(self, "_rec"):
                self._rec = open(os.environ["TAP_REC"], "ab")
            self._rec.write(pcm.tobytes())


# --------------------------------------------------------------------------
# Voice SDR + channelizer
# --------------------------------------------------------------------------
class VoiceSDR(threading.Thread):
    """Streams IQ from rtl_tcp and emits demodulated audio per LCN."""

    def __init__(self, serial, center_hz, gain, ppm, channels, sink, deemph_tau,
                 port=1240, bias=True):
        super().__init__(daemon=True, name=f"sdr-{serial}")
        self.serial, self.center, self.gain, self.ppm = serial, center_hz, gain, ppm
        self.port, self.bias = port, bias
        self.taps = []
        self.sink = sink                         # callable(lcn, audio_int16)
        self.channels = {}                       # lcn -> state
        h = sps.firwin(OVL + 1, 10_000, fs=FS)   # +/-10 kHz channel filter
        self.Hf = np.fft.fft(h, N).astype(np.complex64)
        self.hidx = (np.fft.fftfreq(K, 1 / K).astype(int)) % N
        for lcn, freq in channels.items():
            off = freq - center_hz
            if abs(off) > FS / 2 * 0.92:
                raise ValueError(f"LCN {lcn} {freq} outside window of {serial}")
            k0 = int(round(off / (FS / N)))
            self.channels[lcn] = ChannelDemod(lcn, k0, deemph_tau)
        self.block = 0
        self.spec_lock = threading.Lock()
        self.spec_avg = None                    # long-term power per bin
        self.spec_now = None                    # short-term power per bin
        self.proc = None
        self.stop = threading.Event()

    def bin_freqs(self):
        return self.center + np.fft.fftfreq(N, 1 / FS)

    def run(self):
        while not self.stop.is_set():
            try:
                self._stream()
            except Exception:
                log.exception("voice sdr %s crashed", self.serial)
            if not self.stop.is_set():
                log.warning("voice sdr %s restarting in 3s", self.serial)
                time.sleep(3)

    def _stream(self):
        log.info("voice sdr %s: center %.4f MHz  LCNs %s  bias-T %s", self.serial,
                 self.center / 1e6, sorted(self.channels), self.bias)
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
                new = iq[0::2] + 1j * iq[1::2]
                buf[:OVL] = buf[STEP:]
                buf[OVL:] = new
                X = np.fft.fft(buf)
                self._spectrum(X)
                for tp in self.taps:
                    tp.process(X, (self.block * STEP) % N)
                for ch in self.channels.values():
                    idx = (ch.k0 + np.fft.fftfreq(K, 1 / K).astype(int)) % N
                    y = np.fft.ifft(X[idx] * self.Hf[self.hidx])[DISCARD:]
                    # phase continuity across blocks for the bin shift
                    y *= np.exp(-2j * np.pi * ch.k0 * ((self.block * STEP) % N) / N)
                    self.sink(ch.lcn, ch.process(y.astype(np.complex64)))
                self.block += 1
        finally:
            if sock:
                sock.close()
            self.proc.kill()
            self.proc.wait()

    def _spectrum(self, X):
        p = np.abs(X) ** 2
        with self.spec_lock:
            if self.spec_avg is None:
                self.spec_avg = p.copy()
                self.spec_now = p.copy()
            else:
                self.spec_avg = 0.998 * self.spec_avg + 0.002 * p
                self.spec_now = 0.7 * self.spec_now + 0.3 * p

    def candidates(self, exclude_hz, step=12_500, top=3):
        """Channel frequencies whose power jumped most vs. the long-term average."""
        with self.spec_lock:
            if self.spec_avg is None:
                return []
            avg, now = self.spec_avg.copy(), self.spec_now.copy()
        freqs = self.bin_freqs()
        lo, hi = self.center - FS * 0.45, self.center + FS * 0.45
        out = []
        for fc in np.arange(np.ceil(lo / step) * step + 6_250, hi, step):
            if any(abs(fc - e) < 9_000 for e in exclude_hz):
                continue
            m = np.abs(freqs - fc) < 5_000
            if m.any():
                out.append((10 * np.log10(now[m].sum() / avg[m].sum()), fc))
        out.sort(reverse=True)
        return out[:top]


class ChannelDemod:
    def __init__(self, lcn, k0, tau):
        self.lcn, self.k0 = lcn, k0
        self.prev = np.complex64(1)
        # de-emphasis (one-pole) at 32 kHz, then 300-3400 Hz band-pass
        a = np.exp(-1.0 / (CH_FS * tau)) if tau > 0 else 0.0
        self.de_b, self.de_a = [1 - a], [1, -a]
        self.de_zi = np.zeros(1)
        self.bp = sps.butter(4, [300, 3400], btype="bandpass", fs=CH_FS, output="sos")
        self.bp_zi = sps.sosfilt_zi(self.bp) * 0
        self.gain = CH_FS / (2 * np.pi * 5_000)   # +/-5 kHz deviation -> +/-1.0

    def process(self, y):
        z = np.concatenate(([self.prev], y))
        self.prev = y[-1]
        d = np.angle(z[1:] * np.conj(z[:-1])).astype(np.float32) * self.gain
        d, self.de_zi = sps.lfilter(self.de_b, self.de_a, d, zi=self.de_zi)
        d, self.bp_zi = sps.sosfilt(self.bp, d, zi=self.bp_zi)
        d = d[::CH_FS // AUDIO_FS]
        return np.clip(d * 8000, -32767, 32767).astype(np.int16)


# --------------------------------------------------------------------------
# Call tracking + file output
# --------------------------------------------------------------------------
class Call:
    def __init__(self, tg, lid, lcn, now, preroll):
        self.tg, self.lid, self.lcn = tg, lid, lcn
        self.start = now
        self.last_seen = now
        self.chunks = list(preroll)
        self.emergency = False


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.get("timezone", "UTC"))
        self.out = cfg["output_dir"]
        self.site, self.system = cfg["site_id"], cfg["system_id"]
        self.hang = cfg.get("hangtime_s", 1.5)
        self.min_dur = cfg.get("min_duration_s", 0.5)
        self.max_dur = cfg.get("max_duration_s", 180)
        self.lcn_freq = {int(k): v for k, v in cfg["lcns"].items()}
        labels = {}
        if cfg.get("labels_file") and os.path.exists(cfg["labels_file"]):
            labels = json.load(open(cfg["labels_file"]))
        self.tg_labels = labels.get("groups", {})
        self.rid_labels = labels.get("users", {})
        self.lock = threading.Lock()
        self.active = {}                                    # lcn -> Call
        self.preroll = {l: deque(maxlen=8) for l in self.lcn_freq}   # ~0.25 s
        self.encq = queue.Queue()
        self.sdrs = []
        self.unknown_lcn_seen = {}
        self.stats = {"calls": 0, "grants": 0, "unknown_lcn": 0}

    # ---- audio in (from SDR threads)
    def audio(self, lcn, pcm):
        with self.lock:
            c = self.active.get(lcn)
            if c is not None:
                c.chunks.append(pcm)
            else:
                self.preroll[lcn].append(pcm)

    # ---- grants in (from CC thread)
    def grant(self, kind, mode, tg, lid, lcn, emergency=False):
        now = time.time()
        self.stats["grants"] += 1
        if lcn not in self.lcn_freq:
            self._unknown_lcn(lcn, tg, now)
            return
        with self.lock:
            c = self.active.get(lcn)
            if kind == "Assignment":
                if c and c.tg == tg and (c.lid == lid or c.lid is None) and now - c.start < 3:
                    c.lid = c.lid or lid           # duplicate assignment burst
                    c.last_seen = now
                    c.emergency |= emergency
                    return
                if c:
                    self._close(lcn)
                self._open(tg, lid, lcn, now, mode)
                self.active[lcn].emergency = emergency
            else:                                  # Update (call in progress)
                if c and c.tg == tg:
                    c.last_seen = now
                    c.emergency |= emergency
                    return
                if c:
                    self._close(lcn)
                self._open(tg, None, lcn, now, mode)
                self.active[lcn].emergency = emergency

    def _open(self, tg, lid, lcn, now, mode):
        self.active[lcn] = Call(tg, lid, lcn, now, self.preroll[lcn])
        self.preroll[lcn].clear()
        log.info("call start  LCN %-2d TG %-5s %-40s LID %s %s", lcn, tg,
                 self.tg_labels.get(str(tg), ""), lid or "-",
                 "" if mode == "Analog" else f"[{mode}]")

    def _close(self, lcn):
        c = self.active.pop(lcn)
        # drop the audio that came after the channel went quiet
        tail = max(0.0, time.time() - (c.last_seen + 0.6))
        pcm = np.concatenate(c.chunks) if c.chunks else np.zeros(0, np.int16)
        cut = int(tail * AUDIO_FS)
        if cut and cut < len(pcm):
            pcm = pcm[:-cut]
        self.encq.put((c, pcm))

    def tick(self):
        now = time.time()
        with self.lock:
            for lcn, c in list(self.active.items()):
                if now - c.last_seen > self.hang or now - c.start > self.max_dur:
                    self._close(lcn)

    def _unknown_lcn(self, lcn, tg, now):
        self.stats["unknown_lcn"] += 1
        if now - self.unknown_lcn_seen.get(lcn, 0) < 20:
            return
        self.unknown_lcn_seen[lcn] = now

        def probe():
            time.sleep(0.8)
            known = list(self.lcn_freq.values())
            cands = []
            for s in self.sdrs:
                cands += s.candidates(known)
            cands.sort(reverse=True)
            log.warning("grant on unmapped LCN %d (TG %s); strongest new carriers: %s",
                        lcn, tg, ", ".join(f"{f/1e6:.4f} MHz +{db:.1f} dB" for db, f in cands[:4]))
        threading.Thread(target=probe, daemon=True).start()

    # ---- encoder thread
    def encoder(self):
        while True:
            c, pcm = self.encq.get()
            try:
                self._write(c, pcm)
            except Exception:
                log.exception("failed writing call")

    def _write(self, c, pcm):
        dur = len(pcm) / AUDIO_FS
        if dur < self.min_dur:
            log.info("call end    LCN %-2d TG %-5s (%.1fs, too short - dropped)", c.lcn, c.tg, dur)
            return
        pcm = level_speech(pcm, AUDIO_FS)         # speech-level normalisation + limiter
        ts = datetime.fromtimestamp(c.start, self.tz)
        tgl = self.tg_labels.get(str(c.tg), "")
        rid = c.lid or ""
        ridl = self.rid_labels.get(rid, "")
        tg_part = safe(f"{c.tg}-{tgl}")
        name = (f"{ts:%Y-%m-%d_%H%M%S}_{tg_part}_{safe(f'{rid}-{ridl}')}"
                f"_{self.site}_{self.system}")
        d = os.path.join(self.out, f"{ts:%Y-%m-%d}", tg_part)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name + ".mp3")
        n = 2
        while os.path.exists(path):
            path = os.path.join(d, f"{name}-{n}.mp3")
            n += 1
        tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".part")
        cmd = ["lame", "--quiet", "-r", "-s", str(AUDIO_FS // 1000), "--bitwidth", "16",
               "--signed", "--little-endian", "-m", "m", "-b", "32",
               "--tt", f"{c.tg} {tgl}".strip(), "--ta", rid or "unknown",
               "--tl", f"EDACS site {self.site}", "--ty", f"{ts:%Y}",
               "--tc", f"LCN {c.lcn} {self.lcn_freq[c.lcn]/1e6:.4f} MHz",
               "--add-id3v2", "-", tmp]
        subprocess.run(cmd, input=pcm.tobytes(), check=True)
        os.replace(tmp, path)
        self.stats["calls"] += 1
        spool = self.cfg.get("spool_dir")
        if spool:
            # hand the call to the transcriber: raw PCM first, job JSON last
            os.makedirs(spool, exist_ok=True)
            job = f"{c.start:.3f}_{c.lcn}"
            pcm.tofile(os.path.join(spool, job + ".pcm"))
            meta = {"start": c.start, "tg": c.tg, "tg_label": tgl, "lid": rid,
                    "lid_label": ridl, "lcn": c.lcn, "duration": dur,
                    "rate": AUDIO_FS, "mp3": path, "emergency": c.emergency}
            with open(os.path.join(spool, "." + job + ".json"), "w") as f:
                json.dump(meta, f)
            os.replace(os.path.join(spool, "." + job + ".json"),
                       os.path.join(spool, job + ".json"))
        log.info("call end    LCN %-2d TG %-5s %.1fs -> %s", c.lcn, c.tg, dur,
                 os.path.relpath(path, self.out))


def safe(s):
    return re.sub(r'[\\/:*?"<>|]', " ", s).strip()


# --------------------------------------------------------------------------
# Control channel (dsd-fme)
# --------------------------------------------------------------------------
def control_channel(cfg, rec, stop):
    cc = cfg["control"]
    port = cc.get("port", 1239)
    while not stop.is_set():
        try:
            tcp = start_rtl_tcp(cc["serial"], port, cc["freq"], 1_536_000, cc.get("gain", 0),
                                cc.get("ppm", 0), cfg.get("bias_tee", True))
            connect_rtl_tcp(tcp, port).close()      # wait until it's listening
            time.sleep(0.5)
        except Exception as e:
            log.warning("control channel rtl_tcp failed: %s; retrying in 3s", e)
            time.sleep(3)
            continue
        cmd = ["dsd-fme"] + cc.get("dsd_args", ["-fh"]) + [
            "-i", f"rtltcp:127.0.0.1:{port}:{cc['freq']/1e6:.4f}M:{cc.get('gain', 0)}:"
                  f"{cc.get('ppm', 0)}:24", "-o", "null"]
        log.info("control channel: %s", " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", bufsize=1)
        last_sync = time.time()
        try:
            for line in p.stdout:
                if stop.is_set():
                    break
                line = ANSI.sub("", line)
                if "EDACS" in line:
                    last_sync = time.time()
                m = GRANT.search(line)
                if m:
                    kind, mode, tg, lid, lcn = m.groups()
                    rec.grant(kind, mode, int(tg), str(int(lid)) if lid else None, int(lcn),
                              "[EMERGENCY]" in line)
        finally:
            p.kill()
            p.wait()
            tcp.kill()
            tcp.wait()
        log.warning("dsd-fme exited (last sync %.0fs ago); restarting in 3s",
                    time.time() - last_sync)
        time.sleep(3)


def tap_control_channel(cfg, rec, stop, primary):
    """dsd-fme on the control channel carved out of a voice SDR's slice."""
    t = cfg["cc_tap"]
    rec.shadow = {"grants": 0, "syncs": 0}
    while not stop.is_set():
        cmd = ["dsd-fme"] + cfg["control"].get("dsd_args", ["-fh"]) + [
            "-i", f"tcp:127.0.0.1:{t['port']}", "-o", "null"]
        log.info("control channel (%s, from %s): %s", "primary" if primary else "shadow",
                 t["serial"], " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", bufsize=1)
        try:
            for line in p.stdout:
                if stop.is_set():
                    break
                line = ANSI.sub("", line)
                if "EDACS" in line:
                    rec.shadow["syncs"] += 1
                m = GRANT.search(line)
                if m:
                    rec.shadow["grants"] += 1
                    if primary:
                        kind, mode, tg, lid, lcn = m.groups()
                        rec.grant(kind, mode, int(tg), str(int(lid)) if lid else None,
                                  int(lcn), "[EMERGENCY]" in line)
        finally:
            p.kill()
            p.wait()
        log.warning("tap dsd-fme exited; restarting in 3s")
        time.sleep(3)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "config.json"))
    rec = Recorder(cfg)
    stop = threading.Event()
    for i, s in enumerate(cfg["voice_sdrs"]):
        chans = {int(l): rec.lcn_freq[int(l)] for l in s["lcns"]}
        sdr = VoiceSDR(s["serial"], s["center_hz"], s.get("gain", 0), s.get("ppm", 0),
                       chans, rec.audio, cfg.get("deemphasis_tau_s", 750e-6),
                       port=s.get("port", 1240 + i), bias=cfg.get("bias_tee", True))
        rec.sdrs.append(sdr)
        sdr.start()
    threading.Thread(target=rec.encoder, daemon=True, name="encoder").start()
    t = cfg.get("cc_tap")
    tap_primary = bool(t and t.get("primary"))
    if t:
        srv = TapServer(t["port"])
        srv.start()
        sdr = next(s for s in rec.sdrs if s.serial == t["serial"])
        sdr.taps.append(Tap(cfg["control"]["freq"], sdr.center, srv))
        threading.Thread(target=tap_control_channel, args=(cfg, rec, stop, tap_primary),
                         daemon=True, name="cc-tap").start()
    if not tap_primary:
        threading.Thread(target=control_channel, args=(cfg, rec, stop), daemon=True,
                         name="cc").start()

    def bye(*_):
        stop.set()
        for s in rec.sdrs:
            s.stop.set()
            if s.proc:
                s.proc.kill()
        sys.exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    last_stat = time.time()
    while True:
        time.sleep(0.1)
        rec.tick()
        if time.time() - last_stat > 300:
            last_stat = time.time()
            log.info("stats: %s, active %s%s", rec.stats, sorted(rec.active),
                     f", cc-tap {rec.shadow}" if getattr(rec, "shadow", None) else "")


if __name__ == "__main__":
    main()
