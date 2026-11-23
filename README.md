# linux-radio-stuff

A small Linux pipeline that records public-safety radio with cheap RTL-SDR dongles,
transcribes every call with Whisper, and uses a local LLM to turn the traffic into
readable incident updates in a private Telegram chat.

```
                     ┌─────────────────────┐
 RTL-SDR ─ rtl_tcp ─►│ edacs_rec.py        │  EDACS trunked system:
 RTL-SDR ─ rtl_tcp ─►│  dsd-fme on the CC  │  dsd-fme reads grants from the control channel,
                     │  FFT channelizer    │  an FFT channelizer demodulates every voice LCN at once
                     └─────────┬───────────┘
                               │ MP3 per call + spool/<id>.pcm/.json
 RTL-SDR ─ rtl_tcp ─►┌─────────┴───────────┐
                     │ local_rec.py        │  conventional channels:
                     │  analog FM + P25    │  power/noise squelch for FM, dsd-fme for P25
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │ edacs_telegram.py   │  faster-whisper (CPU) → transcripts/*.jsonl
                     │  + incidents.py     │  OpenAI-compatible LLM threads calls into incidents
                     └─────────┬───────────┘
                               ▼
                     Telegram: one message per incident, edited in place as it develops,
                     optionally with the incident's calls joined into one playable clip
```

`radiowatch.py` is a terminal dashboard (Textual) for the whole thing: live transcripts,
channels on the air, Whisper latency, LLM passes, open incidents and service health.

## What's in here

| Path | What it does |
| --- | --- |
| `radio/edacs_rec.py` | EDACS (analog voice) multi-call recorder. One or more SDRs each stream a 2.4 MHz slice; an overlap-save FFT channelizer pulls out every voice LCN continuously; control-channel grants from `dsd-fme` gate the audio into per-call MP3s. The control channel can have its own dongle or be carved out of a voice dongle's slice (`cc_tap`). |
| `radio/local_rec.py` | Conventional channels: analog FM with a power + noise ("quieting") squelch and optional CTCSS gate, and P25 via `dsd-fme`, either fed from the channelizer or given a whole dongle. |
| `radio/edacs_telegram.py` | Watches the spool, transcribes with faster-whisper, filters Whisper hallucinations and repetition loops, logs transcripts, and posts to Telegram in one of three modes: `incidents`, `digest`, `per_call`. Keyword / emergency-flag alerts go out immediately. |
| `radio/incidents.py` | Every couple of minutes, sends new transcripts plus open incidents to the LLM, which assigns each call to an incident and rates its severity. Notable incidents get their own Telegram message (text, or audio with a caption); minor ones go into an hourly silent round-up. Falls back to a plain digest if the LLM is down. |
| `radio/geomap.py` | Optional incident maps: geocodes the LLM's location (US Census geocoder, OpenStreetMap Nominatim, intersections from street geometry) and draws a close-up map plus one showing the incident relative to your home, posted as a reply album. Only for incidents within a set radius. |
| `radio/audio_util.py` | Speech-level normalisation (RMS of active frames, soft limiter). |
| `radio/radiowatch.py` | Terminal dashboard. |
| `tools/` | `incident_dryrun.py` / `digest_dryrun.py` replay saved transcripts through the LLM and print instead of sending (good for prompt tuning). `whisper_bench.py` compares Whisper models on your own recordings. |
| `config/` | Example configs. All frequencies, serials and IDs in them are placeholders. |
| `systemd/` | Hardened unit files for the three services. |
| `docs/` | [Setup](docs/setup.md) and [hardware notes and gotchas](docs/hardware.md). |

## Requirements

- Linux with 1+ RTL-SDR dongles (each covers about 2 MHz; a typical EDACS site needs 1–3).
- [`dsd-fme`](https://github.com/lwvmobile/dsd-fme), `rtl_tcp` / `librtlsdr`, `lame`, `ffmpeg`.
- Python 3.10+ with `numpy`, `scipy`, `faster-whisper` (and `textual` for the dashboard).
- Any OpenAI-compatible chat-completions server for the summaries: vLLM, llama.cpp
  server, Ollama, LM Studio, or a hosted API. A mid-size instruction model that can
  produce JSON works; the default prompts were tuned on a ~30B model.
- A Telegram bot token and a private chat for it to post in.

CPU is enough for Whisper `small.en` (int8, real-time factor about 0.5 on 3 threads of
a modern desktop CPU). Larger models are more accurate but may not keep up with a busy
system; benchmark with `tools/whisper_bench.py`.

## Quick start

See [docs/setup.md](docs/setup.md) for the full walkthrough. In short:

```sh
sudo git clone https://github.com/dbagwell00/linux-radio-stuff /opt/radio
sudo useradd --system --home-dir /opt/radio --shell /usr/sbin/nologin radio
sudo chown -R radio:radio /opt/radio && cd /opt/radio
sudo -u radio python3 -m venv venv && sudo -u radio venv/bin/pip install -r requirements.txt
sudo -u radio cp config/recorder.example.json recorder.json        # edit: your site's LCNs, your serials
sudo -u radio cp config/transcriber.example.json transcriber.json  # edit: area, prompt, LLM
sudo -u radio cp config/labels.example.json labels.json            # edit: your talkgroup names
sudo -u radio install -m 600 config/telegram.env.example telegram.env  # add bot token + chat id
sudo cp systemd/edacs-rec.service systemd/edacs-telegram.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now edacs-rec edacs-telegram
```

## Output

Recordings use the UniTrunker / Trunking Recorder naming so existing tools can read them:

```
recordings/YYYY-MM-DD/<TG>-<label>/YYYY-MM-DD_HHMMSS_<TG>-<label>_<LID>-<lid label>_<site>_<system>.mp3
```

Transcripts are appended to `transcripts/YYYY-MM-DD.jsonl`, one JSON object per call.

## Please use this responsibly

- **Know your local laws.** Listening to unencrypted public-safety radio is legal in many
  places but not everywhere, and some places restrict scanners in vehicles or forbid
  using what you hear to commit a crime. Recording, transcribing and *redistributing*
  radio traffic can be restricted even where listening is fine (in the US, see
  47 U.S.C. § 605; other countries have their own rules). This project is meant for
  personal awareness: keep the Telegram chat private.
- Radio traffic often includes names, addresses, medical details and plate/licence
  lookups of real people. Treat the recordings, transcripts and messages accordingly:
  keep them private, don't post them publicly, and don't keep them longer than you need.
- This software does not and will not decrypt encrypted traffic.
- Transcripts and LLM summaries are frequently wrong. Never rely on them for safety
  decisions; follow official alerts and call emergency services when needed.

## Security notes

- `rtl_tcp` and the internal audio taps listen on `127.0.0.1` only.
- Secrets live in `telegram.env` (mode 600; the transcriber warns if it isn't) or in
  environment variables. Real configs and runtime data are in `.gitignore`.
- Anyone with a radio can put words into the transcripts, so treat the transcripts and
  LLM output as untrusted: the LLM's reply is only parsed as JSON, and every field is
  HTML-escaped before it goes to Telegram. The LLM has no tools and cannot act on anything.
- If your LLM server is on another machine, use HTTPS or keep it on a trusted network,
  and don't expose the server to the internet without authentication.
- The systemd units run as an unprivileged `radio` user with a read-only system and
  write access only to `/opt/radio`.

## License

MIT, see [LICENSE](LICENSE).
