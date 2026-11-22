# Setup

These steps assume a dedicated Linux box or VM, the code in `/opt/radio`, and a system
user called `radio`. Adjust paths in the unit files if you put it elsewhere.

## 1. System packages

You need `rtl_tcp` and `librtlsdr` (from `rtl-sdr`; the [RTL-SDR Blog fork](https://github.com/rtlsdrblog/rtl-sdr-blog)
is recommended for their V3/V4 dongles), `lame`, `ffmpeg`, Python 3.10+ with `venv`,
and [`dsd-fme`](https://github.com/lwvmobile/dsd-fme) built from source.

Debian / Ubuntu:

```sh
sudo apt install rtl-sdr librtlsdr-dev lame ffmpeg python3-venv git
```

Arch:

```sh
sudo pacman -S rtl-sdr lame ffmpeg python git
```

Then build dsd-fme following its README and make sure `dsd-fme` is on the `PATH`
(the services call it by name).

## 2. Keep the kernel's DVB driver off the dongles

The kernel's DVB-T driver claims RTL2832U dongles as TV tuners. Blacklist it, then
unplug and replug the dongles (or reboot):

```sh
printf 'blacklist dvb_usb_rtl28xxu\nblacklist rtl2832\nblacklist rtl2830\n' | \
  sudo tee /etc/modprobe.d/blacklist-rtlsdr.conf
```

If you run this in a VM with USB passthrough, blacklist it **on the host too**. See
[hardware.md](hardware.md).

## 3. Give each dongle a unique serial

The recorders find dongles by serial number, so every dongle must have a different one.
With one dongle plugged in at a time:

```sh
rtl_eeprom -s 00000101     # then unplug / replug
rtl_test -t                # check: every dongle listed, no duplicate serials
```

## 4. Install the code

```sh
sudo git clone https://github.com/dbagwell00/linux-radio-stuff /opt/radio
sudo useradd --system --home-dir /opt/radio --shell /usr/sbin/nologin radio
sudo chown -R radio:radio /opt/radio
cd /opt/radio
sudo -u radio python3 -m venv venv
sudo -u radio venv/bin/pip install -r requirements.txt
```

The `radio` user needs access to the USB devices. Most distributions' rtl-sdr package
ships a udev rule that grants this to a group (often `plugdev`, sometimes uaccess
only). Check with `ls -l /dev/bus/usb/*/*` and add the user to that group, for example:

```sh
sudo usermod -aG plugdev radio
```

## 5. Find your system's frequencies

For an EDACS site you need the control channel and the frequency of every LCN (logical
channel number) **in LCN order**: EDACS grants name LCNs, not frequencies. RadioReference
lists LCN plans for most sites. Talkgroup names go in `labels.json` (decimal EDACS group
numbers; RadioReference lists them too).

A quick check that the control channel decodes:

```sh
rtl_tcp -a 127.0.0.1 -p 1239 -d 0 -f <cc_hz> -s 1536000 &
dsd-fme -fh -i rtltcp:127.0.0.1:1239:<cc MHz>M:0:0:24 -o null
```

You should see `Voice Group Channel Assignment` lines. If you see grants on LCNs you
haven't mapped, `edacs_rec.py` logs the strongest new carriers in the voice slices to
help you find them.

## 6. Configure

```sh
cd /opt/radio
sudo -u radio cp config/recorder.example.json      recorder.json
sudo -u radio cp config/labels.example.json        labels.json
sudo -u radio cp config/transcriber.example.json   transcriber.json
sudo -u radio cp config/conventional.example.json  conventional.json   # only if you use local_rec
sudo -u radio install -m 600 config/telegram.env.example telegram.env
```

**recorder.json**

- `lcns`: LCN number → frequency in Hz, including the control channel.
- `control`: the control channel's dongle serial and frequency.
- `voice_sdrs`: one entry per voice dongle, with a `center_hz` and the LCNs it covers.
  Every LCN must lie within about ±1.1 MHz of the centre. Avoid putting a channel
  exactly on the centre frequency (the DC spike); offset the centre by 50–100 kHz.
- `cc_tap` (optional): decode the control channel from a voice dongle's slice instead of
  a dedicated dongle. Set `primary: true` to use it and free up the control dongle.
  It catches somewhat fewer grants than a dedicated dongle, but grants repeat, so
  calls are rarely lost.
- `bias_tee`: `true` powers an inline LNA through the antenna port (see
  [hardware.md](hardware.md)). Leave it `false` unless you have something that needs it.

**transcriber.json**

- `area`: a short description of where the radio traffic is, used in the LLM prompts.
- `initial_prompt`: a sentence of place names, highways and jargon to help Whisper's
  spelling. Don't put bare acronyms in it (see the comment in `edacs_telegram.py`).
- `llm_url` / `llm_model`: your OpenAI-compatible server. Set `llm_disable_thinking`
  for Qwen3-style models that otherwise emit a long `<think>` block.
- `mode`: `incidents` (default), `digest` or `per_call`. Without an LLM, use `per_call`.

**telegram.env**: create a bot with @BotFather, add it to a private chat or group,
send a message there, and read the chat id from
`https://api.telegram.org/bot<token>/getUpdates`.

## 7. Run

Try each piece in the foreground first:

```sh
sudo -u radio venv/bin/python radio/edacs_rec.py recorder.json
sudo -u radio venv/bin/python radio/edacs_telegram.py transcriber.json
```

Then install the services:

```sh
sudo cp systemd/edacs-rec.service systemd/edacs-telegram.service /etc/systemd/system/
# sudo cp systemd/edacs-local.service /etc/systemd/system/   # conventional channels
sudo systemctl daemon-reload
sudo systemctl enable --now edacs-rec edacs-telegram
journalctl -fu edacs-rec -u edacs-telegram
```

## 8. Dashboard

```sh
cp config/radiowatch.example.json ~/radiowatch.json    # edit base_dir etc.
/opt/radio/venv/bin/python /opt/radio/radio/radiowatch.py ~/radiowatch.json
```

To read the services' logs, the user running it must be in the `systemd-journal` group
(or root).

## Tuning

- **Prompts**: replay real traffic without sending anything:
  `cd /opt/radio && sudo -u radio venv/bin/python tools/incident_dryrun.py 60 transcriber.json`
- **Whisper model**: `tools/whisper_bench.py transcripts/<day>.jsonl small.en medium.en`
  prints the real-time factor of each model on your own calls. Anything above about
  0.7 will fall behind at busy times.
- **Too many / too few messages**: `incident_min_severity_post` (1–3) and the severity
  guidance in `SYSTEM_PROMPT` in `incidents.py`.
- **Storage**: MP3s are 32 kbit/s mono (about 14 MB per hour of audio). Nothing is
  deleted automatically; add a cron job or systemd timer if you need retention, e.g.
  `find /opt/radio/recordings -type f -mtime +30 -delete`.
