# Hardware notes and gotchas

Lessons from running this around the clock on four RTL-SDR dongles.

## RF chain

A setup that works well for several dongles on one antenna:

```
antenna → powered splitter (e.g. a TV/VHF-UHF distribution amp) → per dongle:
          wideband LNA (bias-T powered) → FM broadcast band-stop filter → RTL-SDR
```

- The **FM band-stop filter** matters more than you'd think near a broadcast site: strong
  88–108 MHz signals overload the RTL tuner and raise the noise floor everywhere.
- A **wideband LNA at the dongle** made a large difference on 800 MHz in my case (about
  +30 dB on the control channel, which turned marginal decodes into solid ones). If
  you're close to the site you may not need it, and too much gain makes overload worse;
  start with tuner gain 0 (AGC) or a low fixed gain and watch the decode rate.
- Check the signal before blaming software: if the control channel isn't clearly
  above the noise in a waterfall (SDR++, GQRX), no decoder setting will fix it.

## Bias-T

An RTL-SDR only powers its antenna port while the process that opened the device keeps
bias-T enabled. `rtl_sdr` has no bias option and dsd-fme's `rtl:` input can't turn it on,
so the recorders open **every** dongle through `rtl_tcp -T` and connect to it over
localhost. If your LNA seems dead, check that `bias_tee` is `true` in the config.

Never enable bias-T on a dongle wired straight to a DC-shorted antenna or to equipment
that doesn't expect 4.5 V on the coax.

## Serial numbers vs. device indexes

`rtl_*` tools accept `-d <index or serial>`, but an all-digit serial like `00000001` is
read as **index 1**. The recorders avoid this by looking the serial up with
`rtlsdr_get_index_by_serial()` and passing the index. Give every dongle a unique serial
with `rtl_eeprom -s`.

## Frequency error and DC offset

Cheap dongles are often a few ppm off; at 850 MHz, 3 ppm is about 2.5 kHz. After FM
demodulation that error appears as a DC offset, which wrecks dsd-fme's FSK slicer.
The control-channel and P25 taps include a 5 Hz high-pass to remove it. For voice it
doesn't matter much, but you can set `ppm` per dongle if you've measured it (for
example with `kal` or against a known carrier). TCXO dongles avoid most of this.

## Running in a VM (Proxmox / KVM)

This runs fine in a VM with the dongles passed through, with a few caveats:

- **CPU type**: use `host` (Proxmox `cpu: host`). The default `kvm64` model hides AVX/AVX2,
  and faster-whisper / CTranslate2 and recent NumPy builds then crash or run very slowly.
- **USB controller**: use a USB 3 (xHCI) controller (Proxmox: set the OS type to Linux
  so it picks `qemu-xhci`). Four dongles at 2.4 MS/s each is a lot of USB traffic.
- **Pass dongles through by USB port, not by vendor:product ID.** All RTL-SDRs share
  the same ID (`0bda:2838`), so ID-based passthrough can't tell them apart and may not
  reattach them after a reset. Port-based passthrough reattaches the same physical
  dongle to the same VM slot.
- **Blacklist the DVB driver on the host as well as in the guest.** Otherwise, when a
  dongle resets, the host kernel grabs it before QEMU can hand it back, and the guest
  sees missing dongles or several with the same serial.
- Symptom of the above: recordings stop, and `rtl_test -t` in the guest shows no
  devices or duplicate serials while the host still shows them attached.

## CPU budget

Rough numbers from a modern 16-core desktop CPU:

- Each 2.4 MS/s channelizer: about half a core.
- Whisper `small.en`, int8, 3 threads: real-time factor about 0.5.
- `medium.en` about 1.5, `large-v3-turbo` about 2.7 on the same threads: more accurate,
  but too slow to keep up with a busy trunked system on CPU alone. A GPU changes this.
