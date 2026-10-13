# DeepFilterNet3

Two engines with the same structure: DeepFilterNet3 (GRU hidden size 256,
weights 2.7 MB, two frames of lookahead) and DeepFilterNet3-LL (hidden size 512,
weights 10.7 MB, no lookahead). DeepFilterNet3 is the default; its weights fit
the 3 MB L3 of the oldest supported CPUs.

## Signal path

The engine takes 48 kHz mono in 480-sample hops. An STFT (FFT 960) feeds ERB
and complex features to the encoder and its GRUs; the decoders produce an ERB
mask and a deep filter for the lowest bins, and an iSTFT returns the samples.
`deepfilternet3/dfn3-plugin` holds this pipeline, the LADSPA plugin and the
voice gate; `dfn3-ladspa` and `dfn3ll-ladspa` add each model's network and
weights.

Only the GRU projection matrices are quantized. They hold about 94% of the
multiply-adds and are the part limited by memory bandwidth. Per output row:

```text
scale[i] = max(|w[i,:]|) / 127
q[i,j]   = round(w[i,j] / scale[i])            in [-127, 127]
y[i]     = (Σj q[i,j] · xq[j]) · scale[i] · xscale
```

The activation vector `x` is quantized to int16 per call. The sum is an exact
`i32` (`pmaddwd`), so SSE4.1, AVX and AVX2 return the same bits as the scalar
loop. Convolutions, grouped linears, biases and gates stay in f32. The committed
blobs were produced by `deepfilternet3/scripts/quantize_int8.py`
([details](../deepfilternet3/scripts/README.md)).

int8 activations (W8A8) were measured and rejected. On the i3 (AVX, no AVX2,
FMA or VNNI) the only int8 × int8 instruction, `pmaddubsw`, saturates at
2·255·127, so an exact A8 product would need 7-bit weights or a widening back to
int16, which is the current kernel. The i5 could run A8 exactly with AVX-VNNI,
but it already spends under 2% of a core on DeepFilterNet3.

## Plugin behavior

- Rejects sample rates other than 48 kHz instead of processing them wrongly.
- Buffers any host block size into 480-sample frames; the output does not
  depend on the block size (tested from 1 to 65536 samples, with the voice gate
  on and off).
- Replaces NaN/Inf input samples with silence and clamps samples beyond ±8
  (+18 dBFS), so a corrupt sample cannot poison the network state. A
  non-finite control value counts as unconnected.
- Skips the network after 2 s of digital silence (mean square below 1e-7,
  about -70 dBFS) and outputs zeros until the input returns, so a muted
  microphone costs almost nothing. The first 2 s still run the network to let
  its recurrent state settle where continuous processing would leave it.
  Freezing it after five frames changed the first 1.5 s of speech after a
  mute, by up to -1.3 dB SI-SDR on DeepFilterNet3 at 20 dB SNR.
- Reports its current delay in samples on the `latency` output port, which
  PipeWire's filter-chain adds to the node's latency.
- Shares the decoded weights between instances in one process and allocates
  nothing in `run()`.
- Does not set `LADSPA_PROPERTY_HARD_RT_CAPABLE`: silence and stage skipping
  make the cost of a frame depend on the signal.

## Controls

Run `analyseplugin libdfn3_ladspa.so` for ranges and defaults. Both engines have
the same ports; new ports are only appended, so existing indices stay valid.

| Port | Default | Effect |
|---|---:|---|
| Attenuation Limit (dB) | 100 | Maximum noise reduction. 0 passes the input through. |
| Min SNR gate (dB) | -10 | Below this estimated SNR a frame is treated as noise only. |
| Skip-all SNR (dB) | 40 | Above this SNR the enhancement stages are skipped. |
| Skip-DF SNR (dB) | 40 | Above this SNR only the deep filter is skipped. |
| Silence expander depth (dB) | 20 | Extra attenuation in speech pauses. |
| Post filter beta | 0 | DeepFilterNet post-filter strength; 0 is off. |
| Startup mute (ms) | 1000 | Output muted while the model state settles. Set 0 for file processing. |
| Silence floor 60 Hz … 16 kHz (dB) | -200 | Optional per-band floor the output must exceed to leave silence; -200 disables it. |
| Voice gate depth (dB) | 40 | Attenuation while the detector hears no speech ([voice gate](#voice-gate)); 0 is off, 60 or more mutes. On, it raises the latency of either engine to 66 ms. Read when the plugin starts running; switching it on or off takes effect at the next activation. |
| latency (output) | | The current input-to-output delay in samples: 1919 or 959 without the voice gate, 3168 with it. |
| Voice detector (0 Silero, 1 FireRedVAD) | 0 | The voice gate's detector. Read when the plugin starts running, like the gate. |
| Voice sensitivity (1 sensitive, 3 normal, 5 strict) | 3 | The speech probability that opens the voice gate ([voice gate](#voice-gate)): 1 opens on the faintest voice and lets more noise through, 5 only on a clear voice. Follows live changes. |

The two skip thresholds default to 40 dB, the top of their range, so no stage
is skipped: at 30 and 20 dB the SI-SDR improvement on noisy speech fell from 11
to 1 dB and no CPU was saved.

An unconnected control runs at the default above, the same value a host that
applies the port hints starts from.

## Voice gate

Both plugins run a voice detector on the raw input and attenuate the denoised
output while it hears no speech. It removes the noise bursts the denoiser lets
through when nobody talks. The detector is Silero VAD (`silero-vad/`, a native
f32 port of the 16 kHz model) unless the Voice detector control chooses
FireRedVAD's streaming model ([FireRedVAD](firered-vad.md)); the rest of the gate
is the same for both, and so is the latency.

- Opens at the speech probability the Voice sensitivity control sets and
  closes after 320 ms in a row below half of it; attack 10 ms, release 150 ms.
  FireRedVAD is surer of itself: at 0.9 it keeps steady room noise out, which
  lower levels let through.

  | Sensitivity | 1 | 2 | 3 | 4 | 5 |
  |---|---:|---:|---:|---:|---:|
  | Silero opens at | 0.03 | 0.05 | 0.10 | 0.20 | 0.40 |
  | FireRedVAD opens at | 0.50 | 0.65 | 0.80 | 0.90 | 0.95 |

- A FireRedVAD chunk takes its most confident 10 ms frame. A decision covers
  32 ms of input, and each sample is gated by the decision about its own chunk
  and the chunks up to 32 ms after it, so word onsets are kept.
- The output is delayed until those decisions exist, so with the gate on both
  engines have the same latency:

  | Engine | Gate off | Gate on |
  |---|---:|---:|
  | DeepFilterNet3 | 1919 samples (40 ms) | 3168 samples (66 ms) |
  | DeepFilterNet3-LL | 959 samples (20 ms) | 3168 samples (66 ms) |

  The gate is on by default in both plugins. DeepFilterNet3-LL then has more
  latency than DeepFilterNet3 without the gate and loses the point of its low
  latency; set its voice gate depth to 0 to keep 20 ms.

- Silero runs as two models that restart in turn every 32.8 s, half a
  period apart; the one restarted longer ago answers. A single model run for
  minutes over music or sound effects drifts until it no longer hears speech,
  and the official ONNX model does the same. A model restarted only seconds
  ago takes steady room noise for speech: restarting every 4.1 s let 64 noise
  bursts through on the recording below, where 32.8 s lets the same 3 as a
  single model.
- Digital silence skips inference; after about 1 s of it the detector state
  is reset.
- Cost, measured as in [performance](performance.md) on one minute of noisy
  speech through FFmpeg, for either engine on the i5-13400: Silero +0.33 s of CPU
  per minute (2.5 G instructions), FireRedVAD +0.55 s (1.6 G instructions; it
  reads 2.2 MB of weights every 10 ms).

Measured on a 10-minute recording of room noise followed by speech, at the
app's settings (bursts above -50 dBFS in the noise part):

| Engine | Gate off | Gate on (40 dB) | Loudest burst, gate on |
|---|---:|---:|---:|
| DeepFilterNet3 | 52 | 3 | -40.8 dBFS |
| DeepFilterNet3-LL | 186 | 2 | -46.5 dBFS |

On speech mixed with noise, the gate leaves SI-SDR unchanged and attenuates
at most 2.6% of speech frames by more than 6 dB. Under steady pink noise it
still attenuates the first 50 ms of 16–20% of words, and of 7–15% of whispered
words; vacuum, keyboard and music under speech stay below 1.5%. Laughter,
singing and real distant speech have not been measured.

On film audio, three 10-minute excerpts of a 5.1 mix with the dialogue taken
from the center channel within the subtitle times, DeepFilterNet3 at full
strength keeps 95% of the dialogue and lets 375 s of the 30 minutes of score
and effects alone through. With the gate at 80 dB it keeps 83% of the dialogue
and lets 3 s through (DeepFilterNet3-LL: 84%, 7 s); a single model that never
restarts keeps 77%. The dialogue it loses is shouting over engines and speech
under loud music, which Silero does not hear in the raw input; listening
to the denoised output as well would need about 42 ms more latency and let
dozens of room-noise bursts through on the recording above.

With FireRedVAD as the detector, measured at sensitivity 4 (0.9), the gate
trades the other way. On the film it keeps 94% of the dialogue (DeepFilterNet3-LL: 95%) and lets 18 s through (23 s).
On the room-noise recording above it lets 6 bursts through, the loudest at
-21.7 dBFS (5, at -19.6 dBFS), and on sixteen microphone recordings of speech it
attenuates 0.66% of speech frames by more than 6 dB, against 0.36% with Silero
(0.80% against 0.46%).

| DeepFilterNet3, detector | Room-noise bursts | Speech frames cut | Film dialogue kept |
|---|---:|---:|---:|
| Silero | 3 | 0.36% | 83% |
| FireRedVAD, sensitivity 4 | 6 | 0.66% | 94% |

DPDFNet's voice gate is off by default ([DPDFNet](dpdfnet.md#voice-gate)): on
the same recording, without the gate, its residual is a constant floor near
-70 dBFS with rare bursts (loudest -16.5 dBFS), which listens as clean.

## PipeWire example

A virtual microphone with DeepFilterNet3, in
`~/.config/pipewire/pipewire.conf.d/99-deepfilternet.conf`:

```ini
context.modules = [
    { name = libpipewire-module-filter-chain
      args = {
        node.description = "Noise-suppressed microphone"
        filter.graph = {
            nodes = [
                { type = ladspa name = dfn plugin = "libdfn3_ladspa"
                  label = "deep_filter_net3_rs_mono"
                  control = { "Attenuation Limit (dB)" = 100.0 } }
            ]
        }
        capture.props = {
            node.name = "capture.dfn3" node.passive = true
            audio.rate = 48000 audio.channels = 1 audio.position = [ MONO ]
        }
        playback.props = {
            node.name = "dfn3_source" media.class = Audio/Source
            audio.rate = 48000 audio.channels = 1 audio.position = [ MONO ]
        }
      }
    }
]
```

For DeepFilterNet3-LL use `libdfn3ll_ladspa` / `deep_filter_net3_ll_rs_mono`;
for DPDFNet, `libdpdfnet_native` / `dpdfnet_native_48hr`.

## Tools

```sh
cargo run --release -p dfn3-tools --bin dfn3_cli -- denoise in.wav out.wav
cargo run --release -p dfn3-tools --bin dfn3_cli -- bench
```

`dfn3ll_cli` does the same for DeepFilterNet3-LL. Both use the embedded
weights; `DFN3_WEIGHTS` or `DFN3LL_WEIGHTS` at build time embeds another blob.
`denoise` takes optional post-filter beta, attenuation limit, Min SNR gate,
Skip-all and Skip-DF values after the file names. The input must be 48 kHz mono,
16-bit PCM or 32-bit float.

`deepfilternet3/*/fuzz` (nightly toolchain, outside the workspace) fuzzes each
plugin's `run()` as a host drives it, with any block size, in-place buffers,
reactivation, hostile control values and NaN/Inf audio. Both crates build
`dfn3-plugin/fuzz/run.rs`:

```sh
cd deepfilternet3/dfn3-ladspa/fuzz && cargo +nightly fuzz run -O run
```

## Validation

`cargo test` covers finite output, stationary-noise suppression, the silence
skip, a golden output per engine, the LADSPA ABI of both plugins
(`dfn3-plugin/tests/conformance`: block-size independence with the voice gate
on and off, in-place processing, rate gate, NaN/Inf and out-of-range input,
non-finite controls, reactivation, the measured delay against the `latency`
port, port defaults against engine defaults, attenuation semantics, the voice
gate's effect, no allocation in `run()`) and bit identity of each SIMD kernel
against the scalar loop. The
quantized path was compared stage by stage with the upstream ONNX model
(encoder and mask at 100–140 dB SNR); that comparison needs ONNX Runtime and is
not part of the test suite.
