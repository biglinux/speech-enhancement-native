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
| Voice gate depth (dB) | 40 | Attenuation while the Silero detector hears no speech ([voice gate](#voice-gate)); 0 is off, 60 or more mutes. On, it raises the latency of either engine to 66 ms. Read when the plugin starts running; switching it on or off takes effect at the next activation. |
| latency (output) | | The current input-to-output delay in samples: 1919 or 959 without the voice gate, 3168 with it. |

The two skip thresholds default to 40 dB, the top of their range, so no stage
is skipped: at 30 and 20 dB the SI-SDR improvement on noisy speech fell from 11
to 1 dB and no CPU was saved.

An unconnected control runs at the default above, the same value a host that
applies the port hints starts from.

## Voice gate

Both plugins run Silero VAD (`silero-vad/`, a native f32 port of the 16 kHz
model) on the raw input and attenuate the denoised output while it hears no
speech. It removes the noise bursts the denoiser lets through when nobody talks.

- Opens at a speech probability of 0.2 and closes after 320 ms in a row below
  0.1; attack 10 ms, release 150 ms. A decision covers 32 ms of input, and each sample is
  gated by the decision about its own chunk and the chunks up to 32 ms after
  it, so word onsets are kept.
- The output is delayed until those decisions exist, so with the gate on both
  engines have the same latency:

  | Engine | Gate off | Gate on |
  |---|---:|---:|
  | DeepFilterNet3 | 1919 samples (40 ms) | 3168 samples (66 ms) |
  | DeepFilterNet3-LL | 959 samples (20 ms) | 3168 samples (66 ms) |

  The gate is on by default in both plugins. DeepFilterNet3-LL then has more
  latency than DeepFilterNet3 without the gate and loses the point of its low
  latency; set its voice gate depth to 0 to keep 20 ms.

- Digital silence skips inference; after about 1 s of it the detector state
  is reset.
- Cost, measured as in [performance](performance.md) on one minute of noisy
  speech: +0.07 s of CPU per minute on the i5-13400 and +0.44 s on the
  i3-2375M, for either engine.

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

DPDFNet has no voice gate: on the same recording its residual is a constant
floor near -70 dBFS with rare bursts (loudest -16.5 dBFS), which listens as
clean.

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
