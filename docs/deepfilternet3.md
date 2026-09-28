# DeepFilterNet3

Two engines with the same structure: DeepFilterNet3 (GRU hidden size 256,
weights 2.7 MB, two frames of lookahead) and DeepFilterNet3-LL (hidden size 512,
weights 10.7 MB, no lookahead). DeepFilterNet3 is the default; its weights fit
the 3 MB L3 of the oldest supported CPUs.

## Signal path

48 kHz mono → STFT (FFT 960, hop 480) → ERB and complex features → encoder,
GRUs → ERB mask and deep filter on the lowest bins → iSTFT.

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
blobs are produced by `deepfilternet3/scripts/quantize_int8.py`
([details](../deepfilternet3/scripts/README.md)).

int8 activations (W8A8) were measured and rejected. On the i3 (AVX, no AVX2,
FMA or VNNI) the only int8 × int8 instruction, `pmaddubsw`, saturates at
2·255·127, so an exact A8 product would need 7-bit weights or a widening back to
int16, which is the current kernel. The i5 could run A8 exactly with AVX-VNNI,
but it already spends under 2% of a core on DeepFilterNet3.

## Plugin behaviour

- Rejects sample rates other than 48 kHz instead of processing them wrongly.
- Buffers any host block size into 480-sample frames; the output does not
  depend on the block size (tested from 1 sample to blocks larger than a frame).
- Replaces NaN/Inf input and control values.
- Shares the decoded weights between instances in one process.
- Does not set `LADSPA_PROPERTY_HARD_RT_CAPABLE`: stage skipping makes the cost
  of a frame depend on the signal.

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
cargo run --release -p dfn-tools --bin dfn3_cli -- \
    deepfilternet3/dfn3-ladspa/dfn3_weights.bin in.wav out.wav
cargo run --release -p dfn-tools --bin dfn3_bench -- \
    deepfilternet3/dfn3-ladspa/dfn3_weights.bin
```

`ll_cli` and `ll_bench` do the same for DeepFilterNet3-LL. The WAV files must
be 48 kHz mono. `deepfilternet3/*/fuzz` holds libFuzzer targets for the
processing path (nightly toolchain, outside the workspace).

## Validation

`cargo test` covers finite and deterministic output, stationary-noise
suppression, a golden output per engine, the LADSPA ABI (block-size
independence, in-place processing, rate gate, NaN/Inf handling, attenuation
semantics) and bit identity of each SIMD kernel against the scalar loop. The
quantized path was compared stage by stage with the upstream ONNX model
(encoder and mask at 100–140 dB SNR); that comparison needs ONNX Runtime and is
not part of the test suite.
