# GTCRN-AEC

A Rust port of the LocalVQE GTCRN-AEC echo canceller (49 K parameters, 2.3 MB
f32 GGUF), exposed to PipeWire as an SPA AEC plugin. The source of the port is
LocalVQE `ggml/gtcrn.cpp` (network) and `ggml/daf_frontend.cpp` (delay-and-filter
front end).

## Why this model

`gtcrn-aec/eval` builds a test set (real speech, a synthetic room response, bulk
delay, loudspeaker non-linearity, recorded room noise; four 10 s scenarios) and
scores every candidate with the same aligned PESQ/STOI/ERLE pipeline. WebRTC runs
through the SPA interface PipeWire ships. 16 kHz results:

| Candidate | Echo removed, dB | Near-end PESQ | Double-talk STOI | RTF | Latency |
|---|---:|---:|---:|---:|---:|
| none | 0.0 | 4.21 | 0.38 | – | – |
| WebRTC (PipeWire) | 18.7 | 4.02 | 0.30 | 0.008 | 10 ms |
| SpeexDSP | 3.9 | 4.09 | 0.40 | 0.008 | 16 ms |
| DTLN-aec 128 | 30.4 | 2.35 | 0.47 | 0.034 | 32 ms |
| GTCRN-AEC 49 K | 11.5 | 4.25 | 0.44 | 0.080 | 16 ms |

Full table: `gtcrn-aec/eval/RESULTS.md`. DTLN removes the most echo but damages
the speaker's own voice. GTCRN-AEC leaves the near-end voice intact and handles
double-talk better than WebRTC; it removes less echo. For a microphone that is
the right trade-off: the BigLinux microphone app enables the AEC only while
something is playing, so most of the time only near-end quality matters.

## Network

- STFT and iSTFT, n_fft 512, hop 256, 257 bins at 16 kHz. The model stores them
  as DFT matrices with the window baked in; at load they are checked to be a
  sine-windowed real DFT and replaced by `realfft` (the matrix path remains for
  any other model).
- ERB sub-bands: the upper 192 bins are merged into 64 bands and split back.
- Encoder of five convolution blocks; blocks 2–4 are GTConv (pointwise, PReLU,
  depthwise 3×3, pointwise, temporal GRU attention), dilations 1, 2, 5.
- Two dual-path grouped RNN blocks (intra bidirectional GRU, inter GRU, linear,
  LayerNorm).
- Mirrored decoder with skip additions and a complex (re/im) mask.
- DAF front end: aligns and filters the loopback reference against the microphone
  before the network. This is what makes it an echo canceller.

The GRUs stay f32. Together they are about 5% of the time; int8 would save about
2% and break the reference comparison.

## Plugin

`module-echo-cancel` loads it with `library.name = aec/libspa-aec-gtcrn`. The
optional `gtcrn.model` argument (or `AEC_GTCRN_MODEL`) overrides the model path,
default `/usr/share/gtcrn-aec-native/localvqe-pi-aec-v1-49k-f32.gguf`.

PipeWire runs at 48 kHz. The plugin resamples to 16 kHz, runs the network per
256-sample hop and resamples back, through FIFOs, so any host block size works;
it requests blocks of 768 samples at 48 kHz. The band above 8 kHz, which the
network never sees, goes through a linear partitioned frequency-domain adaptive
filter (`aec-gtcrn/src/hbaec.rs`) and is added back. `Streamer::process_hop` and the DAF allocate nothing after
warm-up (`aec-gtcrn/tests/rt_alloc.rs`), also on the first call from a fresh
thread (`tests/xthread.rs`), as the PipeWire RT thread does.

## DAF numerical note

The DAF trajectory forks on a one-ulp edge in the block Kalman clamp:

```rust
if s > 2.0 { m_ *= 2.0 / s; }                    // m_·x2 == 2.0 in exact arithmetic
let fac = (1.0 - 0.5 * m_ * sc.x2[k]).max(0.0);  // 0 ± 1 ulp: rounding picks 0 or ~6e-8
```

At onset this moves the Kalman prior by seven decades on every bin, so any change
in rounding (SIMD width, FMA, batching) produces a different sample stream. The
LocalVQE C++ reference has the same clamp. Both branches score the same on the
evaluation set (within 0.1 dB), so the DAF is gated by echo reduction
(`run_aec_stream_cancels_echo`, > 25 dB; a working filter reaches ~28, a broken
one stays under 10) and by `eval/`, not by sample equality. Making the clamp
exact would change the algorithm and needs its own evaluation run.

## Verification

- Every network stage is bit-exact against an oracle dumped from the reference
  on the same GGUF (`eval/dump_gtcrn.cpp`).
- End to end, the offline path matches the LocalVQE ggml CLI within 8.5e-5
  relative, and the streaming path matches `localvqe --stream` within 7e-5, with
  the same scores on the evaluation set.
- The `realfft` STFT is 131 dB SNR from the matrix path.

Tests that need the model read `AEC_GTCRN_GGUF`; the stage oracle also needs
`AEC_GTCRN_FIXTURES`. Offline run:

```sh
cargo run --release -p aec-gtcrn --example aec_run -- \
    gtcrn-aec/model/localvqe-pi-aec-v1-49k-f32.gguf mic.f32 ref.f32 out.f32
```

Inputs are raw 16 kHz mono f32; `AEC_STREAM=1` selects the streaming path the
plugin uses.
