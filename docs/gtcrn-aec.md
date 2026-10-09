# GTCRN-AEC

A Rust port of the LocalVQE GTCRN-AEC echo canceller (49 K parameters, 2.3 MB
f32 GGUF), exposed to PipeWire as an SPA AEC plugin. The source of the port is
LocalVQE `ggml/gtcrn.cpp` (network) and `ggml/daf_frontend.cpp` (delay-and-filter
front end).

## Why this model

`gtcrn-aec/eval` builds a synthetic test set (speech, a synthetic room response,
bulk delay, loudspeaker non-linearity, room noise; four 10 s scenarios) and
scores every candidate with the same aligned PESQ/STOI/ERLE pipeline. WebRTC runs
through the SPA interface PipeWire ships. The GTCRN-AEC row is the LocalVQE CLI
at 16 kHz, not this port or the 48 kHz plugin. Results:

| Candidate | Echo removed, dB | Near-end PESQ | Double-talk STOI | RTF | Frame |
|---|---:|---:|---:|---:|---:|
| none | 0.0 | 4.21 | 0.38 | – | – |
| WebRTC (PipeWire) | 18.7 | 4.02 | 0.30 | 0.008 | 10 ms |
| SpeexDSP | 3.9 | 4.09 | 0.40 | 0.008 | 16 ms |
| DTLN-aec 128 | 30.4 | 2.35 | 0.47 | 0.034 | 32 ms |
| GTCRN-AEC 49 K | 11.5 | 4.25 | 0.44 | 0.080 | 16 ms |

Method, commands and the full table: `gtcrn-aec/eval/RESULTS.md`. "Frame" is
the frame or hop length, not the delay through the plugin (see Latency). DTLN
removes the most echo but damages the speaker's own voice. GTCRN-AEC leaves the
near-end voice as intact as no processing at all and removes less echo than
WebRTC. For a microphone that is the right trade-off: the BigLinux microphone
app enables the AEC only while something is playing, so most of the time only
near-end quality matters.

## Network

- STFT and iSTFT, n_fft 512, hop 256, 257 bins at 16 kHz. The model stores them
  as DFT matrices with the window baked in; at load they are checked to be a
  sine-windowed real DFT, which the streaming path runs with `realfft`. A model
  whose matrices are anything else is rejected.
- ERB sub-bands: the upper 192 bins are merged into 64 bands and split back.
- Encoder of five convolution blocks; blocks 2–4 are GTConv (pointwise, PReLU,
  depthwise 3×3, pointwise, temporal GRU attention), dilations 1, 2, 5.
- Two dual-path grouped RNN blocks (intra bidirectional GRU, inter GRU, linear,
  LayerNorm).
- Mirrored decoder with skip additions and a complex (re/im) mask.
- DAF front end: a GCC-PHAT estimate of the bulk delay (up to 1 s, refined
  every 0.5 s and locked once confident, after at least 3.5 s of audio), then a
  partitioned frequency-domain Kalman filter steered by small GRUs, which
  removes the linear echo before the network. Once locked, the delay is not
  estimated again, as in LocalVQE: if the echo path's bulk delay later changes
  (another output device, for instance), the Kalman filter's 1 s span absorbs
  what it can until a numeric fault makes the DAF re-acquire the delay.

One GCC-PHAT observation costs three 32768-point FFTs, about 1 ms. In the
streaming path it runs as 73 steps of similar size, four per 128-sample DAF
block, so no audio callback carries the whole cost; each observation window
closes 18 blocks (144 ms) before its result is applied, which keeps the
estimates and the lock on the same samples as a synchronous estimator. The
whole-file path (`run_aec`) runs each observation at once.

The GRUs stay f32. Together they are about 5% of the time; int8 would save about
2% and break the reference comparison.

## Plugin

`module-echo-cancel` loads it with `library.name = aec/libspa-aec-gtcrn`.

- Streams: the capture, playback and output streams must all be 48 kHz planar
  f32 (`F32P`); `init`/`init2` return `-EINVAL` otherwise. `init2` sets each
  stream to mono, and `module-echo-cancel` adopts that, so its stereo default
  needs no configuration; `init`, which older hosts call, requires mono. The plugin
  asks for blocks of 768 samples (`768/48000`) but accepts any block size,
  and the output is the same sample for sample whatever the block size:
  every decision happens on the pipeline's own hop and block boundaries.
- Properties: `gtcrn.model` sets the model path, default
  `/usr/share/gtcrn-aec-native/localvqe-pi-aec-v1-49k-f32.gguf`. A model that
  cannot be read or does not match the shipped model's tensors fails `init`
  with a message on stderr (the PipeWire log) and `-ENOENT`, `-EACCES` or
  `-EINVAL`.
- Bands: up to 8 kHz the signal is resampled to 16 kHz and goes through the
  network in 256-sample hops, buffered through FIFOs. Above 8 kHz, which the
  network never sees, a linear partitioned frequency-domain adaptive filter
  (`gtcrn-aec/src/hbaec.rs`) cancels echo from the loopback reference, and the
  result is added back. Its overlap-save constraint visits one of the 64
  partitions per block, which keeps the adapting filter at about 2% of an
  i5-13400 core.
- High-band duck: each hop's output-to-input amplitude ratio of the low band
  drives the high band's gain, fully off at 0.15 or below and fully on at 0.45
  or above, smoothed over about 10 ms. Strong suppression means far-end echo
  dominates, so the high band is muted; near-end speech and double talk keep
  the low band and therefore the high band. The high-band filter adapts only
  while that ratio is below 0.5; each 256-sample block reads it from the
  newest hop whose input the block has entirely seen.
- Pause: `activate` and `deactivate` do nothing. `module-echo-cancel` calls
  them from the main thread while the data thread may be running, so they
  cannot reset the engine; after a pause it continues from its buffered and
  adapted state, including the locked delay.
- Faults: input samples that are not finite become silence. A non-finite value
  inside the network path or the high-band filter resets that part and
  silences its output for one hop or block; the DAF then re-acquires the delay
  as at start-up.

`Engine::run`, `Streamer::process_hop` and the DAF allocate nothing after
warm-up, also on the first call from a thread other than the one that built
them, as on the PipeWire data thread (`gtcrn-aec/tests/rt_alloc.rs` and the
`rt_alloc_tests` in `spa-aec-gtcrn`).

## Latency

Both bands leave the plugin 1728 samples (36 ms) after they enter, whatever
the host block size:

- 768 samples (16 ms): one network hop, buffered before the first output.
- 768 samples (16 ms): the STFT overlap-add, which completes a sample one hop
  after it enters the 512-sample window.
- 2 × 96 samples (4 ms): the down- and up-sampling filters.

The network alone, at 16 kHz, accounts for the first two (32 ms); its 16 ms hop
is the "Frame" in the table above. `both_bands_have_the_documented_latency`
checks the total.

## DAF numerical note

The DAF trajectory forks on a one-ulp edge in the block Kalman clamp:

```rust
if s > 2.0 { m_ *= 2.0 / s; }                    // m_·x2 == 2.0 in exact arithmetic
let fac = (1.0 - 0.5 * m_ * sc.x2[k]).max(0.0);  // 0 ± 1 ulp: rounding picks 0 or ~6e-8
```

At onset this moves the Kalman prior by seven decades on every bin, so any change
in rounding (SIMD width, FMA, batching) produces a different sample stream. The
LocalVQE C++ reference has the same clamp. Both branches scored the same on the
evaluation set (within 0.1 dB) when this was measured, so the DAF is gated by
echo reduction (`run_aec_stream_cancels_echo` asserts more than 25 dB; it
measures 28.3 dB, a broken filter stays under 10) and by `eval/`, not by sample
equality. Making the clamp exact would change the algorithm and needs its own
evaluation run.

## Verification

- Network stages against a dump of the reference run on the same GGUF
  (`eval/dump_gtcrn.cpp`): every stage within 1e-4 absolute
  (`core_stages_match_reference_dump`, ignored unless `AEC_GTCRN_FIXTURES`
  points at the dump).
- Streaming network against the batch one: within 1e-5 of the output peak,
  measured 8e-7 (`streaming_core_matches_batch`).
- Measured during the port and not covered by a test: the offline path matched
  the LocalVQE ggml CLI within 8.5e-5 relative, the streaming path matched
  `localvqe --stream` within 7e-5, and the `realfft` STFT was 131 dB SNR from
  the matrix path.

Tests load the model from `gtcrn-aec/model/`; `AEC_GTCRN_GGUF` overrides it.
`src/gtcrn_aec_schema.txt`, the tensor list `Model::load` checks, is generated
from the model: `schema_lists_the_shipped_model` fails and prints the new text
when it is out of date. Offline run:

```sh
cargo run --release -p gtcrn-aec --example aec_run -- \
    gtcrn-aec/model/localvqe-pi-aec-v1-49k-f32.gguf mic.f32 ref.f32 out.f32
```

Inputs are raw 16 kHz mono f32; `AEC_STREAM=1` selects the streaming path the
plugin uses.
