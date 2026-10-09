# DPDFNet-2 48 kHz HR

A dedicated executor for the DPDFNet 48 kHz HR profile at upstream commit
`9bd9844a`. Depth 2 is shipped; depth 8 loads with the same code at a higher
cost. The 8 and 16 kHz models have different features and topology
and are not supported.

The rules below reproduce the upstream model. Do not simplify any of them
without a differential comparison (`dpdfnet/tools/compare_reference.py`).

## Conventions

FFT 960, hop 480, 481 bins, 96 deep-filter bins, 64 channels, deep-filter order
5. Internal layout is `[time, frequency, channel]`; a complex spectrum is
`[frequency, re/im]`. The FFT is unnormalized: the spectrum is multiplied by
`wnorm` before the features and divided by it on the way out; the inverse FFT is
divided by 960. The window is exported from the upstream object, and the loader
checks that it is power-complementary at a 480-sample shift.

## Features

```text
mag       = sqrt(re² + im²)
db        = 10·log10(mag + 1e-10)            10, not 20: it is what the model was trained on
mu_next   = alpha·mu + beta·db
feat_mag  = (db − mu_next) / (40 + eps)
s_next    = alpha·s + beta·mag
feat_cplx = spectrum / sqrt(s_next + eps)    first 96 bins
```

`beta` is `1 − alpha` computed in Python and exported separately; recomputing
it in f32 gives a different constant. The initial states `mu0` and `s0` come from
`initial_state()`, not zeros. There is no adaptive variance normalization.

## Encoder and mask decoder

| Point | Shape |
|---|---|
| normalized magnitude | 481 |
| magnitude conv input | 3×480×1 (Nyquist bin dropped) |
| e0, e1, e2, e3 | 480×64, 160×64, 80×64, 40×64 |
| complex input | 3×96×2 |
| c0, c1 | 96×64, 48×64 |
| DPRNN inputs | 40×64 and 48×64 |
| branch projections | 512 each, concatenated magnitude first: 1024 |
| encoder GRU | grouped linear → 256, 1 GRU, grouped linear → 512 |
| mask decoder GRU | grouped linear → 256, 2 GRUs, grouped linear → 512 |
| upsampling | factors 2, 2, 3 with skips and convolutions |
| mask | 480 sigmoid values plus reflect padding |

The skips take `e0..e3` before the DPRNNs. In the pinned `Conv2dNormAct`, a 1×1
kernel disables the extra pointwise step but keeps the computed grouping, so the
skips have no dense convolution. `mask[480] = mask[478]` (reflect, not
replicate). Subpixel phases are interleaved by frequency.

## DPRNN

Each branch has 2 blocks (8 for depth 8). Per block:

1. The spectral GRU input matrices are applied to every frequency. The two
   recurrences run over frequency forward and backward with a zero state on
   every frame; outputs are `[forward, backward]` per frequency.
2. Linear 128→64, LayerNorm over 64 channels, residual add.
3. Temporal GRU 64→64 with one persistent state per frequency, computed four
   frequencies at a time.
4. Linear 64→64, LayerNorm, residual add.

LayerNorm uses the population variance (divide by 64) and cannot be folded into
the weights.

## GRUs and quantization

PyTorch orders gates `r, z, n`; the kernels use `z, r, n`. The exporter reorders
W, R and both bias sets explicitly:

```text
z  = σ(Wz·x + Rz·h + bWz + bRz)
r  = σ(Wr·x + Rr·h + bWr + bRr)
n  = tanh(Wn·x + bWn + r·(Rn·h + bRn))      the recurrent bias stays inside r·(…)
h' = (1 − z)·n + z·h
```

Weights are symmetric int8 in [-127, 127] with one scale per row; the loader
rejects -128. Activations are int16 in [-16383, 16383] with one scale per
vector, rounded half away from zero like `f32::round`. `127·16383·1024 <
INT32_MAX`, so no accumulator can overflow. Convolutions, other projections,
states, normalizations and gates stay in f32. When the reciprocal of a tiny
activation scale overflows f32, the quantizer divides instead, or writes zeros
if the scale itself rounds to zero. f32 PyTorch and the SIMD kernels are not
bit-identical; the integer matrix products are, on every SIMD tier.

## Deep-filter decoder and delays

The DF GRU input is grouped by 8, the skip and output by 16; there are two GRUs
of 256. The last projection gives 960 values, applies `tanh`, and only then adds
the convolutional path over five frames of `c0`. Coefficients are
`[96 frequencies, 5 taps, re/im]`. For frame `t` in the `before_df` profile:

```text
masked(t)  = mask(t) · raw(t−2)
coef       = coef(t−2)
DF_low     = Σtap coef[tap] · masked(t−4+tap)
DF_high    = masked(t−2)[96:]
dry        = raw(t−4)
```

The attenuation mix uses that aligned `dry`, never the current input; mixing
different instants cancels parts of the voice. Histories are ring buffers whose
logical index 0 is the oldest frame after the push.

## PCM, latency and faults

Each inference consumes 480 new samples. Analysis, overlap-add and emission of
the previous block add two hops to the network's four: **2880 samples, 60 ms**,
reported on the `latency` output port. Host quantum, resamplers and other
filters come on top.

Splitting the same input into blocks of 1, 7, 64, 128, 256, 480, 512, 960, 1024
or 8192 samples gives identical output. Attenuation 0 dB is the aligned dry
signal, 100 dB full enhancement; changes are smoothed over up to five hops and
every frame is still inferred. NaN/Inf input is replaced by zero. An FFT error,
a non-finite result or a panic sets the `Processing fault` port and outputs
silence until reset; the plugin never falls back to raw audio, and no panic
unwinds into the host.

## Memory and threads

One aligned allocation holds the weights; tensors are `Arc` views into it. The
file is read, not mapped. int8 matrices are never expanded to f32. Instances in
one process share the weights through a weak cache whose mutex is only taken in
instantiate and cleanup. State and scratch buffers belong to each instance. `run`
enables flush-to-zero and denormals-are-zero, so silence decaying through the
recurrent state stays fast, and restores the host's MXCSR before returning.
Warm-up touches the buffers and resolves the SIMD dispatch before the first
`run`.

## Whole recordings

`dpdfnet-enhance` (`src/offline.rs`) enhances a file on several threads for the
converters; live audio keeps the LADSPA plugin. It reads interleaved f32 48 kHz
frames on stdin and writes the same number of frames on stdout, already aligned:
the 2880-sample latency is removed.

```sh
ffmpeg -i in.m4a -ar 48000 -f f32le - |
    dpdfnet-enhance --channels 2 --attenuation 48 |
    ffmpeg -f f32le -ar 48000 -ac 2 -i - out.flac
```

`--channels` is 1 to 64 (default 1), `--attenuation` 0 to 100 dB (default
100), `--threads` 1 to 256 stage threads (default below); `--help` lists them.
The model comes from `DPDFNET_NATIVE_MODEL` or the installed bundle. Usage
errors exit with 2, processing errors with 1, and raw samples are never written
to a terminal.

Each channel runs through the network as a pipeline. `Model::into_stages`
splits it into stages that keep only their own state: the features and input
convolutions, each DPRNN block, the encoder GRU with the deep-filter decoder,
and the two halves of the mask decoder with the final filter. While a later
stage works on hop t an earlier one already works on hop t+1, and every stage
does the same arithmetic in the same order as `process_spectrum`. So the output
is bit-identical to the plugin, preceded by its opening half second played
backwards as in the converters, for any length and any thread count.

Stage threads are pinned, one per physical core, fastest cores first. Left to
the scheduler, stages that sleep between hops often share the two hyperthreads
of one core and run at about half speed: 2.5 s instead of 1.2 s for a minute
of mono on the i5-13400. The default thread count is the number of cores of
the fastest kind; more threads land on E-cores or hyperthreads and are slower.

| | LADSPA | `dpdfnet-enhance` |
|---|---:|---:|
| i5-13400, 1 min stereo | 7.8 s | 1.6 s (6 threads) |
| i5-13400, 5 min stereo | 37.6 s | 7.9 s |
| i5-13400, 5 min mono | 19.7 s | 5.4 s |
| i3-2375M, 1 min stereo | 53.3 s | 28.0 s (2 threads) |
| i3-2375M, 1 min mono | 26.6 s | 15.4 s |

With 6 stage threads the slowest group is the front with the first ERB block,
159 µs a hop against 106 µs for a perfect split; finer stages would need the
DPRNN blocks cut into their spectral and temporal halves. Memory stays near
13 MB mono and 18 MB stereo whatever the length. A processing fault is an
error, never silence.

## Model bundle

`dpdfnet/model/dpdfnet2_48khz_hr-w8a16/` holds `manifest.json` (shapes,
offsets, constants, SHA-256 of `weights.bin`), `packing.json` and `weights.bin`
(3.6 MB). The plugin loads `/usr/share/dpdfnet-native/dpdfnet2_48khz_hr-w8a16`
or the directory in `DPDFNET_NATIVE_MODEL`, and checks shapes, bounds,
finiteness, alignment and the hash before running.

Reproduce it (Python with PyTorch, `dpdfnet/tools/requirements-validation.txt`;
none of this is needed at runtime):

```sh
python dpdfnet/tools/fetch_inputs.py --directory inputs \
    --hf-revision dd6818d00f50c836fed43a6243ebe49116de5964
python dpdfnet/tools/export_model.py --upstream inputs/upstream \
    --checkpoint inputs/checkpoints/dpdfnet2_48khz_hr.pth \
    --mode w8a16 --output row-major-w8a16
python dpdfnet/tools/pack_matrices.py --source row-major-w8a16 \
    --output dpdfnet2_48khz_hr-w8a16
```

`packing.json` records the SHA-256 of the row-major source and of the packed
result; packing only moves bytes. The loader accepts only packed (schema 2)
bundles. `export_model.py --mode f32` gives the unquantized reference used by
`compare_reference.py`, packed the same way; it checks the 14 intermediate
points that `dpdfnet_native_trace` exposes (listed in
`dpdfnet/include/dpdfnet_native.h`) and reports the first divergence.

## Diagnostics

```sh
cargo run --profile release-unwind -p dpdfnet-native --example inspect -- BUNDLE
cargo run --profile release-unwind -p dpdfnet-native --example benchmark -- BUNDLE [SECONDS]
python dpdfnet/tools/check_abi.py --bundle BUNDLE
```

`inspect` prints why a bundle does not load. `benchmark` times every hop of the
streaming path and fails on any heap allocation while processing.
`check_abi.py` drives the C API and the LADSPA descriptor of the built library,
in place and out of place.

## Why int8 and not int4

Measured against the f32 model on noisy speech, the shipped int8 bundle loses
0.00 dB SI-SDR. Round-to-nearest int4 loses 0.4–0.7 dB; GPTQ int4 with groups of
32 loses 0.05–0.13 dB. The possible gain is small: on the i3, cycles stalled on
memory beyond L2 are 6% of the total, and unpacking 4-bit values costs extra
instructions on a CPU with 128-bit integer SIMD. int8 stays.
