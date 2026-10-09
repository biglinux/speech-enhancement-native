# speech-enhancement-native

Native Rust inference engines for published speech-enhancement models, built as
LADSPA and PipeWire plugins for BigLinux.

The models come from their authors: DeepFilterNet3 (Schröter et al.), DPDFNet
(Ceva), GTCRN-AEC (LocalVQE) and Silero VAD (Silero Team). This repository
reimplements their inference path without ONNX Runtime, OpenVINO or PyTorch.
Recurrent weights are int8 with int16 activations (W8A16), and the kernels are
chosen at runtime for the CPU: scalar, SSE4.1, AVX or AVX2+FMA. The int8 matrix
products accumulate exactly in integers, so their result does not depend on the
tier.

## Packages

| Package | Plugin | File | Label / name |
|---|---|---|---|
| `deepfilternet3-native` | DeepFilterNet3 noise suppression (default) | `/usr/lib/ladspa/libdfn3_ladspa.so` | `deep_filter_net3_rs_mono` |
| | DeepFilterNet3-LL, larger model, no lookahead | `/usr/lib/ladspa/libdfn3ll_ladspa.so` | `deep_filter_net3_ll_rs_mono` |
| | Lookahead peak limiter | `/usr/lib/ladspa/liblimiter_ladspa.so` | `biglinux_lookahead_limiter_mono` |
| `dpdfnet-native` | DPDFNet-2 48 kHz HR noise suppression | `/usr/lib/ladspa/libdpdfnet_native.so` | `dpdfnet_native_48hr` |
| | The same model over whole files, on several threads ([how](docs/dpdfnet.md#whole-recordings)) | `/usr/bin/dpdfnet-enhance` | |
| `gtcrn-aec-native` | GTCRN-AEC echo cancellation | `/usr/lib/spa-0.2/aec/libspa-aec-gtcrn.so` | `library.name = aec/libspa-aec-gtcrn` |

All plugins take 48 kHz mono and accept any host block size. Both DeepFilterNet3
plugins include a Silero voice gate that mutes the output while nobody speaks.
It is on by default and raises the latency of both engines to 66 ms, from 40 ms
for DeepFilterNet3 and 20 ms for DeepFilterNet3-LL ([details](docs/deepfilternet3.md#voice-gate)).
Each LADSPA plugin reports its current delay, in samples, on an output control
port named `latency`, which PipeWire's filter-chain adds to the node latency.

## CPU cost

CPU seconds per minute of audio, one core, measured with `perf stat`
([method](docs/performance.md)):

| Engine | i5-13400 (AVX2) | i3-2375M (AVX, 2011) |
|---|---:|---:|
| DeepFilterNet3 | 0.8 | 5.2 |
| DeepFilterNet3-LL | 2.1 | 15.2 |
| DPDFNet-2 48 kHz HR | 3.9 | 27.3 |
| GTCRN-AEC (16 kHz core) | 1.2 | 7.2 |
| Voice gate, added to either DeepFilterNet3 engine | 0.07 | 0.44 |

## Build

Rust 1.98.1 or newer (`rust-toolchain.toml`).

```sh
cargo build --release --locked --workspace --exclude dpdfnet-native
cargo build --locked --profile release-unwind -p dpdfnet-native

cargo test --release --locked --workspace --exclude dpdfnet-native
cargo test --locked --profile release-unwind -p dpdfnet-native
```

The tests use the models in the repository.

`pkgbuild/PKGBUILD` builds the three packages the same way.

## Layout

```text
ops/              int8/SIMD kernels and resampler shared by all engines
deepfilternet3/   DeepFilterNet3 and -LL engines, their shared LADSPA plugin crate,
                  CLIs and benchmark, weight quantizer
silero-vad/       Silero VAD and the voice gate of the DeepFilterNet3 plugins
limiter-ladspa/   peak limiter
dpdfnet/          DPDFNet engine, model bundle, export tools
gtcrn-aec/        GTCRN-AEC engine, PipeWire SPA plugin, model, evaluation harness
pkgbuild/         Arch/BigLinux split package
testdata/         synthetic speech and the allocation counter shared by the tests
docs/             design notes per engine and performance method
```

## Documentation

- [DeepFilterNet3](docs/deepfilternet3.md)
- [DPDFNet](docs/dpdfnet.md)
- [GTCRN-AEC](docs/gtcrn-aec.md)
- [Performance and numerical rules](docs/performance.md)

## License

Code: MIT OR Apache-2.0 ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)); the `gtcrn-aec` crates are Apache-2.0. Model
weights keep their upstream licenses. Sources, checksums and changes are listed
in [NOTICE](NOTICE).
