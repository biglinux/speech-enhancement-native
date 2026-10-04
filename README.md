# speech-enhancement-native

Native Rust inference engines for published speech-enhancement models, built as
LADSPA and PipeWire plugins for BigLinux.

The models come from their authors: DeepFilterNet3 (Schröter et al.), DPDFNet
(Ceva) and GTCRN-AEC (LocalVQE). This repository reimplements their inference
path without ONNX Runtime, OpenVINO or PyTorch. Recurrent weights are int8 with
int16 activations (W8A16), and the kernels are chosen at runtime for the CPU:
scalar, SSE4.1, AVX or AVX2+FMA. The int8 matrix products accumulate exactly in
integers, so their result does not depend on the tier.

## Packages

| Package | Plugin | File | Label / name |
|---|---|---|---|
| `deepfilternet3-native` | DeepFilterNet3 noise suppression (default) | `/usr/lib/ladspa/libdfn3_ladspa.so` | `deep_filter_net3_rs_mono` |
| | DeepFilterNet3-LL, larger model, no lookahead | `/usr/lib/ladspa/libdfn3ll_ladspa.so` | `deep_filter_net3_ll_rs_mono` |
| | Lookahead peak limiter | `/usr/lib/ladspa/liblimiter_ladspa.so` | `biglinux_lookahead_limiter_mono` |
| `dpdfnet-native` | DPDFNet-2 48 kHz HR noise suppression | `/usr/lib/ladspa/libdpdfnet_native.so` | `dpdfnet_native_48hr` |
| | The same model over whole files, on several threads ([how](docs/dpdfnet.md#whole-recordings)) | `/usr/bin/dpdfnet-enhance` | |
| `gtcrn-aec-native` | GTCRN-AEC echo cancellation | `/usr/lib/spa-0.2/aec/libspa-aec-gtcrn.so` | `library.name = aec/libspa-aec-gtcrn` |

All plugins take 48 kHz mono and accept any host block size.

## CPU cost

CPU seconds per minute of audio, one core, measured with `perf stat`
([method](docs/performance.md)):

| Engine | i5-13400 (AVX2) | i3-2375M (AVX, 2011) |
|---|---:|---:|
| DeepFilterNet3 | 0.8 | 5.2 |
| DeepFilterNet3-LL | 2.1 | 15.2 |
| DPDFNet-2 48 kHz HR | 3.9 | 27.3 |
| GTCRN-AEC (16 kHz core) | 1.2 | 7.2 |

## Build

Rust 1.98.1 or newer (`rust-toolchain.toml`).

```sh
cargo build --release --locked --workspace --exclude dpdfnet-native
cargo build --locked --profile release-unwind -p dpdfnet-native

export AEC_GTCRN_GGUF=$PWD/gtcrn-aec/model/localvqe-pi-aec-v1-49k-f32.gguf
export DPDFNET_TEST_MODEL=$PWD/dpdfnet/model/dpdfnet2_48khz_hr-w8a16
cargo test --release --locked --workspace --exclude dpdfnet-native
cargo test --locked --profile release-unwind -p dpdfnet-native
cargo test --locked --profile release-unwind -p dpdfnet-native --test runtime --test offline -- --ignored
```

`pkgbuild/PKGBUILD` builds the three packages the same way.

## Layout

```text
ops/              int8/SIMD kernels shared by DeepFilterNet3 and GTCRN-AEC
deepfilternet3/   DeepFilterNet3 and -LL engines, CLIs, benchmarks, weight quantizer
limiter-ladspa/   peak limiter
dpdfnet/          DPDFNet engine, its own kernel crate, model bundle, export tools
gtcrn-aec/        GTCRN-AEC engine, PipeWire SPA plugin, model, evaluation harness
pkgbuild/         Arch/BigLinux split package
docs/             design notes per engine and performance method
```

`dpdfnet/ops` started as a copy of `ops` and diverged. Merging them needs the
bit-identity tests of both engines to pass on every SIMD tier.

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
