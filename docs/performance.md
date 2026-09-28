# Performance and numerical rules

## Reference machines

- Intel Core i3-2375M (Sandy Bridge, 2011): AVX without AVX2, FMA or VNNI,
  3 MB L3, 1.5 GHz. This is the machine a change has to help.
- Intel Core i5-13400: AVX2, FMA, AVX-VNNI, hybrid P/E cores. Measure on a P-core.

## Current cost

CPU seconds per minute of audio on one core:

| Engine | i5-13400 | i3-2375M |
|---|---:|---:|
| DeepFilterNet3 | 0.8 | 5.2 |
| DeepFilterNet3-LL | 2.1 | 15.2 |
| DPDFNet-2 48 kHz HR | 3.9 | 27.3 |
| GTCRN-AEC, 16 kHz streaming core | 1.2 | 7.2 |

Input: one minute of noisy speech at 48 kHz for the denoisers
(`dfn3_cli`, `ll_cli`, `oldcpu_probe render`) and two minutes of 16 kHz
microphone and loopback signals for the AEC (`aec_run` with `AEC_STREAM=1`).

## Measuring

```sh
taskset -c 4 perf stat -e task-clock,cycles:u,instructions:u -- <command>
```

- Pin the process to one core and compare user-space cycles, not wall time; the
  i5 is often busy with other work.
- Alternate the two builds (A B A B) and repeat until the spread is smaller than
  the effect.
- A change to the numerics must keep the output bit-identical (compare the md5
  of the raw f32 output) unless it states why not and passes the engine's quality
  gate.
- `perf record` + `perf annotate` locates the hot loop; check it for scalar
  `ss`/`sd` instructions, `xmm` where `ymm` is available, and bounds checks.
- Sampling needs `kernel.perf_event_paranoid` ≤ 1 for user space.

## Build rules

- **No `target-cpu`.** The packages run on any x86-64. Wider SIMD is reached only
  through `#[target_feature]` functions chosen at runtime by `simd_tier()`
  (0 scalar, 1 SSE4.1, 2 AVX, 3 AVX2+FMA), which is resolved once and cached.
- **f32 reductions do not autovectorize** without fast-math, so every dot
  product that matters has an explicit kernel.
- **Bit identity across tiers:** exact kernels use separate multiply and add
  (no FMA) and the scalar loop's summation order. The int8 × int16 products
  accumulate exactly in `i32` and have no such restriction.
- **`panic = "abort"`** for the release profile. With unwinding the GTCRN-AEC
  streaming path is 37% slower on the i3 (RTF 0.166 against 0.121; DeepFilterNet3
  and DPDFNet do not change). DPDFNet is built with `--profile release-unwind`
  because it catches initialization panics at the C ABI.
- `cargo test` always builds with unwinding. Do not benchmark binaries that were
  left in `target/` by a test run.

## Measured and rejected

| Idea | Result |
|---|---|
| int8 activations (W8A8) | `pmaddubsw` saturates on AVX; exact A8 needs 7-bit weights or int16 widening, which is the current kernel. |
| int4 DPDFNet weights | GPTQ with groups of 32 loses 0.05–0.13 dB SI-SDR against 0.00 dB for int8; memory stalls beyond L2 are only 6% of the i3's cycles. |
| int8 GTCRN-AEC GRUs | The GRUs are 5% of the time; about 2% to gain and the reference comparison would break. |
| `vzeroupper` fixes | `other_assists.avx_to_sse` reads zero; there is no AVX→SSE transition penalty to remove. |
| Branchless Kalman select in the DAF | No change in cycles. |
