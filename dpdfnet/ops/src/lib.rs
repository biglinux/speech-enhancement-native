//! Numeric primitives for the DFN3 pipeline.
//!
//! Independent Rust implementation of the standard NN ops the model needs
//! (matvec, grouped linear, ONNX GRU cell, depthwise/pointwise conv on the
//! frequency axis). The scalar fallbacks use exact libm math (`exp`/`tanh`/`sin`);
//! the AVX GRU gate runs a vectorised range-reduced polynomial (`exp8`/`tanh8`,
//! ~1e-6 rel, >100 dB), which is the path that actually executes on the i3/i5.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design
#![allow(clippy::too_many_arguments)] // GRU/GEMV kernels take weight+bias+dims explicitly

#[inline]
pub fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Best-effort page-lock of a slice so it cannot be swapped or evicted during
/// realtime processing. A page fault in the audio callback costs milliseconds —
/// a guaranteed xrun; locking the weights removes that worst case under memory
/// pressure. Failure (e.g. a low `RLIMIT_MEMLOCK`) is ignored: the plugin still
/// runs, just without the guarantee.
pub fn mlock_slice<T>(s: &[T]) {
    if s.is_empty() {
        return;
    }
    // SAFETY: the pointer/length describe exactly this slice's allocation; mlock
    // only pins those pages in the page table and never dereferences the pointer.
    unsafe {
        libc::mlock(s.as_ptr().cast(), std::mem::size_of_val(s));
    }
}

/// Attenuation-limit control (dB) to the engine's noisy-mix fraction, so the
/// LADSPA plugin, the CLIs and tests all convert the same way. DeepFilterNet
/// semantics: 0 dB (or below) keeps the input (1.0), >=100 dB is full reduction
/// (0.0), and a non-finite control is treated as full reduction rather than
/// poisoning the output with NaN gains.
#[inline]
#[must_use]
pub fn atten_lim_from_db(db: f32) -> f32 {
    if !db.is_finite() || db >= 100.0 {
        0.0
    } else if db <= 0.0 {
        1.0
    } else {
        10f32.powf(-db / 20.0)
    }
}

// Exact libm math (scalar fallbacks; the AVX GRU gate uses a vectorised poly).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
pub fn tanh_f(x: f32) -> f32 {
    x.tanh()
}
#[inline]
pub fn log10_f(x: f32) -> f32 {
    x.log10()
}
#[inline]
pub fn sin_unit(x: f32) -> f32 {
    x.sin()
}

#[inline]
pub fn vadd(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

#[inline]
pub fn vdot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// SIMD tier resolved once: 3 = AVX2+FMA (Haswell+), 2 = AVX (Sandy Bridge),
/// 1 = SSE4.1 (pre-AVX Core 2 / Atom-class Celeron/Pentium), 0 = scalar. The int8
/// GEMV only needs SSE4.1, so tier 1 keeps cheap no-AVX CPUs off the scalar path.
///
/// Tier 3 explicitly checks AVX2 *and* FMA: they are independent features (AVX2
/// does not imply FMA), and this tier's kernels need both. Tier 2 uses AVX, which
/// is sufficient for the floating-point kernels. Integer GEMV dispatch checks
/// AVX2/SSE4.1 independently rather than assuming another CPUID bit implies them.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn simd_tier() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(u8::MAX);
    let c = CACHE.load(Ordering::Relaxed);
    if c != u8::MAX {
        return c;
    }
    let t = if cfg!(feature = "force-sse41") {
        if is_x86_feature_detected!("sse4.1") {
            1
        } else {
            0
        }
    } else if !cfg!(feature = "force-avx1")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        3
    } else if is_x86_feature_detected!("avx") {
        2
    } else if is_x86_feature_detected!("sse4.1") {
        1
    } else {
        0
    };
    CACHE.store(t, Ordering::Relaxed);
    t
}

/// Chooses the fused op inside the vdot kernel: real FMA on the avx2 expansion, a
/// separate mul+add on the avx expansion. A const generic can't do this because the
/// `_mm256_fmadd_ps` intrinsic is feature-gated at compile time and would reject the
/// non-fma monomorphization, so two macro expansions produce two correct bodies.
#[cfg(target_arch = "x86_64")]
macro_rules! fma_sel {
    (true, $x:expr, $y:expr, $acc:expr) => {
        std::arch::x86_64::_mm256_fmadd_ps($x, $y, $acc)
    };
    (false, $x:expr, $y:expr, $acc:expr) => {
        std::arch::x86_64::_mm256_add_ps($acc, std::arch::x86_64::_mm256_mul_ps($x, $y))
    };
}

/// f32 dot product, 8-wide with 4 independent accumulators to hide the multiply
/// latency on long vectors (STFT/iSTFT rows are 512 / 257 wide). `$fma` picks FMA
/// vs mul+add so the same loop serves both the avx2+fma and avx tiers.
#[cfg(target_arch = "x86_64")]
macro_rules! vdot_body {
    ($a:expr, $b:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (a, b) = ($a, $b);
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0usize;
        while i + 32 <= n {
            for l in 0..4 {
                let x = _mm256_loadu_ps(pa.add(i + l * 8));
                let y = _mm256_loadu_ps(pb.add(i + l * 8));
                acc[l] = fma_sel!($fma, x, y, acc[l]);
            }
            i += 32;
        }
        while i + 8 <= n {
            let x = _mm256_loadu_ps(pa.add(i));
            let y = _mm256_loadu_ps(pb.add(i));
            acc[0] = fma_sel!($fma, x, y, acc[0]);
            i += 8;
        }
        let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let lo = _mm256_castps256_ps128(s);
        let hi = _mm256_extractf128_ps::<1>(s);
        let q = _mm_add_ps(lo, hi);
        let q = _mm_hadd_ps(q, q);
        let q = _mm_hadd_ps(q, q);
        let mut out = _mm_cvtss_f32(q);
        while i < n {
            out += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        out
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vdot_avx2(a: &[f32], b: &[f32]) -> f32 {
    vdot_body!(a, b, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn vdot_avx(a: &[f32], b: &[f32]) -> f32 {
    vdot_body!(a, b, false)
}

/// f32 dot product. SIMD tiers (avx2+fma, avx) are not bit-identical to each other
/// or to the scalar fallback — callers that need cross-tier bit-identity must not
/// use this (the AEC's STFT/iSTFT tolerate it under the golden SDR>60 gate).
#[must_use]
pub fn vdot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vdot_f32: shape mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            3 => return unsafe { vdot_avx2(a, b) },
            2 => return unsafe { vdot_avx(a, b) },
            _ => {}
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// In-place AXPY: `y[i] += a * x[i]`, 8-wide. Element-wise, so the accumulation
/// order matches the scalar loop (only FMA rounding differs on tier 3). Used by the
/// AEC conv2d fast path; a fresh kernel, not the denoiser's `pointwise_conv2d`, so
/// the denoiser's cross-tier bit-identity is untouched.
#[cfg(target_arch = "x86_64")]
macro_rules! axpy_body {
    ($y:expr, $x:expr, $a:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (y, x, a) = ($y, $x, $a);
        let n = y.len();
        let av = _mm256_set1_ps(a);
        let mut i = 0usize;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            let yv = _mm256_loadu_ps(y.as_ptr().add(i));
            _mm256_storeu_ps(y.as_mut_ptr().add(i), fma_sel!($fma, av, xv, yv));
            i += 8;
        }
        while i < n {
            y[i] += a * x[i];
            i += 1;
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(y: &mut [f32], x: &[f32], a: f32) {
    axpy_body!(y, x, a, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn axpy_avx(y: &mut [f32], x: &[f32], a: f32) {
    axpy_body!(y, x, a, false)
}

/// Pointwise (1x1) conv `out[co][w] = bias[co] + Σ_ci weight[co][ci]·in[ci][w]`,
/// SIMD over width with the whole ci reduction inside one call (no per-(co,ci)
/// dispatch). `$fma` picks FMA vs mul+add. A fresh AEC kernel — not the denoiser's
/// scalar `pointwise_conv2d`, so its cross-tier bit-identity is untouched.
#[cfg(target_arch = "x86_64")]
macro_rules! pointwise_body {
    ($out:expr, $inp:expr, $w:expr, $b:expr, $ci_n:expr, $co_n:expr, $width:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (out, inp, w, b) = ($out, $inp, $w, $b);
        let (c_in, c_out, width) = ($ci_n, $co_n, $width);
        for co in 0..c_out {
            let o = &mut out[co * width..co * width + width];
            o.fill(b[co]);
            for ci in 0..c_in {
                let wt = w[co * c_in + ci];
                let wv = _mm256_set1_ps(wt);
                let inr = &inp[ci * width..ci * width + width];
                let mut f = 0;
                while f + 8 <= width {
                    let acc = _mm256_loadu_ps(o.as_ptr().add(f));
                    let xv = _mm256_loadu_ps(inr.as_ptr().add(f));
                    _mm256_storeu_ps(o.as_mut_ptr().add(f), fma_sel!($fma, wv, xv, acc));
                    f += 8;
                }
                while f < width {
                    o[f] += wt * inr[f];
                    f += 1;
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pointwise_avx2(
    out: &mut [f32],
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    pointwise_body!(out, inp, w, b, c_in, c_out, width, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pointwise_avx(
    out: &mut [f32],
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    pointwise_body!(out, inp, w, b, c_in, c_out, width, false)
}

/// Pointwise (1x1) conv, SIMD (see `pointwise_body`). `out` is `[c_out][width]`,
/// `inp` `[c_in][width]`, `w` `[c_out][c_in]`, `b` `[c_out]`.
pub fn pointwise_f32(
    out: &mut [f32],
    inp: &[f32],
    w: &[f32],
    b: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    assert!(out.len() >= c_out.checked_mul(width).expect("pointwise output overflow"));
    assert!(inp.len() >= c_in.checked_mul(width).expect("pointwise input overflow"));
    assert!(w.len() >= c_out.checked_mul(c_in).expect("pointwise weights overflow"));
    assert!(b.len() >= c_out);
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            3 => return unsafe { pointwise_avx2(out, inp, w, b, c_in, c_out, width) },
            2 => return unsafe { pointwise_avx(out, inp, w, b, c_in, c_out, width) },
            _ => {}
        }
    }
    for co in 0..c_out {
        let o = &mut out[co * width..co * width + width];
        o.fill(b[co]);
        for ci in 0..c_in {
            let wt = w[co * c_in + ci];
            let inr = &inp[ci * width..ci * width + width];
            for f in 0..width {
                o[f] += wt * inr[f];
            }
        }
    }
}

/// In-place AXPY `y += a*x` (see `axpy_body`). `x` and `y` are the same length.
pub fn axpy_f32(y: &mut [f32], x: &[f32], a: f32) {
    assert_eq!(y.len(), x.len(), "axpy_f32: shape mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            3 => return unsafe { axpy_avx2(y, x, a) },
            2 => return unsafe { axpy_avx(y, x, a) },
            _ => {}
        }
    }
    for i in 0..y.len() {
        y[i] += a * x[i];
    }
}

/// Eight GRU cells sharing one weight set, evaluated one per SIMD lane. Layout is
/// feature-major: `x[i*8+lane]` is input feature `i` of lane, `h[j*8+lane]` its
/// hidden state (updated in place). `wih`/`whh` are `[3*nh][nin]`/`[3*nh][nh]`
/// row-major, `bih`/`bhh` are `[3*nh]`; PyTorch gate order `[r,z,n]` with separate
/// input/hidden biases — identical math to the DAF's scalar `gru_cell`, so a caller
/// packs 8 independent bins/partitions per call. `$fma` selects FMA vs mul+add.
#[cfg(target_arch = "x86_64")]
macro_rules! gru8_body {
    ($x:expr, $nin:expr, $h:expr, $nh:expr, $wih:expr, $whh:expr, $bih:expr, $bhh:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (x, h, wih, whh, bih, bhh) = ($x, $h, $wih, $whh, $bih, $bhh);
        let (nin, nh) = ($nin, $nh);
        let mut gi = [_mm256_setzero_ps(); 48]; // max 3*nh = 48 (bins GRU nh=16)
        let mut gh = [_mm256_setzero_ps(); 48];
        for g in 0..3 * nh {
            let mut ai = _mm256_set1_ps(bih[g]);
            let wr = &wih[g * nin..g * nin + nin];
            for i in 0..nin {
                let xi = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                ai = fma_sel!($fma, _mm256_set1_ps(wr[i]), xi, ai);
            }
            gi[g] = ai;
            let mut ah = _mm256_set1_ps(bhh[g]);
            let vr = &whh[g * nh..g * nh + nh];
            for i in 0..nh {
                let hi = _mm256_loadu_ps(h.as_ptr().add(i * 8));
                ah = fma_sel!($fma, _mm256_set1_ps(vr[i]), hi, ah);
            }
            gh[g] = ah;
        }
        let one = _mm256_set1_ps(1.0);
        for j in 0..nh {
            let r = sigmoid8(_mm256_add_ps(gi[j], gh[j]));
            let z = sigmoid8(_mm256_add_ps(gi[nh + j], gh[nh + j]));
            let n = tanh8(_mm256_add_ps(
                gi[2 * nh + j],
                _mm256_mul_ps(r, gh[2 * nh + j]),
            ));
            let hprev = _mm256_loadu_ps(h.as_ptr().add(j * 8));
            let hnew = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(one, z), n),
                _mm256_mul_ps(z, hprev),
            );
            _mm256_storeu_ps(h.as_mut_ptr().add(j * 8), hnew);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gru8_avx2(
    x: &[f32],
    nin: usize,
    h: &mut [f32],
    nh: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
) {
    gru8_body!(x, nin, h, nh, wih, whh, bih, bhh, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gru8_avx(
    x: &[f32],
    nin: usize,
    h: &mut [f32],
    nh: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
) {
    gru8_body!(x, nin, h, nh, wih, whh, bih, bhh, false)
}

/// One scalar GRU lane (PyTorch `[r,z,n]`), the trusted reference the batched
/// kernel matches and the fallback on non-AVX targets. `x`/`h` are strided by 8
/// (feature-major), so lane `L` reads `x[i*8+L]` and updates `h[j*8+L]`.
#[inline]
fn gru_lane(
    x: &[f32],
    nin: usize,
    h: &mut [f32],
    nh: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
    lane: usize,
) {
    let mut gi = [0.0f32; 48];
    let mut gh = [0.0f32; 48];
    for g in 0..3 * nh {
        let mut s = bih[g];
        for i in 0..nin {
            s += wih[g * nin + i] * x[i * 8 + lane];
        }
        gi[g] = s;
        let mut t = bhh[g];
        for i in 0..nh {
            t += whh[g * nh + i] * h[i * 8 + lane];
        }
        gh[g] = t;
    }
    for j in 0..nh {
        let r = 1.0 / (1.0 + (-(gi[j] + gh[j])).exp());
        let z = 1.0 / (1.0 + (-(gi[nh + j] + gh[nh + j])).exp());
        let n = (gi[2 * nh + j] + r * gh[2 * nh + j]).tanh();
        h[j * 8 + lane] = (1.0 - z) * n + z * h[j * 8 + lane];
    }
}

/// Batched 8-lane GRU (see `gru8_body`). Dispatches to the AVX2+FMA / AVX kernels,
/// falling back to eight scalar `gru_lane` evaluations. `nh <= 16`.
pub fn gru8(
    x: &[f32],
    nin: usize,
    h: &mut [f32],
    nh: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
) {
    assert!(nh <= 16, "gru8: hidden width exceeds fixed scratch");
    let gates = 3 * nh; // nh has been bounded above.
    assert!(x.len() >= nin.checked_mul(8).expect("gru8 input overflow"));
    assert!(h.len() >= nh * 8);
    assert!(wih.len() >= gates.checked_mul(nin).expect("gru8 weights overflow"));
    assert!(whh.len() >= gates * nh);
    assert!(bih.len() >= gates && bhh.len() >= gates);
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            3 => return unsafe { gru8_avx2(x, nin, h, nh, wih, whh, bih, bhh) },
            2 => return unsafe { gru8_avx(x, nin, h, nh, wih, whh, bih, bhh) },
            _ => {}
        }
    }
    for lane in 0..8 {
        gru_lane(x, nin, h, nh, wih, whh, bih, bhh, lane);
    }
}

/// Quantise f32 -> i16 (per-vector abs-max / 16383, symmetric). Returns the scale.
/// 14-bit activation (~80 dB) keeps the int32 `pmaddwd` accumulator well below 2^31
/// (127·16383·512 ≈ 1.07e9) so the GEMV is exact-integer and deterministic.
pub fn quantize_i16(x: &[f32], out: &mut [i16]) -> f32 {
    assert!(out.len() >= x.len(), "quantize_i16: output too short");
    let amax = abs_max(x);
    let scale = if amax > 0.0 { amax / 16383.0 } else { 1.0 };
    if scale == 0.0 {
        // Values below representable row-scale precision. Do not form 0 * infinity.
        out[..x.len()].fill(0);
        return 0.0;
    }
    let inv = 1.0 / scale;
    if inv.is_finite() {
        quantize_round(x, out, inv);
    } else {
        // A tiny finite vector can have an overflowing reciprocal. Division is
        // bounded here even though 1/scale is not. No impact on the normal hot path.
        for (o, &v) in out.iter_mut().zip(x) {
            *o = (v / scale).round().clamp(-16383.0, 16383.0) as i16;
        }
    }
    scale
}

/// `max(|x|)`. For finite inputs the SIMD reductions are bit-identical to the
/// left-fold, since `max` over finite values is order-independent.
fn abs_max(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            2 | 3 => {
                return unsafe {
                    if cfg!(feature = "r8-amax") {
                        round8::amax_avx(x)
                    } else {
                        abs_max_avx(x)
                    }
                }
            }
            1 => {
                return unsafe {
                    if cfg!(feature = "r8-amax") {
                        round8::amax_sse(x)
                    } else {
                        abs_max_sse(x)
                    }
                }
            }
            _ => {}
        }
    }
    x.iter().fold(0f32, |a, &v| a.max(v.abs()))
}

/// `out[i] = round_ties_away(x[i]*inv)` clamped to ±16383. SIMD paths reproduce
/// Rust `f32::round` (half away from zero) via truncate + fractional compare, so
/// the emitted int16 is bit-identical to the scalar loop across every tier.
fn quantize_round(x: &[f32], out: &mut [i16], inv: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            #[cfg(feature = "r10-quant-pack")]
            3 => return unsafe { round10::quantize_avx2(x, out, inv) },
            #[cfg(not(feature = "r10-quant-pack"))]
            3 => return unsafe { quantize_round_avx(x, out, inv) },
            2 => return unsafe { quantize_round_avx(x, out, inv) },
            1 => {
                #[cfg(feature = "r10-quant-pack")]
                return unsafe { round10::quantize_sse(x, out, inv) };
                #[cfg(not(feature = "r10-quant-pack"))]
                return unsafe { quantize_round_sse(x, out, inv) };
            }
            _ => {}
        }
    }
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v * inv).round().clamp(-16383.0, 16383.0) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn abs_max_avx(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut m = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        m = _mm256_max_ps(
            m,
            _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i)), absmask),
        );
        i += 8;
    }
    let mut buf = [0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), m);
    let mut r = buf.iter().copied().fold(0f32, f32::max);
    while i < n {
        r = r.max((*x.get_unchecked(i)).abs());
        i += 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn abs_max_sse(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let mut m = _mm_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        m = _mm_max_ps(m, _mm_and_ps(_mm_loadu_ps(x.as_ptr().add(i)), absmask));
        i += 4;
    }
    let mut buf = [0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), m);
    let mut r = buf.iter().copied().fold(0f32, f32::max);
    while i < n {
        r = r.max((*x.get_unchecked(i)).abs());
        i += 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn quantize_round_avx(x: &[f32], out: &mut [i16], inv: f32) {
    use std::arch::x86_64::*;
    let vinv = _mm256_set1_ps(inv);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let lim = _mm256_set1_ps(16383.0);
    let nlim = _mm256_set1_ps(-16383.0);
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let signmask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let prod = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vinv);
        let trunc = _mm256_round_ps::<0x0B>(prod); // truncate toward zero, no exception
        let frac = _mm256_sub_ps(prod, trunc);
        let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_and_ps(frac, absmask), half);
        // copysign(1, prod), zeroed where |frac| < 0.5.
        let bump = _mm256_and_ps(_mm256_or_ps(one, _mm256_and_ps(prod, signmask)), ge);
        let r = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(trunc, bump), nlim), lim);
        let ri = _mm256_cvtps_epi32(r); // r is integer-valued and in range: exact
        let packed = _mm_packs_epi32(
            _mm256_castsi256_si128(ri),
            _mm256_extractf128_si256::<1>(ri),
        );
        _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), packed);
        i += 8;
    }
    while i < n {
        *out.get_unchecked_mut(i) =
            (*x.get_unchecked(i) * inv).round().clamp(-16383.0, 16383.0) as i16;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn quantize_round_sse(x: &[f32], out: &mut [i16], inv: f32) {
    use std::arch::x86_64::*;
    let vinv = _mm_set1_ps(inv);
    let half = _mm_set1_ps(0.5);
    let one = _mm_set1_ps(1.0);
    let lim = _mm_set1_ps(16383.0);
    let nlim = _mm_set1_ps(-16383.0);
    let absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let signmask = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        let prod = _mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i)), vinv);
        let trunc = _mm_round_ps::<0x0B>(prod);
        let frac = _mm_sub_ps(prod, trunc);
        let ge = _mm_cmpge_ps(_mm_and_ps(frac, absmask), half);
        let bump = _mm_and_ps(_mm_or_ps(one, _mm_and_ps(prod, signmask)), ge);
        let r = _mm_min_ps(_mm_max_ps(_mm_add_ps(trunc, bump), nlim), lim);
        let ri = _mm_cvtps_epi32(r);
        let packed = _mm_packs_epi32(ri, ri);
        _mm_storel_epi64(out.as_mut_ptr().add(i).cast(), packed);
        i += 4;
    }
    while i < n {
        *out.get_unchecked_mut(i) =
            (*x.get_unchecked(i) * inv).round().clamp(-16383.0, 16383.0) as i16;
        i += 1;
    }
}

/// W8A16 GEMV: `y[i] = (Σ_j q[i,j]·xq[j]) · rowscale[i] · xscale`, integer `pmaddwd`
/// core (needs only SSE4.1, so it runs on cheap no-AVX CPUs and the AVX-only i3).
/// Exact integer accumulation, so the result is bit-identical across all tiers.
pub fn matvec_i8_i16(
    y: &mut [f32],
    q: &[i8],
    ws: &[f32],
    xq: &[i16],
    xscale: f32,
    m: usize,
    n: usize,
) {
    // The public safe API must validate before entering raw-pointer kernels.
    assert!(y.len() >= m && ws.len() >= m && xq.len() >= n);
    assert!(q.len() >= m.checked_mul(n).expect("GEMV dimensions overflow"));
    // Arbitrary callers may use full-range i16, not only our 14-bit quantizer.
    // Bound the absolute sum so EVERY partial i32 accumulator is representable.
    // n <= 511 is always safe. Larger sums take a wide scalar fallback (not used by shipped model shapes).
    let i32_safe = n <= 511
        || xq[..n].iter().fold(0u64, |sum, &x| {
            sum.saturating_add(i64::from(x).unsigned_abs())
        }) <= i32::MAX as u64 / 128;
    if !i32_safe {
        for i in 0..m {
            let row = &q[i * n..(i + 1) * n];
            let sum = row
                .iter()
                .zip(&xq[..n])
                .fold(0i128, |a, (&w, &x)| a + i128::from(w) * i128::from(x));
            y[i] = sum as f32 * ws[i] * xscale;
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !cfg!(feature = "force-sse41")
            && !cfg!(feature = "force-avx1")
            && is_x86_feature_detected!("avx2")
        {
            // SAFETY: exact feature and all buffer lengths checked above.
            unsafe { matvec_i8_i16_avx2(y, q, ws, xq, xscale, m, n) };
            return;
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: exact feature and all buffer lengths checked above.
            unsafe { matvec_i8_i16_sse(y, q, ws, xq, xscale, m, n) };
            return;
        }
    }
    for i in 0..m {
        let row = &q[i * n..i * n + n];
        let mut s = 0i32;
        for j in 0..n {
            s += row[j] as i32 * xq[j] as i32;
        }
        y[i] = s as f32 * ws[i] * xscale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hsum_i32(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b_01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b_10_11_00_01));
    _mm_cvtsi128_si32(s)
}

/// 8-wide exp (AVX, no FMA — runs on the i3): range-reduce to `2^r · poly(f)`,
/// degree-5 Taylor on |f| ≤ ln2/2 (~1e-6 rel, >100 dB). `2^r` reconstructed via
/// SSE 128-bit int shifts so no AVX2 is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-87.0)),
        _mm256_set1_ps(88.0),
    );
    let r = _mm256_round_ps::<0x08>(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    let f = _mm256_sub_ps(x, _mm256_mul_ps(r, _mm256_set1_ps(std::f32::consts::LN_2)));
    let mut p = _mm256_set1_ps(1.0 / 120.0);
    let step = |p, c| _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(c));
    p = step(p, 1.0 / 24.0);
    p = step(p, 1.0 / 6.0);
    p = step(p, 0.5);
    p = step(p, 1.0);
    p = step(p, 1.0);
    #[cfg(feature = "oldcpu-exp-bits")]
    let pow2 = {
        // r is integral and in [-126,127]. (r+127)*2^23 is exactly representable
        // as f32 AND in positive i32 range. CVTTPS2DQ exists in AVX1 at 256 bits.
        // Casting the resulting INTEGER BIT PATTERN to f32 constructs 2^r.
        // No AVX2 shift, rounding change, reciprocal, or approximate exponent.
        let bits = _mm256_cvttps_epi32(_mm256_mul_ps(
            _mm256_add_ps(r, _mm256_set1_ps(127.0)),
            _mm256_set1_ps(8_388_608.0),
        ));
        _mm256_castsi256_ps(bits)
    };
    #[cfg(not(feature = "oldcpu-exp-bits"))]
    let pow2 = {
        let ri = _mm256_cvtps_epi32(r);
        let bias = _mm_set1_epi32(127);
        let lo = _mm_slli_epi32::<23>(_mm_add_epi32(_mm256_castsi256_si128(ri), bias));
        let hi = _mm_slli_epi32::<23>(_mm_add_epi32(_mm256_extractf128_si256::<1>(ri), bias));
        _mm256_set_m128(_mm_castsi128_ps(hi), _mm_castsi128_ps(lo))
    };
    _mm256_mul_ps(p, pow2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sigmoid8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn tanh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let s = sigmoid8(_mm256_mul_ps(x, _mm256_set1_ps(2.0)));
    _mm256_sub_ps(_mm256_mul_ps(s, _mm256_set1_ps(2.0)), _mm256_set1_ps(1.0))
}

/// Vectorised log10 for x > 0 (8 lanes). Splits off the IEEE exponent (128-bit int
/// halves, so AVX suffices — same idiom as `exp8`), then a degree-7 atanh series for
/// log2 of the mantissa. ~1e-4 relative, enough for the DAF controller features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn log10_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let bits = _mm256_castps_si256(x);
    let lo = _mm256_castsi256_si128(bits);
    let hi = _mm256_extractf128_si256::<1>(bits);
    let exp_half =
        |b: __m128i| _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32::<23>(b), _mm_set1_epi32(127)));
    let e = _mm256_set_m128(exp_half(hi), exp_half(lo));
    let mant_half = |b: __m128i| {
        _mm_castsi128_ps(_mm_or_si128(
            _mm_and_si128(b, _mm_set1_epi32(0x007f_ffff)),
            _mm_set1_epi32(0x3f80_0000),
        ))
    };
    let m = _mm256_set_m128(mant_half(hi), mant_half(lo)); // mantissa in [1,2)
    let one = _mm256_set1_ps(1.0);
    let t = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one)); // (m-1)/(m+1), |t|<=1/3
    let t2 = _mm256_mul_ps(t, t);
    // log2(m) = (2/ln2)·(t + t^3/3 + t^5/5 + t^7/7)
    let mut p = _mm256_set1_ps(1.0 / 7.0);
    p = _mm256_add_ps(_mm256_mul_ps(p, t2), _mm256_set1_ps(1.0 / 5.0));
    p = _mm256_add_ps(_mm256_mul_ps(p, t2), _mm256_set1_ps(1.0 / 3.0));
    p = _mm256_add_ps(_mm256_mul_ps(p, t2), one);
    let log2m = _mm256_mul_ps(
        _mm256_mul_ps(p, t),
        _mm256_set1_ps(2.0 / std::f32::consts::LN_2),
    );
    let log2x = _mm256_add_ps(e, log2m);
    _mm256_mul_ps(log2x, _mm256_set1_ps(std::f32::consts::LOG10_2))
}

/// In-place vectorised `out[i] = log10(out[i])` for `out[i] > 0`. AVX path where
/// available (tiers 2/3), else scalar libm. ~1e-4 rel on the SIMD path — callers
/// needing exact log10 must use the scalar `log10_f`.
pub fn log10_slice(out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_tier() >= 2 {
            // SAFETY: tier >= 2 means AVX is present.
            unsafe {
                use std::arch::x86_64::*;
                let n = out.len();
                let mut i = 0;
                while i + 8 <= n {
                    if out[i..i + 8].iter().all(|x| x.is_normal() && *x > 0.0) {
                        let v = _mm256_loadu_ps(out.as_ptr().add(i));
                        _mm256_storeu_ps(out.as_mut_ptr().add(i), log10_8(v));
                    } else {
                        // IEEE zero/subnormal/negative/infinite/NaN semantics.
                        for x in &mut out[i..i + 8] {
                            *x = x.log10();
                        }
                    }
                    i += 8;
                }
                for v in &mut out[i..] {
                    *v = v.log10();
                }
            }
            return;
        }
    }
    for v in out {
        *v = v.log10();
    }
}

/// Vectorised GRU gate update (`linear_before_reset=1`), 8 lanes/iter. `hs` % 8 == 0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gate8(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i < hs {
        let ld = |s: &[f32], o: usize| _mm256_loadu_ps(s.as_ptr().add(o));
        let z = sigmoid8(_mm256_add_ps(
            _mm256_add_ps(ld(wx, i), ld(rh, i)),
            _mm256_add_ps(ld(b, i), ld(b, 3 * hs + i)),
        ));
        let rr = sigmoid8(_mm256_add_ps(
            _mm256_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
            _mm256_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
        ));
        let inner = _mm256_mul_ps(rr, _mm256_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
        let hh = tanh8(_mm256_add_ps(
            _mm256_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
            inner,
        ));
        let hprev = ld(h, i);
        let newh = _mm256_add_ps(
            _mm256_mul_ps(_mm256_sub_ps(one, z), hh),
            _mm256_mul_ps(z, hprev),
        );
        _mm256_storeu_ps(h.as_mut_ptr().add(i), newh);
        i += 8;
    }
}

/// SSE4.1 4-lane `exp`, `sigmoid`, `tanh` and gate update: identical arithmetic to
/// `exp8`/`sigmoid8`/`tanh8`/`gate8` at 128-bit width. IEEE ops are per-lane, so each
/// lane is bit-identical to the AVX path — old CPUs get the same approximation as
/// modern ones, not the scalar-libm fallback.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn exp4(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    let x = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-87.0)), _mm_set1_ps(88.0));
    let r = _mm_round_ps::<0x08>(_mm_mul_ps(x, _mm_set1_ps(std::f32::consts::LOG2_E)));
    let f = _mm_sub_ps(x, _mm_mul_ps(r, _mm_set1_ps(std::f32::consts::LN_2)));
    let mut p = _mm_set1_ps(1.0 / 120.0);
    let step = |p, c| _mm_add_ps(_mm_mul_ps(p, f), _mm_set1_ps(c));
    p = step(p, 1.0 / 24.0);
    p = step(p, 1.0 / 6.0);
    p = step(p, 0.5);
    p = step(p, 1.0);
    p = step(p, 1.0);
    let ri = _mm_cvtps_epi32(r);
    let pow2 = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(ri, _mm_set1_epi32(127))));
    _mm_mul_ps(p, pow2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sigmoid4(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    let e = exp4(_mm_sub_ps(_mm_setzero_ps(), x));
    _mm_div_ps(_mm_set1_ps(1.0), _mm_add_ps(_mm_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn tanh4(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    let s = sigmoid4(_mm_mul_ps(x, _mm_set1_ps(2.0)));
    _mm_sub_ps(_mm_mul_ps(s, _mm_set1_ps(2.0)), _mm_set1_ps(1.0))
}

/// Vectorised GRU gate update, 4 lanes/iter. `hs` % 4 == 0. See `gate8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn gate4(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    use std::arch::x86_64::*;
    let one = _mm_set1_ps(1.0);
    let mut i = 0;
    while i < hs {
        let ld = |s: &[f32], o: usize| _mm_loadu_ps(s.as_ptr().add(o));
        let z = sigmoid4(_mm_add_ps(
            _mm_add_ps(ld(wx, i), ld(rh, i)),
            _mm_add_ps(ld(b, i), ld(b, 3 * hs + i)),
        ));
        let rr = sigmoid4(_mm_add_ps(
            _mm_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
            _mm_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
        ));
        let inner = _mm_mul_ps(rr, _mm_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
        let hh = tanh4(_mm_add_ps(
            _mm_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
            inner,
        ));
        let hprev = ld(h, i);
        let newh = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(one, z), hh), _mm_mul_ps(z, hprev));
        _mm_storeu_ps(h.as_mut_ptr().add(i), newh);
        i += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matvec_i8_i16_avx2(
    y: &mut [f32],
    q: &[i8],
    ws: &[f32],
    xq: &[i16],
    xscale: f32,
    m: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let n16 = n & !15;
    let qp = q.as_ptr();
    let xp = xq.as_ptr();
    for i in 0..m {
        let row = qp.add(i * n);
        let mut acc = _mm256_setzero_si256();
        let mut j = 0;
        while j < n16 {
            let wi16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(j) as *const __m128i)); // 16 i8->i16
            let xi16 = _mm256_loadu_si256(xp.add(j) as *const __m256i);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wi16, xi16)); // 8 i32
            j += 16;
        }
        let s128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256(acc, 1),
        );
        let mut s = hsum_i32(s128);
        while j < n {
            s += *row.add(j) as i32 * *xp.add(j) as i32;
            j += 1;
        }
        y[i] = s as f32 * *ws.get_unchecked(i) * xscale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn matvec_i8_i16_sse(
    y: &mut [f32],
    q: &[i8],
    ws: &[f32],
    xq: &[i16],
    xscale: f32,
    m: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let n8 = n & !7;
    let qp = q.as_ptr();
    let xp = xq.as_ptr();
    for i in 0..m {
        let row = qp.add(i * n);
        let mut acc = _mm_setzero_si128();
        let mut j = 0;
        while j < n8 {
            let wi16 = _mm_cvtepi8_epi16(_mm_loadl_epi64(row.add(j) as *const __m128i)); // 8 i8->i16
            let xi16 = _mm_loadu_si128(xp.add(j) as *const __m128i);
            acc = _mm_add_epi32(acc, _mm_madd_epi16(wi16, xi16)); // 4 i32
            j += 8;
        }
        let mut s = hsum_i32(acc);
        while j < n {
            s += *row.add(j) as i32 * *xp.add(j) as i32;
            j += 1;
        }
        y[i] = s as f32 * *ws.get_unchecked(i) * xscale;
    }
}

/// `y[N] = A[M,N]^T * x[M]`, A row-major (accumulate columns / SAXPY).
#[inline]
pub fn matvec_t(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m);
    assert!(a.len() >= m.checked_mul(n).expect("matvec_t dimensions overflow"));
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            3 => {
                // SAFETY: tier 3 means AVX2+FMA is present.
                unsafe {
                    if cfg!(feature = "r8-matvec-wide") && n >= 64 {
                        round8::matvec_avx2(y, a, x, m, n);
                    } else {
                        matvec_t_avx2(y, a, x, m, n);
                    }
                };
                return;
            }
            2 => {
                // SAFETY: tier 2 means AVX is present.
                unsafe {
                    if cfg!(feature = "r8-matvec-wide") && n >= 64 {
                        round8::matvec_avx(y, a, x, m, n);
                    } else {
                        matvec_t_avx(y, a, x, m, n);
                    }
                };
                return;
            }
            1 => {
                // SAFETY: tier 1 means SSE4.1 is present (SSE is a subset).
                unsafe {
                    if cfg!(feature = "r8-matvec-wide") && n >= 32 {
                        round8::matvec_sse(y, a, x, m, n);
                    } else {
                        matvec_t_sse(y, a, x, m, n);
                    }
                };
                return;
            }
            _ => {}
        }
    }
    for j in 0..n {
        let mut s = 0.0f32;
        for i in 0..m {
            s += x[i] * a[i * n + j];
        }
        y[j] = s;
    }
}

/// Output-tiled transposed matvec: hold each output column-block in registers
/// across the whole `m` reduction (one store per column, not `m` load/store
/// round-trips). Per-column accumulation order is `i = 0..m`, so tiers 1/2 and
/// the scalar path stay bit-identical; tier 3 differs only by FMA rounding.
#[cfg(target_arch = "x86_64")]
macro_rules! matvec_t_body {
    ($y:expr, $a:expr, $x:expr, $m:expr, $n:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (y, a, x, m, n) = ($y, $a, $x, $m, $n);
        let ap = a.as_ptr();
        let yp = y.as_mut_ptr();
        let xp = x.as_ptr();
        let mut j = 0usize;
        while j + 32 <= n {
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            for i in 0..m {
                let xi = _mm256_set1_ps(*xp.add(i));
                let row = ap.add(i * n + j);
                a0 = fma_sel!($fma, _mm256_loadu_ps(row), xi, a0);
                a1 = fma_sel!($fma, _mm256_loadu_ps(row.add(8)), xi, a1);
                a2 = fma_sel!($fma, _mm256_loadu_ps(row.add(16)), xi, a2);
                a3 = fma_sel!($fma, _mm256_loadu_ps(row.add(24)), xi, a3);
            }
            _mm256_storeu_ps(yp.add(j), a0);
            _mm256_storeu_ps(yp.add(j + 8), a1);
            _mm256_storeu_ps(yp.add(j + 16), a2);
            _mm256_storeu_ps(yp.add(j + 24), a3);
            j += 32;
        }
        while j + 8 <= n {
            let mut acc = _mm256_setzero_ps();
            for i in 0..m {
                let xi = _mm256_set1_ps(*xp.add(i));
                acc = fma_sel!($fma, _mm256_loadu_ps(ap.add(i * n + j)), xi, acc);
            }
            _mm256_storeu_ps(yp.add(j), acc);
            j += 8;
        }
        while j < n {
            let mut s = 0.0f32;
            for i in 0..m {
                s += *xp.add(i) * *ap.add(i * n + j);
            }
            *yp.add(j) = s;
            j += 1;
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn matvec_t_avx(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    matvec_t_body!(y, a, x, m, n, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_t_avx2(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    matvec_t_body!(y, a, x, m, n, true)
}

/// SSE4.1 fallback for the transposed matvec: same output-tiling as the AVX
/// paths at 128-bit width, so SSE-only CPUs leave the scalar path. Non-FMA, so
/// bit-identical to the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn matvec_t_sse(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    use std::arch::x86_64::*;
    let ap = a.as_ptr();
    let yp = y.as_mut_ptr();
    let xp = x.as_ptr();
    let mut j = 0usize;
    while j + 16 <= n {
        let mut a0 = _mm_setzero_ps();
        let mut a1 = _mm_setzero_ps();
        let mut a2 = _mm_setzero_ps();
        let mut a3 = _mm_setzero_ps();
        for i in 0..m {
            let xi = _mm_set1_ps(*xp.add(i));
            let row = ap.add(i * n + j);
            a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(row), xi));
            a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(row.add(4)), xi));
            a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_loadu_ps(row.add(8)), xi));
            a3 = _mm_add_ps(a3, _mm_mul_ps(_mm_loadu_ps(row.add(12)), xi));
        }
        _mm_storeu_ps(yp.add(j), a0);
        _mm_storeu_ps(yp.add(j + 4), a1);
        _mm_storeu_ps(yp.add(j + 8), a2);
        _mm_storeu_ps(yp.add(j + 12), a3);
        j += 16;
    }
    while j + 4 <= n {
        let mut acc = _mm_setzero_ps();
        for i in 0..m {
            let xi = _mm_set1_ps(*xp.add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(ap.add(i * n + j)), xi));
        }
        _mm_storeu_ps(yp.add(j), acc);
        j += 4;
    }
    while j < n {
        let mut s = 0.0f32;
        for i in 0..m {
            s += *xp.add(i) * *ap.add(i * n + j);
        }
        *yp.add(j) = s;
        j += 1;
    }
}

/// Grouped linear (einsum `g i -> g o`): per group, `y_g = W_g^T @ x_g`,
/// with `W_g` laid out `[in_per_group, out_per_group]`.
pub fn grouped_linear(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    groups: usize,
    in_pg: usize,
    out_pg: usize,
) {
    #[cfg(all(target_arch = "x86_64", feature = "r9-grouped-fusion"))]
    if simd_tier() == 3 && matches!(out_pg, 8 | 16 | 32) && groups >= 2 {
        let ni = groups.checked_mul(in_pg).expect("grouped input overflow");
        let no = groups.checked_mul(out_pg).expect("grouped output overflow");
        let nw = ni.checked_mul(out_pg).expect("grouped weights overflow");
        assert!(x.len() >= ni && y.len() >= no && w.len() >= nw);
        // SAFETY: validated slices; tier 3 requires AVX2 and FMA.
        unsafe { round9::grouped(y, x, w, groups, in_pg, out_pg) };
        return;
    }
    for g in 0..groups {
        let xg = &x[g * in_pg..g * in_pg + in_pg];
        let wg = &w[g * in_pg * out_pg..g * in_pg * out_pg + in_pg * out_pg];
        let yg = &mut y[g * out_pg..g * out_pg + out_pg];
        matvec_t(yg, wg, xg, in_pg, out_pg);
    }
}

/// Depthwise conv, kt=1, kf=3, groups=channels. `src` is one input frame
/// `[in_f, co]` (channel-contiguous); `w` is `[3, co]` tap weights; result (bias +
/// present taps, no activation) is written to `out[f*spacing+offset .. +co]` for
/// f in 0..of. Each output channel-tile is held in a register across the three
/// frequency taps, so `out` is written once per position instead of read-modified
/// three times. Non-FMA (mul+add), bit-identical to the scalar tap loop it replaces.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_1x3(
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    assert!(w.len() >= 3 * co && b.len() >= co);
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            2 | 3 => {
                // SAFETY: tier >= 2 means AVX is present.
                unsafe {
                    depthwise_1x3_avx(out, src, w, b, co, of, in_f, stride, pad, spacing, offset)
                };
                return;
            }
            1 => {
                // SAFETY: tier 1 means SSE (SSE4.1 detected).
                unsafe {
                    depthwise_1x3_sse(out, src, w, b, co, of, in_f, stride, pad, spacing, offset)
                };
                return;
            }
            _ => {}
        }
    }
    for f in 0..of {
        let dst = &mut out[f * spacing + offset..f * spacing + offset + co];
        dst.copy_from_slice(&b[..co]);
        for k in 0..3 {
            let fr = f * stride + k;
            if fr < pad || fr - pad >= in_f {
                continue;
            }
            let xr = &src[(fr - pad) * co..(fr - pad) * co + co];
            let wk = &w[k * co..k * co + co];
            for c in 0..co {
                dst[c] += xr[c] * wk[c];
            }
        }
    }
}

/// Gathers the present taps for one output position: pointers into `src`/`w` for
/// each frequency tap that is inside `[0, in_f)`. At most three.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dw_taps(
    src: &[f32],
    w: &[f32],
    co: usize,
    f: usize,
    stride: usize,
    pad: usize,
    in_f: usize,
) -> ([(*const f32, *const f32); 3], usize) {
    let mut taps = [(std::ptr::null(), std::ptr::null()); 3];
    let mut n = 0;
    for k in 0..3 {
        let fr = f * stride + k;
        if fr >= pad && fr - pad < in_f {
            taps[n] = (src[(fr - pad) * co..].as_ptr(), w[k * co..].as_ptr());
            n += 1;
        }
    }
    (taps, n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_1x3_avx(
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    use std::arch::x86_64::*;
    let bp = b.as_ptr();
    for f in 0..of {
        let (taps, nt) = dw_taps(src, w, co, f, stride, pad, in_f);
        let dst = out.as_mut_ptr().add(f * spacing + offset);
        let mut c = 0;
        while c + 8 <= co {
            let mut acc = _mm256_loadu_ps(bp.add(c));
            for &(xr, wk) in &taps[..nt] {
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(_mm256_loadu_ps(wk.add(c)), _mm256_loadu_ps(xr.add(c))),
                );
            }
            _mm256_storeu_ps(dst.add(c), acc);
            c += 8;
        }
        while c < co {
            let mut s = *bp.add(c);
            for &(xr, wk) in &taps[..nt] {
                s += *wk.add(c) * *xr.add(c);
            }
            *dst.add(c) = s;
            c += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_1x3_sse(
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    use std::arch::x86_64::*;
    let bp = b.as_ptr();
    for f in 0..of {
        let (taps, nt) = dw_taps(src, w, co, f, stride, pad, in_f);
        let dst = out.as_mut_ptr().add(f * spacing + offset);
        let mut c = 0;
        while c + 4 <= co {
            let mut acc = _mm_loadu_ps(bp.add(c));
            for &(xr, wk) in &taps[..nt] {
                acc = _mm_add_ps(
                    acc,
                    _mm_mul_ps(_mm_loadu_ps(wk.add(c)), _mm_loadu_ps(xr.add(c))),
                );
            }
            _mm_storeu_ps(dst.add(c), acc);
            c += 4;
        }
        while c < co {
            let mut s = *bp.add(c);
            for &(xr, wk) in &taps[..nt] {
                s += *wk.add(c) * *xr.add(c);
            }
            *dst.add(c) = s;
            c += 1;
        }
    }
}

/// int8-weight GRU cell, W8A16: activations quantised to int16 per matvec, `pmaddwd`
/// integer core; gates and state stay f32. `xq16` is scratch, len >= max(input, hs).
#[allow(clippy::too_many_arguments)]
pub fn gru_cell_q(
    h: &mut [f32],
    x: &[f32],
    wq: &[i8],
    ws: &[f32],
    rq: &[i8],
    rs: &[f32],
    b: &[f32],
    hs: usize,
    input: usize,
    scratch: &mut [f32],
    xq16: &mut [i16],
) {
    let gates = hs.checked_mul(3).expect("GRU dimensions overflow");
    let biases = hs.checked_mul(6).expect("GRU bias dimensions overflow");
    assert!(h.len() >= hs && x.len() >= input && b.len() >= biases);
    assert!(scratch.len() >= biases && xq16.len() >= input.max(hs));
    assert!(wq.len() >= gates.checked_mul(input).expect("GRU weights overflow"));
    assert!(
        rq.len()
            >= gates
                .checked_mul(hs)
                .expect("GRU recurrent weights overflow")
    );
    assert!(ws.len() >= gates && rs.len() >= gates);
    let x = &x[..input];
    let h = &mut h[..hs];
    let (wx, rest) = scratch.split_at_mut(gates);
    let rh = &mut rest[..3 * hs];
    let sx = quantize_i16(x, &mut xq16[..input]);
    matvec_i8_i16(wx, wq, ws, &xq16[..input], sx, 3 * hs, input);
    let sh = quantize_i16(h, &mut xq16[..hs]);
    matvec_i8_i16(rh, rq, rs, &xq16[..hs], sh, 3 * hs, hs);

    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 && (hs % 8 == 0) {
        // SAFETY: tier >= 2 means AVX is present; hs is a multiple of 8.
        unsafe { gate8(h, wx, rh, b, hs) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if simd_tier() == 1 && (hs % 4 == 0) {
        // SAFETY: tier 1 means SSE4.1 is present; hs is a multiple of 4.
        unsafe { gate4(h, wx, rh, b, hs) };
        return;
    }

    let (wbz, wbr, wbh) = (&b[0..hs], &b[hs..2 * hs], &b[2 * hs..3 * hs]);
    let (rbz, rbr, rbh) = (&b[3 * hs..4 * hs], &b[4 * hs..5 * hs], &b[5 * hs..6 * hs]);

    for i in 0..hs {
        let z = sigmoid(wx[i] + rh[i] + wbz[i] + rbz[i]);
        let rr = sigmoid(wx[hs + i] + rh[hs + i] + wbr[i] + rbr[i]);
        let hh = tanh_f(wx[2 * hs + i] + wbh[i] + rr * (rh[2 * hs + i] + rbh[i]));
        h[i] = (1.0 - z) * hh + z * h[i];
    }
}

/// Depthwise 1-D conv on the frequency axis, kernel 3, stride 1, pad 1.
/// Accumulates one temporal row (`w0,w1,w2` are that row's taps) into `out`.
pub fn dw_row_k3s1_accum(out: &mut [f32], src: &[f32], w0: f32, w1: f32, w2: f32) {
    let width = out.len();
    if width == 0 {
        return;
    }
    out[0] += w1 * src[0] + if width > 1 { w2 * src[1] } else { 0.0 };
    for ow in 1..width.saturating_sub(1) {
        out[ow] += w0 * src[ow - 1] + w1 * src[ow] + w2 * src[ow + 1];
    }
    if width > 1 {
        out[width - 1] += w0 * src[width - 2] + w1 * src[width - 1];
    }
}

/// Depthwise 1-D conv, kernel 3, stride 2, pad 1. `out.len() == src.len()/2`.
pub fn dw_row_k3s2_accum(out: &mut [f32], src: &[f32], w0: f32, w1: f32, w2: f32) {
    let w_in = src.len();
    for ow in 0..out.len() {
        let il = 2 * ow as isize - 1;
        let im = 2 * ow;
        let ir = 2 * ow + 1;
        let mut s = w1 * src[im];
        if il >= 0 {
            s += w0 * src[il as usize];
        }
        if ir < w_in {
            s += w2 * src[ir];
        }
        out[ow] += s;
    }
}

/// Pointwise (1x1) conv: `in[C_in, W]`, `weight[C_out, C_in]`, `bias[C_out]`
/// -> `out[C_out, W]`.
pub fn pointwise_conv2d(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    // co outer / ci inner: each output row is accumulated in place (stays hot in
    // L1/registers) while the small input tile is re-read from L1. Same summation
    // order as before -> bit-identical, but far fewer out load-modify-store passes.
    for co in 0..c_out {
        let o = &mut out[co * width..co * width + width];
        o.fill(bias[co]);
        for ci in 0..c_in {
            let wt = weight[co * c_in + ci];
            let in_row = &input[ci * width..ci * width + width];
            for w in 0..width {
                o[w] += wt * in_row[w];
            }
        }
    }
}

/// Post-model silence expander: a level-keyed downward expander that ducks the
/// residual noise floor during pauses **without muting** (finite depth), so quiet
/// wanted background (music) survives at low depth.
///
/// Why level-based and not the model's SNR gate: DFN3 normalises its input by a
/// running per-band mean, so absolute level is invisible to it — the residual
/// mic/room floor in a pause reaches the net looking like "noise at normal scale"
/// and its LSNR gate never fires. The information that decides "this is silence"
/// is the absolute output level, so the gate has to key on that.
///
/// Runs after the model on the processed hop. Pure scalar f32 (only per-sample
/// multiplies in the hot path; the one `sqrt`/`powf` are per hop), so it is
/// bit-identical across SIMD tiers and never touches the W8A16 arithmetic. With
/// `depth_db <= 0` it is an exact pass-through (the shipped golden vectors run at
/// depth 0 and are unaffected).
pub struct SilenceExpander {
    /// Adaptive noise-floor estimate, linear RMS (minimum statistics).
    floor: f32,
    /// Current smoothed gain, ramped per sample to avoid zipper noise.
    gain: f32,
}

impl Default for SilenceExpander {
    fn default() -> Self {
        Self::new()
    }
}

impl SilenceExpander {
    /// Floor tracker: instant drop to a lower RMS, slow rise (+~6 dB/s) otherwise.
    const FLOOR_RISE: f32 = 1.0069;
    /// Floor clamped to a plausible noise range so a loud passage or dead silence
    /// cannot drive the thresholds absurdly high or low. 10^(-90/20)..10^(-30/20).
    const FLOOR_MIN: f32 = 3.162_278e-5;
    const FLOOR_MAX: f32 = 0.031_622_78;
    /// Soft-knee thresholds, in dB above the floor. At/below `KNEE_LO` the signal
    /// is treated as pure floor and gets the full depth; at/above `KNEE_HI` it is
    /// signal and passes untouched; between, the attenuation eases smoothly. A
    /// wide knee (not a hard gate) is what stops a fluctuating residual from
    /// snapping the gain fully open and letting bursts of hiss through — it just
    /// modulates the duck a little. Relative to the floor, so it is independent of
    /// mic gain and host (ffmpeg/mpv/PipeWire). Speech sits well above KNEE_HI.
    const KNEE_LO_DB: f32 = 6.0;
    const KNEE_HI_DB: f32 = 20.0;
    /// One-pole ramp coefficients: attack ~2 ms, release ~120 ms at 48 kHz.
    /// `1 - exp(-1/(t·sr))`. The release is the graceful hold over speech gaps.
    const ATTACK: f32 = 0.010_362_6;
    const RELEASE: f32 = 0.000_173_6;

    #[must_use]
    pub fn new() -> Self {
        Self {
            // Start at the top of the range, not the bottom: the floor tracker
            // drops instantly but rises only slowly, so a low start would take
            // seconds to climb to a real noise floor (leaving it disengaged the
            // whole time). Starting high lets the first quiet hop instant-drop the
            // floor to the true level, so it engages within a hop.
            floor: Self::FLOOR_MAX,
            gain: 1.0,
        }
    }

    /// Duck the processed hop `buf` by up to `depth_db` while it sits near the
    /// noise floor. `depth_db <= 0` (or non-finite) is an exact pass-through.
    pub fn process_hop(&mut self, buf: &mut [f32], depth_db: f32) {
        if !depth_db.is_finite() || depth_db <= 0.0 || buf.is_empty() {
            return;
        }
        // Per-hop level and adaptive floor.
        let ms: f32 = buf.iter().map(|&s| s * s).sum::<f32>() / buf.len() as f32;
        let rms = ms.sqrt();
        // Only track the floor from frames with real content. The model's own gate
        // hard-zeros the frames it judges pure noise, and a zeroed frame would
        // otherwise crash the floor to the bottom — then the *residual* frames it
        // leaks would read as huge signal above that crashed floor and escape the
        // duck entirely. Skipping them keeps the floor at the residual level, so
        // the residual gets ducked and the already-silent frames stay silent.
        if rms >= Self::FLOOR_MIN {
            self.floor = rms
                .min(self.floor * Self::FLOOR_RISE)
                .clamp(Self::FLOOR_MIN, Self::FLOOR_MAX);
        }
        // How far the hop sits above the floor, in dB, mapped through the soft
        // knee to the attenuation to apply (0 at/above KNEE_HI, full depth
        // at/below KNEE_LO, linear between).
        let excess_db = 20.0 * (rms / self.floor).max(1e-9).log10();
        let frac = ((Self::KNEE_HI_DB - excess_db) / (Self::KNEE_HI_DB - Self::KNEE_LO_DB))
            .clamp(0.0, 1.0);
        let target = 10f32.powf(-depth_db * frac / 20.0);
        for s in buf {
            let coeff = if target > self.gain {
                Self::ATTACK
            } else {
                Self::RELEASE
            };
            self.gain += (target - self.gain) * coeff;
            *s *= self.gain;
        }
    }
}

#[cfg(test)]
mod expander_tests {
    use super::SilenceExpander;

    const HOP: usize = 480;

    fn energy(buf: &[f32]) -> f64 {
        buf.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
    }

    fn tone(amp: f32, phase: &mut f32) -> [f32; HOP] {
        let mut b = [0.0f32; HOP];
        for s in &mut b {
            *s = amp * phase.sin();
            *phase += 0.4;
        }
        b
    }

    #[test]
    fn depth_zero_is_exact_passthrough() {
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..50 {
            let orig = tone(5e-4, &mut ph);
            let mut buf = orig;
            ex.process_hop(&mut buf, 0.0);
            assert_eq!(buf, orig, "depth 0 must not touch the samples");
        }
    }

    #[test]
    fn a_sustained_noise_floor_is_learned_and_ducked() {
        // Minimum-statistics: a steady low level is learned as the floor and
        // gated; only bursts well above it (speech) keep the gate open. Feed a
        // constant low level long enough for the floor to rise to it.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        let mut last = 1.0;
        // ~1 s only: the floor must engage fast (it starts high and drops to the
        // real level), not creep up from the bottom over many seconds.
        for _ in 0..100 {
            let mut buf = tone(5e-4, &mut ph);
            ex.process_hop(&mut buf, 24.0);
            last = (energy(&buf) / energy(&tone(5e-4, &mut 0.0))).sqrt();
        }
        assert!(last < 0.15, "floor should duck within ~1 s, got {last:.3}");
    }

    #[test]
    fn loud_signal_passes() {
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        let mut last = 1.0;
        // Well above the floor ceiling (as speech is above a real noise floor),
        // so the soft knee leaves it untouched.
        for _ in 0..40 {
            let ref_buf = tone(0.9, &mut 0.0);
            let mut buf = tone(0.9, &mut ph);
            ex.process_hop(&mut buf, 24.0);
            last = (energy(&buf) / energy(&ref_buf)).sqrt();
        }
        assert!(last > 0.95, "loud signal should pass ~unity, got {last:.3}");
    }

    #[test]
    fn a_weak_sustained_tone_is_heavily_ducked_voice01() {
        // VOICE-01: a steady -23 dBFS tone (amp 0.1 -> RMS 0.0707) is a plausible
        // quiet-voice level, yet with no speech classifier the minimum-statistics
        // floor learns it and ducks it hard — matching the review's independent
        // NumPy reproduction (~28 dB at depth 30). This is NOT a human-speech
        // measurement; it documents that the expander cannot tell weak sustained
        // speech from a noise floor, so an aggressive gate is unsafe as a default.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        let mut atten_db = 0.0;
        for _ in 0..300 {
            let refb = tone(0.1, &mut 0.0);
            let mut buf = tone(0.1, &mut ph);
            ex.process_hop(&mut buf, 30.0);
            let ratio = (energy(&buf) / energy(&refb)).sqrt();
            atten_db = 20.0 * ratio.max(1e-6).log10();
        }
        eprintln!("VOICE-01 sustained -23 dBFS tone, depth 30: {atten_db:.1} dB attenuation");
        assert!(
            atten_db < -12.0,
            "weak sustained tone should be ducked, got {atten_db:.1} dB"
        );
    }

    #[test]
    fn speech_onset_recovers_within_a_hop_and_is_not_swallowed() {
        // VOICE-01 asks for speech-onset behaviour before any aggressive default.
        // After a quiet floor ducks the gate, the ~2 ms attack must reopen it
        // inside the first 10 ms hop so the start of a word is not chopped.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..120 {
            let mut b = tone(5e-4, &mut ph);
            ex.process_hop(&mut b, 30.0);
        }
        let refb = tone(0.9, &mut 0.0);
        let mut onset = tone(0.9, &mut ph);
        ex.process_hop(&mut onset, 30.0);
        let onset_ratio = (energy(&onset) / energy(&refb)).sqrt();
        let mut steady = 0.0;
        for _ in 0..5 {
            let r = tone(0.9, &mut 0.0);
            let mut b = tone(0.9, &mut ph);
            ex.process_hop(&mut b, 30.0);
            steady = (energy(&b) / energy(&r)).sqrt();
        }
        eprintln!("VOICE-01 onset: first-hop {onset_ratio:.3}, steady {steady:.3}");
        assert!(
            steady > 0.95,
            "steady speech must pass ~unity, got {steady:.3}"
        );
        assert!(
            onset_ratio > 0.7,
            "speech onset swallowed: {onset_ratio:.3}"
        );
    }

    #[test]
    fn a_brief_gap_between_words_is_held_not_chopped() {
        // VOICE-01 speech-offset behaviour: the slow (~120 ms) release must hold
        // the gate open across a short pause so the next word is not attenuated
        // as if speech had ended.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..40 {
            let mut b = tone(5e-4, &mut ph);
            ex.process_hop(&mut b, 30.0);
        }
        for _ in 0..40 {
            let mut b = tone(0.9, &mut ph);
            ex.process_hop(&mut b, 30.0);
        }
        // A ~20 ms gap between words.
        for _ in 0..2 {
            let mut b = tone(5e-4, &mut ph);
            ex.process_hop(&mut b, 30.0);
        }
        let refb = tone(0.9, &mut 0.0);
        let mut word = tone(0.9, &mut ph);
        ex.process_hop(&mut word, 30.0);
        let ratio = (energy(&word) / energy(&refb)).sqrt();
        eprintln!("VOICE-01 word after 20 ms gap: {ratio:.3}");
        assert!(
            ratio > 0.9,
            "word after a brief gap was chopped: {ratio:.3}"
        );
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut ex = SilenceExpander::new();
            let mut ph = 0.0;
            let mut out = Vec::new();
            for hop in 0..30 {
                // out.len() advances by 480 and is always even: use the hop
                // index so this test actually alternates loud and quiet audio.
                let mut buf = tone(if hop % 2 == 0 { 0.3 } else { 5e-4 }, &mut ph);
                ex.process_hop(&mut buf, 24.0);
                out.extend_from_slice(&buf);
            }
            out
        };
        assert_eq!(run(), run());
    }
}

// The W8A16 GEMV claims a bit-identical result across scalar / SSE4.1 / AVX2
// because the accumulation is exact integer. The runtime dispatch only ever
// runs the tier of the CPU running the test, so this forces each kernel
// directly (it can, since every tier is available on an AVX2 host) and compares
// them to an independent scalar reference over thousands of random shapes,
// including sizes that are not multiples of the SIMD width so the remainder
// paths are exercised.
#[cfg(all(test, target_arch = "x86_64"))]
mod simd_equiv {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn i8v(&mut self) -> i8 {
            (self.next() >> 56) as i8
        }
        fn i16v(&mut self) -> i16 {
            // Activation range produced by quantize_i16: |x| <= 16383.
            ((self.next() >> 40) as i64 % 32767 - 16383) as i16
        }
        fn pos(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 + 1e-4
        }
    }

    fn reference(q: &[i8], ws: &[f32], xq: &[i16], xscale: f32, m: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for i in 0..m {
            let mut s = 0i32;
            for j in 0..n {
                s += q[i * n + j] as i32 * xq[j] as i32;
            }
            y[i] = s as f32 * ws[i] * xscale;
        }
        y
    }

    #[test]
    fn matvec_i8_i16_bit_identical_across_tiers() {
        let mut r = Lcg(0x1234_5678_9abc_def0);
        let shapes = [
            (1usize, 1usize),
            (3, 7),
            (8, 16),
            (16, 17),
            (32, 31),
            (13, 64),
            (64, 127),
            (48, 512),
        ];
        for _ in 0..300 {
            for &(m, n) in &shapes {
                let q: Vec<i8> = (0..m * n).map(|_| r.i8v()).collect();
                let ws: Vec<f32> = (0..m).map(|_| r.pos()).collect();
                let xq: Vec<i16> = (0..n).map(|_| r.i16v()).collect();
                let xscale = r.pos();
                let want = reference(&q, &ws, &xq, xscale, m, n);

                let mut got = vec![0.0f32; m];
                matvec_i8_i16(&mut got, &q, &ws, &xq, xscale, m, n);
                assert_eq!(got, want, "dispatch m={m} n={n}");

                if is_x86_feature_detected!("sse4.1") {
                    let mut g = vec![0.0f32; m];
                    // SAFETY: guarded by the sse4.1 detection above.
                    unsafe { matvec_i8_i16_sse(&mut g, &q, &ws, &xq, xscale, m, n) };
                    assert_eq!(g, want, "sse m={m} n={n}");
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let mut g = vec![0.0f32; m];
                    // SAFETY: guarded by the avx2+fma detection above.
                    unsafe { matvec_i8_i16_avx2(&mut g, &q, &ws, &xq, xscale, m, n) };
                    assert_eq!(g, want, "avx2 m={m} n={n}");
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn quantize_round_matches_scalar_including_ties() {
        // Exact half-integer ties (round away from zero), their neighbours, both
        // signs, zero, and the ±16383 clamp edge.
        let mut xs: Vec<f32> = Vec::new();
        for k in -20i32..=20 {
            for d in [-0.5f32, -0.4, 0.0, 0.4, 0.5, 0.6] {
                xs.push(k as f32 + d);
            }
        }
        xs.extend([16383.0, 16383.5, -16383.5, 16384.0, -16384.0, 0.0, -0.0]);
        let mut r = Lcg(0x5151_5151_2626_2626);
        for _ in 0..2000 {
            xs.push((r.pos() - r.pos()) * 20000.0);
        }
        let want: Vec<i16> = xs
            .iter()
            .map(|&v| (v * 1.0f32).round().clamp(-16383.0, 16383.0) as i16)
            .collect();
        if is_x86_feature_detected!("avx") {
            let mut g = vec![0i16; xs.len()];
            // SAFETY: avx detected.
            unsafe { quantize_round_avx(&xs, &mut g, 1.0) };
            assert_eq!(g, want, "avx");
        }
        if is_x86_feature_detected!("sse4.1") {
            let mut g = vec![0i16; xs.len()];
            // SAFETY: sse4.1 detected.
            unsafe { quantize_round_sse(&xs, &mut g, 1.0) };
            assert_eq!(g, want, "sse");
        }
        // Full pipeline (amax + scale + round) against a scalar oracle.
        let scalar_quant = |x: &[f32]| -> (f32, Vec<i16>) {
            let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let scale = if amax > 0.0 { amax / 16383.0 } else { 1.0 };
            if scale == 0.0 {
                return (0.0, vec![0; x.len()]);
            }
            let inv = 1.0 / scale;
            let q = x
                .iter()
                .map(|&v| (v * inv).round().clamp(-16383.0, 16383.0) as i16)
                .collect();
            (scale, q)
        };
        for len in [1usize, 3, 7, 8, 15, 64, 256] {
            let x: Vec<f32> = (0..len).map(|_| (r.pos() - r.pos()) * 12.0).collect();
            let (ws, wq) = scalar_quant(&x);
            let mut gq = vec![0i16; len];
            let gs = quantize_i16(&x, &mut gq);
            assert_eq!(gs, ws, "scale len={len}");
            assert_eq!(gq, wq, "quant len={len}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn depthwise_1x3_matches_scalar_across_tiers() {
        let mut r = Lcg(0xdeed_beef_0bad_c0de);
        // co covers 8-wide (avx) and 4-wide (sse) tiles with and without remainder.
        let cases = [
            (64usize, 40usize, 80usize, 1usize, 1usize),
            (64, 20, 40, 2, 1),
            (32, 13, 39, 3, 1),
            (12, 7, 7, 1, 0),
            (64, 96, 96, 1, 1),
        ];
        for &(co, of, in_f, stride, pad) in &cases {
            let src: Vec<f32> = (0..in_f * co).map(|_| r.pos() - r.pos()).collect();
            let w: Vec<f32> = (0..3 * co).map(|_| r.pos() - r.pos()).collect();
            let b: Vec<f32> = (0..co).map(|_| r.pos() - r.pos()).collect();
            let mut want = vec![0.0f32; of * co];
            for f in 0..of {
                let dst = &mut want[f * co..f * co + co];
                dst.copy_from_slice(&b);
                for k in 0..3 {
                    let fr = f * stride + k;
                    if fr < pad || fr - pad >= in_f {
                        continue;
                    }
                    let xr = &src[(fr - pad) * co..(fr - pad) * co + co];
                    for c in 0..co {
                        dst[c] += xr[c] * w[k * co + c];
                    }
                }
            }
            if is_x86_feature_detected!("avx") {
                let mut g = vec![0.0f32; of * co];
                // SAFETY: guarded by the avx detection above.
                unsafe {
                    depthwise_1x3_avx(&mut g, &src, &w, &b, co, of, in_f, stride, pad, co, 0)
                };
                assert_eq!(g, want, "avx co={co} of={of} stride={stride}");
            }
            if is_x86_feature_detected!("sse4.1") {
                let mut g = vec![0.0f32; of * co];
                // SAFETY: guarded by the sse4.1 detection above (sse is a subset).
                unsafe {
                    depthwise_1x3_sse(&mut g, &src, &w, &b, co, of, in_f, stride, pad, co, 0)
                };
                assert_eq!(g, want, "sse co={co} of={of} stride={stride}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gate4_bit_identical_to_gate8() {
        if !(is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma"))
        {
            return;
        }
        let mut r = Lcg(0x00c0_ffee_1234_5678);
        for &hs in &[8usize, 64, 256] {
            for _ in 0..200 {
                // GRU pre-activations span a few units either side of zero.
                let g = |r: &mut Lcg| (r.pos() - r.pos()) * 4.0;
                let wx: Vec<f32> = (0..3 * hs).map(|_| g(&mut r)).collect();
                let rh: Vec<f32> = (0..3 * hs).map(|_| g(&mut r)).collect();
                let b: Vec<f32> = (0..6 * hs).map(|_| g(&mut r)).collect();
                let h0: Vec<f32> = (0..hs).map(|_| g(&mut r)).collect();
                let mut h8 = h0.clone();
                let mut h4 = h0.clone();
                // SAFETY: features gated above.
                unsafe {
                    gate8(&mut h8, &wx, &rh, &b, hs);
                    gate4(&mut h4, &wx, &rh, &b, hs);
                }
                assert_eq!(h8, h4, "hs={hs}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn matvec_t_matches_scalar_across_tiers() {
        let mut r = Lcg(0x0fed_cba9_8765_4321);
        // n hits every output-tile boundary (32/8/scalar for AVX, 16/4/scalar for SSE).
        let shapes = [
            (1usize, 1usize),
            (5, 7),
            (16, 32),
            (48, 33),
            (7, 48),
            (64, 96),
            (32, 130),
        ];
        for _ in 0..200 {
            for &(m, n) in &shapes {
                let a: Vec<f32> = (0..m * n).map(|_| r.pos() - r.pos()).collect();
                let x: Vec<f32> = (0..m).map(|_| r.pos() - r.pos()).collect();
                let mut want = vec![0.0f32; n];
                for j in 0..n {
                    let mut s = 0.0f32;
                    for i in 0..m {
                        s += x[i] * a[i * n + j];
                    }
                    want[j] = s;
                }
                if is_x86_feature_detected!("sse4.1") {
                    let mut g = vec![0.0f32; n];
                    // SAFETY: guarded by the sse4.1 detection above.
                    unsafe { matvec_t_sse(&mut g, &a, &x, m, n) };
                    assert_eq!(g, want, "sse m={m} n={n}");
                }
                if is_x86_feature_detected!("avx") {
                    let mut g = vec![0.0f32; n];
                    // SAFETY: guarded by the avx detection above.
                    unsafe { matvec_t_avx(&mut g, &a, &x, m, n) };
                    assert_eq!(g, want, "avx m={m} n={n}");
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let mut g = vec![0.0f32; n];
                    // SAFETY: guarded by the avx2+fma detection above. FMA reassociates
                    // the multiply-add, so this path is close, not bit-identical.
                    unsafe { matvec_t_avx2(&mut g, &a, &x, m, n) };
                    for (k, (&got, &wnt)) in g.iter().zip(&want).enumerate() {
                        assert!(
                            (got - wnt).abs() <= 1e-4 * (1.0 + wnt.abs()),
                            "avx2 m={m} n={n} j={k}: {got} vs {wnt}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod gru8_tests {
    use super::{gru8, gru_lane};

    // Small deterministic LCG in [-0.5, 0.5); keeps GRU gates in their active range.
    fn fill(v: &mut [f32], seed: &mut u32) {
        for x in v {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *x = (*seed >> 8) as f32 / 16_777_216.0 - 0.5;
        }
    }

    // The batched kernel must match eight independent scalar GRU lanes. On x86 this
    // pits the AVX(2) path against the libm scalar reference (SIMD sigmoid/tanh are
    // ~1e-6); a transpose/index bug shows up as an O(1) error, far above tol.
    fn check(nin: usize, nh: usize) {
        let mut seed = 7u32;
        let mut wih = vec![0.0f32; 3 * nh * nin];
        let mut whh = vec![0.0f32; 3 * nh * nh];
        let mut bih = vec![0.0f32; 3 * nh];
        let mut bhh = vec![0.0f32; 3 * nh];
        let mut x = vec![0.0f32; nin * 8];
        let mut h = vec![0.0f32; nh * 8];
        fill(&mut wih, &mut seed);
        fill(&mut whh, &mut seed);
        fill(&mut bih, &mut seed);
        fill(&mut bhh, &mut seed);
        fill(&mut x, &mut seed);
        fill(&mut h, &mut seed);

        let mut h_ref = h.clone();
        for lane in 0..8 {
            gru_lane(&x, nin, &mut h_ref, nh, &wih, &whh, &bih, &bhh, lane);
        }
        gru8(&x, nin, &mut h, nh, &wih, &whh, &bih, &bhh);

        let maxd = h
            .iter()
            .zip(&h_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("gru8 nin={nin} nh={nh} max abs diff vs scalar = {maxd:e}");
        for (a, b) in h.iter().zip(&h_ref) {
            assert!((a - b).abs() < 1e-3, "gru8 nin={nin} nh={nh}: {a} vs {b}");
        }
    }

    #[test]
    fn matches_scalar_bins_gru() {
        check(18, 16); // DAF per-bin controller
    }

    #[test]
    fn matches_scalar_partition_gru() {
        check(10, 8); // DAF per-partition controller
    }
}

#[cfg(test)]
mod log10_tests {
    use super::log10_slice;

    #[test]
    fn matches_libm() {
        let mut seed = 3u32;
        let mut g = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16_777_216.0
        };
        // positive inputs spanning a wide range (as the DAF features do)
        let x: Vec<f32> = (0..133)
            .map(|_| (g() * 2.0 - 1.0).exp() * (1.0 + g() * 1e4) + 1e-10)
            .collect();
        let mut y = x.clone();
        log10_slice(&mut y);
        for (a, r) in y.iter().zip(&x) {
            let want = r.log10();
            assert!(
                (a - want).abs() <= 1e-3 * want.abs().max(1.0),
                "log10 {r}: {a} vs {want}"
            );
        }
    }
}

#[cfg(test)]
mod safe_boundary_tests {
    use super::*;

    #[test]
    #[should_panic(expected = "vdot_f32: shape mismatch")]
    fn vdot_rejects_mismatched_shape_before_dispatch() {
        let _ = vdot_f32(&[0.0; 8], &[]);
    }

    #[test]
    #[should_panic(expected = "axpy_f32: shape mismatch")]
    fn axpy_rejects_mismatched_shape_before_dispatch() {
        axpy_f32(&mut [0.0; 8], &[], 1.0);
    }

    #[test]
    fn gemv_handles_full_i16_without_wrapping() {
        let q = [-128i8; 512];
        let x = [i16::MIN; 512];
        let mut y = [0.0];
        matvec_i8_i16(&mut y, &q, &[1.0], &x, 1.0, 1, 512);
        assert_eq!(y[0], 2147483648.0);
    }

    #[test]
    fn log10_edges_match_scalar_semantics() {
        let input = [
            0.0f32,
            -1.0,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(1),
            f32::MIN_POSITIVE / 2.0,
            1.0,
            10.0,
        ];
        let mut output = input;
        log10_slice(&mut output);
        for (got, x) in output.into_iter().zip(input) {
            let expected = x.log10();
            assert!(got == expected || (got.is_nan() && expected.is_nan()));
        }
    }
}

/// DPDFNet addition: update a GRU from already-computed projections.
/// Gate order z,r,n. Bias order input z,r,n followed by recurrent z,r,n.
/// PyTorch reset-after equation; exporter explicitly reorders PyTorch r,z,n.
/// Allows batching all independent input projections before recurrent sweeps.
pub fn gru_update(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32]) {
    let hs = h.len();
    assert!(wx.len() >= 3 * hs && rh.len() >= 3 * hs && b.len() >= 6 * hs);
    #[cfg(all(feature = "oldcpu-rational-gates", target_arch = "x86_64"))]
    if oldcpu_rational::try_update(h, wx, rh, b) {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 && hs % 8 == 0 {
        unsafe { gate8(h, wx, rh, b, hs) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if simd_tier() == 1 && hs % 4 == 0 {
        unsafe { gate4(h, wx, rh, b, hs) };
        return;
    }
    for i in 0..hs {
        let z = sigmoid(wx[i] + rh[i] + b[i] + b[3 * hs + i]);
        let r = sigmoid(wx[hs + i] + rh[hs + i] + b[hs + i] + b[4 * hs + i]);
        let n = (wx[2 * hs + i] + b[2 * hs + i] + r * (rh[2 * hs + i] + b[5 * hs + i])).tanh();
        h[i] = (1.0 - z) * n + z * h[i];
    }
}

#[cfg(test)]
mod dpdfnet_quantizer_edges {
    use super::*;
    #[test]
    fn tiny_vectors_do_not_saturate_from_an_infinite_reciprocal() {
        let x = [1e-38f32, 0.5e-38, 0.0, -0.5e-38];
        let mut q = [0; 4];
        let scale = quantize_i16(&x, &mut q);
        assert!(scale > 0.0 && scale.is_finite());
        assert!(q[1] > 7500 && q[1] < 8500 && q[2] == 0 && q[3] == -q[1]);
    }
    #[test]
    fn underflowed_scale_does_not_produce_nan() {
        let mut q = [1; 4];
        let scale = quantize_i16(&[f32::from_bits(1); 4], &mut q);
        assert_eq!(scale, 0.0);
        assert_eq!(q, [0; 4]);
    }
}

#[cfg(all(feature = "oldcpu-rational-gates", target_arch = "x86_64"))]
mod oldcpu_rational;

#[cfg(all(test, target_arch = "x86_64"))]
mod oldcpu_tests;

#[cfg(target_arch = "x86_64")]
#[path = "round8.rs"]
mod round8;

#[cfg(all(target_arch = "x86_64", any(feature = "r9-grouped-fusion", test)))]
mod round9;

#[cfg(any(feature = "r10-quant-pack", feature = "r10-linear-plan"))]
mod round10;
#[cfg(feature = "r10-linear-plan")]
pub use round10::GroupedLinearPlan;
