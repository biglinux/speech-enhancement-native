//! f32 and int16 SIMD kernels of the DPDFNet engine: dot product, AXPY, the
//! transposed matvec behind the grouped linears, the depthwise 1x3 convolution,
//! the GRU gate update and activation quantization. `simd_tier()` selects each
//! kernel once per CPU. The scalar fallbacks use exact libm math; the AVX GRU
//! gate uses a range-reduced polynomial (`exp8`/`tanh8`, ~1e-6 relative).

#![allow(clippy::needless_range_loop)] // numeric kernels index by design
#![allow(clippy::too_many_arguments)] // kernels take weights, bias and dimensions explicitly

/// Attenuation control (dB) to the dry-mix fraction: 0 dB or below keeps the
/// input (1.0), 100 dB and above is full reduction (0.0), and a non-finite
/// control means full reduction rather than a NaN gain.
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
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SIMD tier resolved once: 3 = AVX2+FMA (Haswell+), 2 = AVX (Sandy Bridge),
/// 1 = SSE4.1 (pre-AVX Core 2, Atom-class Celeron/Pentium), 0 = scalar. AVX2
/// does not imply FMA, and tier 3 kernels need both.
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
/// order matches the scalar loop; only FMA rounding differs on tier 3.
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

/// Quantise f32 -> i16 (per-vector abs-max / 16383, symmetric). Returns the scale.
/// 14-bit activations (~80 dB) keep int32 sums of up to 1024 int8 products
/// exact (127·16383·1024 < 2^31).
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
            2 | 3 => return unsafe { abs_max_avx(x) },
            1 => return unsafe { abs_max_sse(x) },
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
            3 => return unsafe { quantize::quantize_avx2(x, out, inv) },
            2 => return unsafe { quantize_round_avx(x, out, inv) },
            1 => {
                return unsafe { quantize::quantize_sse(x, out, inv) };
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

/// 8-wide exp (AVX, no FMA): range-reduce to `2^r · poly(f)`,
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
                    if n >= 64 {
                        matvec_wide::matvec_avx2(y, a, x, m, n);
                    } else {
                        matvec_t_avx2(y, a, x, m, n);
                    }
                };
                return;
            }
            2 => {
                // SAFETY: tier 2 means AVX is present.
                unsafe {
                    if n >= 64 {
                        matvec_wide::matvec_avx(y, a, x, m, n);
                    } else {
                        matvec_t_avx(y, a, x, m, n);
                    }
                };
                return;
            }
            1 => {
                // SAFETY: tier 1 means SSE4.1 is present (SSE is a subset).
                unsafe {
                    if n >= 32 {
                        matvec_wide::matvec_sse(y, a, x, m, n);
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

// The runtime dispatch only runs the tier of the CPU running the test, so these
// call each kernel directly and compare it with the scalar path over random
// shapes, including sizes that are not multiples of the SIMD width.
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
        fn pos(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 + 1e-4
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
}

/// Updates a GRU from already computed projections.
/// Gate order z,r,n. Bias order input z,r,n followed by recurrent z,r,n.
/// PyTorch reset-after equation; exporter explicitly reorders PyTorch r,z,n.
/// Allows batching all independent input projections before recurrent sweeps.
pub fn gru_update(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32]) {
    let hs = h.len();
    assert!(wx.len() >= 3 * hs && rh.len() >= 3 * hs && b.len() >= 6 * hs);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 && hs.is_multiple_of(8) {
        unsafe { gate8(h, wx, rh, b, hs) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if simd_tier() == 1 && hs.is_multiple_of(4) {
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
mod quantizer_edges {
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

#[cfg(all(test, target_arch = "x86_64"))]
mod exp8_tests;

#[cfg(target_arch = "x86_64")]
mod matvec_wide;

mod quantize;
