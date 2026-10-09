//! Numeric kernels shared by the DeepFilterNet3, DPDFNet, GTCRN-AEC and Silero
//! engines: f32 and W8A16 matrix products, GRU cells, convolutions on the
//! frequency axis, activation quantization and the 48/16 kHz resamplers.
//!
//! [`simd_tier`] picks each kernel once per CPU: scalar, SSE4.1, AVX or AVX2+FMA.
//! The scalar fallbacks use exact libm math; the SIMD GRU gates and `log10` use
//! range-reduced polynomials. Each kernel's docs say which tiers give the same bits.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design
#![allow(clippy::too_many_arguments)] // kernels take weights, bias and dimensions explicitly

/// Picks the multiply-add inside the 256-bit kernels: real FMA in the AVX2
/// expansion, a separate multiply and add in the AVX one. A const generic cannot do
/// this because `_mm256_fmadd_ps` is gated on a target feature at compile time and
/// would reject the non-FMA monomorphization.
#[cfg(target_arch = "x86_64")]
macro_rules! fma_sel {
    (true, $x:expr, $y:expr, $acc:expr) => {
        std::arch::x86_64::_mm256_fmadd_ps($x, $y, $acc)
    };
    (false, $x:expr, $y:expr, $acc:expr) => {
        std::arch::x86_64::_mm256_add_ps($acc, std::arch::x86_64::_mm256_mul_ps($x, $y))
    };
}

mod conv;
mod gru;
mod matvec;
pub mod pack_format;
mod packed;
mod quant;
pub mod resample;

pub use conv::{depthwise_1x3, dw_row_k3s1_accum, dw_row_k3s2_accum, pointwise_conv2d};
pub use gru::{gru_cell_packed, gru_update, gru8};
pub use matvec::{grouped_linear, matvec_t};
pub use quant::quantize_i16;

/// Sets flush-to-zero and denormals-are-zero on the calling thread and restores
/// the previous MXCSR when dropped, so the host and the other plugins on its audio
/// thread keep their floating-point environment. Recurrent state decays toward
/// denormals in silence, and denormal arithmetic on x86 is slow enough to push a
/// hop into an xrun.
///
/// The guard is not `Send`: dropping it on another thread would load this
/// thread's MXCSR there.
#[must_use]
pub struct DenormalGuard {
    #[cfg(target_arch = "x86_64")]
    saved: u32,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl DenormalGuard {
    #[inline]
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let saved = read_mxcsr();
            // FTZ = bit 15, DAZ = bit 6.
            write_mxcsr(saved | 0x8040);
            Self {
                saved,
                _not_send: std::marker::PhantomData,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        Self {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Default for DenormalGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DenormalGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        write_mxcsr(self.saved);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn read_mxcsr() -> u32 {
    let mut csr = 0u32;
    // SAFETY: stmxcsr stores the 32-bit MXCSR to a valid, writable u32.
    unsafe {
        std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut csr, options(nostack, preserves_flags));
    }
    csr
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn write_mxcsr(csr: u32) {
    // SAFETY: the value comes from stmxcsr with at most FTZ/DAZ changed, so ldmxcsr
    // cannot fault. Rust assumes the default FP environment (see `_mm_setcsr`): a
    // denormal folded at compile time may differ from one computed at run time,
    // which only changes values and never a memory access.
    unsafe {
        std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &csr, options(nostack, readonly, preserves_flags));
    }
}

#[inline]
pub fn relu_inplace(x: &mut [f32]) {
    // An unconditional store of a select vectorizes; a store only when negative
    // branches on every element. -0.0 and NaN pass through.
    for v in x {
        *v = if *v < 0.0 { 0.0 } else { *v };
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

/// Attenuation-limit control (dB) to the engine's noisy-mix fraction, so every
/// plugin, CLI and test converts the same way. DeepFilterNet semantics: 0 dB (or
/// below) keeps the input (1.0), >=100 dB is full reduction (0.0), and a
/// non-finite control is treated as full reduction rather than poisoning the
/// output with NaN gains.
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

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn vadd(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// Dot product summed left to right, for callers that need the same bits on every
/// tier. [`vdot_f32`] is the faster SIMD form.
#[inline]
pub fn vdot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// SIMD tier, resolved once: 3 = AVX2+FMA (Haswell+), 2 = AVX (Sandy Bridge),
/// 1 = SSE4.1 (pre-AVX Core 2, Atom-class Celeron/Pentium), 0 = scalar. The int8
/// GEMV only needs SSE4.1, so tier 1 keeps cheap no-AVX CPUs off the scalar path.
///
/// Tier 3 checks AVX2 *and* FMA: they are independent features and its kernels
/// need both. The `force-sse41` and `force-avx1` diagnostic features cap the tier
/// to compare tiers on one machine.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn simd_tier() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(u8::MAX);
    let c = CACHE.load(Ordering::Relaxed);
    if c != u8::MAX {
        return c;
    }
    let t = if !cfg!(any(feature = "force-sse41", feature = "force-avx1"))
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        3
    } else if !cfg!(feature = "force-sse41") && is_x86_feature_detected!("avx") {
        2
    } else if is_x86_feature_detected!("sse4.1") {
        1
    } else {
        0
    };
    CACHE.store(t, Ordering::Relaxed);
    t
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_tier() -> u8 {
    0
}

/// f32 dot product, 8-wide with 4 independent accumulators to hide the multiply
/// latency on long vectors. `$fma` picks FMA or a multiply and add.
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

/// # Safety
/// The CPU must support AVX2 and FMA, and `b` must be at least as long as `a`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vdot_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: every load reads below `a.len()`, which `b` covers too.
    unsafe { vdot_body!(a, b, true) }
}

/// # Safety
/// The CPU must support AVX, and `b` must be at least as long as `a`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn vdot_avx(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: every load reads below `a.len()`, which `b` covers too.
    unsafe { vdot_body!(a, b, false) }
}

/// f32 dot product. The SIMD tiers reassociate the sum, so they are not
/// bit-identical to each other or to [`vdot`].
#[must_use]
pub fn vdot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vdot_f32: shape mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            // SAFETY: tier 3 guarantees AVX2+FMA; the lengths are equal.
            3 => return unsafe { vdot_avx2(a, b) },
            // SAFETY: tier 2 guarantees AVX; the lengths are equal.
            2 => return unsafe { vdot_avx(a, b) },
            _ => {}
        }
    }
    vdot(a, b)
}

/// In-place AXPY `y[i] += a * x[i]`, 8-wide. Element-wise, so the accumulation
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

/// # Safety
/// The CPU must support AVX2 and FMA, and `x` must be at least as long as `y`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(y: &mut [f32], x: &[f32], a: f32) {
    // SAFETY: every load and store stays below `y.len()`, which `x` covers too.
    unsafe { axpy_body!(y, x, a, true) }
}

/// # Safety
/// The CPU must support AVX, and `x` must be at least as long as `y`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn axpy_avx(y: &mut [f32], x: &[f32], a: f32) {
    // SAFETY: every load and store stays below `y.len()`, which `x` covers too.
    unsafe { axpy_body!(y, x, a, false) }
}

/// In-place AXPY `y += a*x` over equal-length slices.
pub fn axpy_f32(y: &mut [f32], x: &[f32], a: f32) {
    assert_eq!(y.len(), x.len(), "axpy_f32: shape mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            // SAFETY: tier 3 guarantees AVX2+FMA; the lengths are equal.
            3 => return unsafe { axpy_avx2(y, x, a) },
            // SAFETY: tier 2 guarantees AVX; the lengths are equal.
            2 => return unsafe { axpy_avx(y, x, a) },
            _ => {}
        }
    }
    for i in 0..y.len() {
        y[i] += a * x[i];
    }
}

/// 8-wide log10 for normal positive input. Splits off the IEEE exponent with
/// 128-bit integer halves, so AVX suffices, then a degree-7 atanh series gives
/// log2 of the mantissa. About 1e-4 relative error.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn log10_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
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

/// In-place `out[i] = log10(out[i])`. On tiers 2 and 3, groups of eight normal
/// positive values take the AVX polynomial (about 1e-4 relative error); everything
/// else keeps libm's result and IEEE edge cases. Callers that need exact log10
/// must use `f32::log10`.
pub fn log10_slice(out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        use std::arch::x86_64::*;
        let (chunks, rest) = out.as_chunks_mut::<8>();
        for c in chunks {
            if c.iter().all(|x| x.is_normal() && *x > 0.0) {
                // SAFETY: tier 2 guarantees AVX; `c` holds exactly eight values.
                unsafe { _mm256_storeu_ps(c.as_mut_ptr(), log10_8(_mm256_loadu_ps(c.as_ptr()))) };
            } else {
                for x in c {
                    *x = x.log10();
                }
            }
        }
        for v in rest {
            *v = v.log10();
        }
        return;
    }
    for v in out {
        *v = v.log10();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn denormal_guard_restores_the_callers_fp_environment() {
        let before = read_mxcsr();
        {
            let _guard = DenormalGuard::new();
            assert_eq!(read_mxcsr() & 0x8040, 0x8040);
            assert_eq!(std::hint::black_box(f32::from_bits(1)) * 1.0, 0.0);
        }
        assert_eq!(read_mxcsr(), before);
        assert_ne!(std::hint::black_box(f32::from_bits(1)) * 1.0, 0.0);
    }

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
    fn log10_matches_libm() {
        let mut seed = 3u32;
        let mut g = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16_777_216.0
        };
        // Positive inputs spanning a wide range, as the DAF features do.
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
