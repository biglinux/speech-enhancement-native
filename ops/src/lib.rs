//! Numeric kernels shared by the DeepFilterNet3, GTCRN-AEC and Silero engines:
//! f32 and W8A16 matrix products, GRU cells, convolutions on the frequency axis,
//! and the 48/16 kHz resamplers.
//!
//! The scalar fallbacks use exact libm math. The AVX GRU gates use range-reduced
//! polynomials (`exp8`/`tanh8`, about 1e-6 relative error).

#![allow(clippy::needless_range_loop)] // numeric kernels index by design
#![allow(clippy::too_many_arguments)] // GRU/GEMV kernels take weight+bias+dims explicitly

#[cfg(test)]
mod kernel_tests;
#[cfg(target_arch = "x86_64")]
mod matvec_port;
pub mod pack_format;
mod packed;
mod quant;
pub mod resample;

/// Sets flush-to-zero and denormals-are-zero on the calling thread and restores
/// the previous MXCSR when dropped, so the host and the other plugins on its audio
/// thread keep their floating-point environment. Recurrent state decays toward
/// denormals in silence, and denormal arithmetic on x86 is slow enough to push a
/// hop into an xrun.
#[must_use]
pub struct DenormalGuard {
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}

impl DenormalGuard {
    #[inline]
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let saved = read_mxcsr();
            // FTZ = bit 15, DAZ = bit 6.
            write_mxcsr(saved | 0x8040);
            Self { saved }
        }
        #[cfg(not(target_arch = "x86_64"))]
        Self {}
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
    // SAFETY: the value comes from stmxcsr with only the FTZ/DAZ bits changed, so
    // no reserved bit is set and ldmxcsr cannot fault.
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
    if cfg!(feature = "force-sse41") {
        return if is_x86_feature_detected!("sse4.1") {
            1
        } else {
            0
        };
    }
    if cfg!(feature = "force-avx1") {
        return if is_x86_feature_detected!("avx") {
            2
        } else if is_x86_feature_detected!("sse4.1") {
            1
        } else {
            0
        };
    }
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(u8::MAX);
    let c = CACHE.load(Ordering::Relaxed);
    if c != u8::MAX {
        return c;
    }
    let t = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
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

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_tier() -> u8 {
    0
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
/// AEC convolutions, which do not need cross-tier bit-identity.
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
/// dispatch). `$fma` picks FMA vs mul+add. Used by the AEC; the denoisers use
/// [`pointwise_conv2d`], which is bit-identical across tiers.
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
fn quantize_i16(x: &[f32], out: &mut [i16]) -> f32 {
    quant::quantize(x, out)
}
fn quantize_i16_reference(x: &[f32], out: &mut [i16]) -> f32 {
    assert!(out.len() >= x.len(), "quantize_i16: output too short");
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if amax > 0.0 { amax / 16383.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v * inv).round().clamp(-16383.0, 16383.0) as i16;
    }
    scale
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
/// needing exact log10 must use `f32::log10`.
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

/// `y[N] = A[M,N]^T * x[M]`, A row-major (accumulate columns / SAXPY).
#[inline]
fn matvec_t(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m);
    assert!(a.len() >= m.checked_mul(n).expect("matvec_t dimensions overflow"));
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            3 => {
                unsafe { matvec_port::matvec_avx2(y, a, x, m, n) };
                return;
            }
            2 => {
                unsafe { matvec_port::matvec_avx(y, a, x, m, n) };
                return;
            }
            1 => {
                unsafe { matvec_port::matvec_sse(y, a, x, m, n) };
                return;
            }
            _ => {}
        }
    }
    y[..n].fill(0.0);
    for i in 0..m {
        let row = &a[i * n..i * n + n];
        let xi = x[i];
        for j in 0..n {
            y[j] += xi * row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[cfg(test)] // the reference matvec_port is checked against
unsafe fn matvec_t_avx(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    use std::arch::x86_64::*;
    for v in y[..n].iter_mut() {
        *v = 0.0;
    }
    let n8 = n & !7;
    let ap = a.as_ptr();
    let yp = y.as_mut_ptr();
    for i in 0..m {
        let xi = _mm256_set1_ps(*x.get_unchecked(i));
        let row = ap.add(i * n);
        let mut j = 0;
        while j < n8 {
            let acc = _mm256_loadu_ps(yp.add(j));
            let r = _mm256_loadu_ps(row.add(j));
            _mm256_storeu_ps(yp.add(j), _mm256_add_ps(acc, _mm256_mul_ps(r, xi)));
            j += 8;
        }
        let xis = *x.get_unchecked(i);
        while j < n {
            *yp.add(j) += xis * *row.add(j);
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[cfg(test)] // the reference matvec_port is checked against
unsafe fn matvec_t_avx2(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    use std::arch::x86_64::*;
    for v in y[..n].iter_mut() {
        *v = 0.0;
    }
    let n8 = n & !7;
    let ap = a.as_ptr();
    let yp = y.as_mut_ptr();
    for i in 0..m {
        let xi = _mm256_set1_ps(*x.get_unchecked(i));
        let row = ap.add(i * n);
        let mut j = 0;
        while j < n8 {
            let acc = _mm256_loadu_ps(yp.add(j));
            let r = _mm256_loadu_ps(row.add(j));
            _mm256_storeu_ps(yp.add(j), _mm256_fmadd_ps(r, xi, acc));
            j += 8;
        }
        let xis = *x.get_unchecked(i);
        while j < n {
            *yp.add(j) += xis * *row.add(j);
            j += 1;
        }
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

/// int8-weight GRU cell over pair-packed matrices (W8A16): activations are
/// quantized to int16 per product, gates and state stay f32. ONNX gate order
/// `[z,r,h]`, `linear_before_reset=1`. `xq16` is scratch, len >= max(input, hs).
pub fn gru_cell_packed(
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
    packed::matvec(wx, wq, ws, &xq16[..input], sx, 3 * hs, input);
    let sh = quantize_i16(h, &mut xq16[..hs]);
    packed::matvec(rh, rq, rs, &xq16[..hs], sh, 3 * hs, hs);

    finish_gru(h, wx, rh, b, hs);
}

fn finish_gru(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 && hs.is_multiple_of(8) {
        // SAFETY: tier >= 2 means AVX is present; hs is a multiple of 8. On tier 1
        // (SSE4.1, no AVX) the scalar gate below runs instead.
        unsafe { gate8(h, wx, rh, b, hs) };
        return;
    }

    let (wbz, wbr, wbh) = (&b[0..hs], &b[hs..2 * hs], &b[2 * hs..3 * hs]);
    let (rbz, rbr, rbh) = (&b[3 * hs..4 * hs], &b[4 * hs..5 * hs], &b[5 * hs..6 * hs]);

    for i in 0..hs {
        let z = sigmoid(wx[i] + rh[i] + wbz[i] + rbz[i]);
        let rr = sigmoid(wx[hs + i] + rh[hs + i] + wbr[i] + rbr[i]);
        let hh = (wx[2 * hs + i] + wbh[i] + rr * (rh[2 * hs + i] + rbh[i])).tanh();
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
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        assert!(out.len() >= c_out.checked_mul(width).expect("pointwise output overflow"));
        assert!(input.len() >= c_in.checked_mul(width).expect("pointwise input overflow"));
        assert!(weight.len() >= c_out.checked_mul(c_in).expect("pointwise weights overflow"));
        assert!(bias.len() >= c_out);
        // SAFETY: AVX was detected at run time; the asserts bound every access.
        unsafe { pointwise_conv2d_avx(out, input, weight, bias, c_in, c_out, width) };
        return;
    }
    pointwise_conv2d_scalar(out, input, weight, bias, c_in, c_out, width);
}

/// AVX body of [`pointwise_conv2d`], the largest single cost in DFN3. The scalar
/// loop rewrites `out` once per input channel; here up to 32 output columns stay
/// in registers across every input channel.
/// Each lane still computes `bias + w0*x0 + w1*x1 + ...` left to right with a
/// separate multiply and add (no FMA), so the result is bit-identical to the
/// scalar loop on every tier.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pointwise_conv2d_avx(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    use std::arch::x86_64::*;
    let x = input.as_ptr();
    for co in 0..c_out {
        let wr = weight.as_ptr().add(co * c_in);
        let o = out.as_mut_ptr().add(co * width);
        let b = _mm256_set1_ps(bias[co]);
        let mut w0 = 0;
        while w0 + 32 <= width {
            let (mut a0, mut a1, mut a2, mut a3) = (b, b, b, b);
            for ci in 0..c_in {
                let k = _mm256_set1_ps(*wr.add(ci));
                let row = x.add(ci * width + w0);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(k, _mm256_loadu_ps(row)));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(k, _mm256_loadu_ps(row.add(8))));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(k, _mm256_loadu_ps(row.add(16))));
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(k, _mm256_loadu_ps(row.add(24))));
            }
            _mm256_storeu_ps(o.add(w0), a0);
            _mm256_storeu_ps(o.add(w0 + 8), a1);
            _mm256_storeu_ps(o.add(w0 + 16), a2);
            _mm256_storeu_ps(o.add(w0 + 24), a3);
            w0 += 32;
        }
        while w0 + 8 <= width {
            let mut a = b;
            for ci in 0..c_in {
                let k = _mm256_set1_ps(*wr.add(ci));
                a = _mm256_add_ps(a, _mm256_mul_ps(k, _mm256_loadu_ps(x.add(ci * width + w0))));
            }
            _mm256_storeu_ps(o.add(w0), a);
            w0 += 8;
        }
        for w in w0..width {
            let mut s = bias[co];
            for ci in 0..c_in {
                s += *wr.add(ci) * *x.add(ci * width + w);
            }
            *o.add(w) = s;
        }
    }
}

/// Reference and non-AVX path of [`pointwise_conv2d`].
fn pointwise_conv2d_scalar(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    // co outer, ci inner: each output row stays hot in L1 while the small input
    // tile is re-read. The AVX body keeps exactly this summation order.
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

#[cfg(all(test, target_arch = "x86_64"))]
mod pointwise_avx_tests {
    use super::*;

    #[test]
    fn avx_pointwise_is_bit_identical_to_the_scalar_loop() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let mut seed = 0x1234_5678_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        for &(c_in, c_out, width) in &[
            (64, 64, 96),
            (64, 64, 48),
            (10, 10, 96),
            (64, 64, 37),
            (3, 5, 7),
            (64, 64, 1),
        ] {
            let input: Vec<f32> = (0..c_in * width).map(|_| next()).collect();
            let weight: Vec<f32> = (0..c_in * c_out).map(|_| next()).collect();
            let bias: Vec<f32> = (0..c_out).map(|_| next()).collect();
            let mut want = vec![0.0; c_out * width];
            let mut got = vec![0.0; c_out * width];
            pointwise_conv2d_scalar(&mut want, &input, &weight, &bias, c_in, c_out, width);
            unsafe { pointwise_conv2d_avx(&mut got, &input, &weight, &bias, c_in, c_out, width) };
            let same = want
                .iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "shape {c_in}x{c_out}x{width}");
        }
    }
}
