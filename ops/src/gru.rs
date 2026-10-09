//! GRU cells: the gate update shared by every engine, the W8A16 cell of the
//! denoisers, and the eight-lane f32 cell of the AEC delay estimator.
//!
//! The scalar paths use exact libm math. The SIMD gates use range-reduced
//! polynomials (`exp8`/`exp4`, about 1e-6 relative error); the 4-lane SSE4.1 forms
//! do the same arithmetic per lane as the 8-lane AVX forms, so both tiers give the
//! same bits.

use crate::{packed, quantize_i16, sigmoid, simd_tier};

/// Updates `h` from already computed projections, with PyTorch's reset-after
/// equation (`linear_before_reset=1`). Gate order is z, r, n in `wx` and `rh`; `b`
/// holds the input biases z, r, n followed by the recurrent ones.
pub fn gru_update(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32]) {
    gru_update_on(simd_tier(), h, wx, rh, b);
}

fn gru_update_on(tier: u8, h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32]) {
    let hs = h.len();
    assert!(wx.len() >= 3 * hs && rh.len() >= 3 * hs && b.len() >= 6 * hs);
    #[cfg(target_arch = "x86_64")]
    {
        if tier >= 2 && hs.is_multiple_of(8) {
            // SAFETY: tier 2 guarantees AVX; the assert bounds every 8-lane load.
            unsafe { gate8(h, wx, rh, b, hs) };
            return;
        }
        if tier >= 1 && hs.is_multiple_of(4) {
            // SAFETY: tier 1 guarantees SSE4.1; the assert bounds every 4-lane load.
            unsafe { gate4(h, wx, rh, b, hs) };
            return;
        }
    }
    let _ = tier;
    for i in 0..hs {
        let z = sigmoid(wx[i] + rh[i] + b[i] + b[3 * hs + i]);
        let r = sigmoid(wx[hs + i] + rh[hs + i] + b[hs + i] + b[4 * hs + i]);
        let n = (wx[2 * hs + i] + b[2 * hs + i] + r * (rh[2 * hs + i] + b[5 * hs + i])).tanh();
        h[i] = (1.0 - z) * n + z * h[i];
    }
}

/// int8-weight GRU cell over pair-packed matrices (W8A16): activations are
/// quantized to int16 per product, gates and state stay f32. ONNX gate order
/// `[z,r,h]`, `linear_before_reset=1`. `scratch` holds at least `6*hs` values and
/// `xq16` at least `max(input, hs)`.
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
    let h = &mut h[..hs];
    let (wx, rest) = scratch.split_at_mut(gates);
    let rh = &mut rest[..gates];
    let sx = quantize_i16(&x[..input], &mut xq16[..input]);
    packed::matvec(wx, wq, ws, &xq16[..input], sx, gates, input);
    let sh = quantize_i16(h, &mut xq16[..hs]);
    packed::matvec(rh, rq, rs, &xq16[..hs], sh, gates, hs);
    gru_update(h, wx, rh, b);
}

/// Eight GRU cells sharing one weight set, one per SIMD lane, in PyTorch gate order
/// `[r,z,n]` with separate input and hidden biases. Layout is feature-major:
/// `x[i*8+lane]` is input feature `i` of a lane and `h[j*8+lane]` its hidden state,
/// updated in place. `wih`/`whh` are `[3*nh][nin]`/`[3*nh][nh]` row-major, `bih` and
/// `bhh` hold `3*nh` values, and `nh <= 16`.
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
    let gates = 3 * nh;
    assert!(x.len() >= nin.checked_mul(8).expect("gru8 input overflow"));
    assert!(h.len() >= nh * 8);
    assert!(wih.len() >= gates.checked_mul(nin).expect("gru8 weights overflow"));
    assert!(whh.len() >= gates * nh);
    assert!(bih.len() >= gates && bhh.len() >= gates);
    #[cfg(target_arch = "x86_64")]
    {
        match simd_tier() {
            // SAFETY: tier 3 guarantees AVX2+FMA; the asserts above give the lengths.
            3 => return unsafe { gru8_avx2(x, nin, h, nh, wih, whh, bih, bhh) },
            // SAFETY: tier 2 guarantees AVX; the asserts above give the lengths.
            2 => return unsafe { gru8_avx(x, nin, h, nh, wih, whh, bih, bhh) },
            _ => {}
        }
    }
    for lane in 0..8 {
        gru_lane(x, nin, h, nh, wih, whh, bih, bhh, lane);
    }
}

/// One lane of [`gru8`] in scalar libm math: the fallback and the test reference.
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
        let r = sigmoid(gi[j] + gh[j]);
        let z = sigmoid(gi[nh + j] + gh[nh + j]);
        let n = (gi[2 * nh + j] + r * gh[2 * nh + j]).tanh();
        h[j * 8 + lane] = (1.0 - z) * n + z * h[j * 8 + lane];
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 8-wide `exp`: `2^r · poly(f)` with a degree-5 Taylor polynomial on
/// `|f| <= ln2/2`. The input is clamped so `r` stays in [-126, 127] and `2^r` is a
/// normal number, built in the exponent field with 128-bit integer shifts so AVX
/// suffices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn exp8(x: __m256) -> __m256 {
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
    let ri = _mm256_cvtps_epi32(r);
    let bias = _mm_set1_epi32(127);
    let lo = _mm_slli_epi32::<23>(_mm_add_epi32(_mm256_castsi256_si128(ri), bias));
    let hi = _mm_slli_epi32::<23>(_mm_add_epi32(_mm256_extractf128_si256::<1>(ri), bias));
    _mm256_mul_ps(
        p,
        _mm256_set_m128(_mm_castsi128_ps(hi), _mm_castsi128_ps(lo)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn sigmoid8(x: __m256) -> __m256 {
    let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn tanh8(x: __m256) -> __m256 {
    let s = sigmoid8(_mm256_mul_ps(x, _mm256_set1_ps(2.0)));
    _mm256_sub_ps(_mm256_mul_ps(s, _mm256_set1_ps(2.0)), _mm256_set1_ps(1.0))
}

/// 4-wide [`exp8`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
fn exp4(x: __m128) -> __m128 {
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
fn sigmoid4(x: __m128) -> __m128 {
    let e = exp4(_mm_sub_ps(_mm_setzero_ps(), x));
    _mm_div_ps(_mm_set1_ps(1.0), _mm_add_ps(_mm_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
fn tanh4(x: __m128) -> __m128 {
    let s = sigmoid4(_mm_mul_ps(x, _mm_set1_ps(2.0)));
    _mm_sub_ps(_mm_mul_ps(s, _mm_set1_ps(2.0)), _mm_set1_ps(1.0))
}

/// [`gru_update`] 8 lanes at a time.
///
/// # Safety
/// The CPU must support AVX, `hs` must be a multiple of 8, `h` must hold `hs`
/// values, `wx` and `rh` `3*hs` and `b` `6*hs`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gate8(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    // SAFETY: each load and store covers lanes `i..i + 8` of a gate block, with
    // `i + 8 <= hs`, inside the lengths the caller guarantees.
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let mut i = 0;
        while i < hs {
            let ld = |s: &[f32], o: usize| _mm256_loadu_ps(s.as_ptr().add(o));
            let z = sigmoid8(_mm256_add_ps(
                _mm256_add_ps(ld(wx, i), ld(rh, i)),
                _mm256_add_ps(ld(b, i), ld(b, 3 * hs + i)),
            ));
            let r = sigmoid8(_mm256_add_ps(
                _mm256_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
                _mm256_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
            ));
            let inner = _mm256_mul_ps(r, _mm256_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
            let n = tanh8(_mm256_add_ps(
                _mm256_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
                inner,
            ));
            let hnew = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(one, z), n),
                _mm256_mul_ps(z, ld(h, i)),
            );
            _mm256_storeu_ps(h.as_mut_ptr().add(i), hnew);
            i += 8;
        }
    }
}

/// [`gate8`] at 128-bit width.
///
/// # Safety
/// The CPU must support SSE4.1, `hs` must be a multiple of 4, `h` must hold `hs`
/// values, `wx` and `rh` `3*hs` and `b` `6*hs`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn gate4(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    // SAFETY: each load and store covers lanes `i..i + 4` of a gate block, with
    // `i + 4 <= hs`, inside the lengths the caller guarantees.
    unsafe {
        let one = _mm_set1_ps(1.0);
        let mut i = 0;
        while i < hs {
            let ld = |s: &[f32], o: usize| _mm_loadu_ps(s.as_ptr().add(o));
            let z = sigmoid4(_mm_add_ps(
                _mm_add_ps(ld(wx, i), ld(rh, i)),
                _mm_add_ps(ld(b, i), ld(b, 3 * hs + i)),
            ));
            let r = sigmoid4(_mm_add_ps(
                _mm_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
                _mm_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
            ));
            let inner = _mm_mul_ps(r, _mm_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
            let n = tanh4(_mm_add_ps(
                _mm_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
                inner,
            ));
            let hnew = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(one, z), n), _mm_mul_ps(z, ld(h, i)));
            _mm_storeu_ps(h.as_mut_ptr().add(i), hnew);
            i += 4;
        }
    }
}

/// [`gru8`] body; `$fma` selects FMA or a multiply and add for the projections.
#[cfg(target_arch = "x86_64")]
macro_rules! gru8_body {
    ($x:expr, $nin:expr, $h:expr, $nh:expr, $wih:expr, $whh:expr, $bih:expr, $bhh:expr, $fma:tt) => {{
        let (x, h, wih, whh, bih, bhh) = ($x, $h, $wih, $whh, $bih, $bhh);
        let (nin, nh) = ($nin, $nh);
        let mut gi = [_mm256_setzero_ps(); 48];
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

/// # Safety
/// The CPU must support AVX2 and FMA; `nh <= 16`, `x` must hold `8*nin` values, `h`
/// `8*nh`, `wih` `3*nh*nin`, `whh` `3*nh*nh`, `bih` and `bhh` `3*nh`.
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
    // SAFETY: the loads read lane groups `i*8..i*8 + 8` of `x` (`i < nin`) and
    // `h` (`i < nh`), which the caller's lengths cover; `nh <= 16` keeps
    // `3*nh` within the 48-entry scratch.
    unsafe { gru8_body!(x, nin, h, nh, wih, whh, bih, bhh, true) }
}

/// # Safety
/// The CPU must support AVX; `nh <= 16`, `x` must hold `8*nin` values, `h`
/// `8*nh`, `wih` `3*nh*nin`, `whh` `3*nh*nh`, `bih` and `bhh` `3*nh`.
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
    // SAFETY: the loads read lane groups `i*8..i*8 + 8` of `x` (`i < nin`) and
    // `h` (`i < nh`), which the caller's lengths cover; `nh <= 16` keeps
    // `3*nh` within the 48-entry scratch.
    unsafe { gru8_body!(x, nin, h, nh, wih, whh, bih, bhh, false) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic values in [-0.5, 0.5), which keep the gates in their active range.
    fn fill(v: &mut [f32], seed: &mut u32) {
        for x in v {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *x = (*seed >> 8) as f32 / 16_777_216.0 - 0.5;
        }
    }

    fn filled(n: usize, seed: &mut u32) -> Vec<f32> {
        let mut v = vec![0.0; n];
        fill(&mut v, seed);
        v
    }

    // The SSE4.1 and AVX gates must agree bit for bit, and both stay within the
    // polynomial error of the libm path.
    #[test]
    fn every_tier_of_the_gate_update_agrees() {
        let mut seed = 0xc0ffee;
        for hs in [4usize, 8, 64, 256] {
            for _ in 0..50 {
                let wx: Vec<f32> = filled(3 * hs, &mut seed).iter().map(|v| v * 8.0).collect();
                let rh: Vec<f32> = filled(3 * hs, &mut seed).iter().map(|v| v * 8.0).collect();
                let b: Vec<f32> = filled(6 * hs, &mut seed).iter().map(|v| v * 8.0).collect();
                let h0 = filled(hs, &mut seed);
                let mut libm = h0.clone();
                gru_update_on(0, &mut libm, &wx, &rh, &b);
                let mut simd: Option<Vec<f32>> = None;
                for tier in 1..=simd_tier() {
                    let mut h = h0.clone();
                    gru_update_on(tier, &mut h, &wx, &rh, &b);
                    for (a, r) in h.iter().zip(&libm) {
                        assert!((a - r).abs() < 1e-5, "tier {tier} hs={hs}: {a} vs {r}");
                    }
                    if hs % 8 == 0 {
                        // gate4 (tier 1) and gate8 (tiers 2 and 3) give the same bits.
                        match &simd {
                            Some(s) => assert_eq!(&h, s, "tier {tier} hs={hs}"),
                            None => simd = Some(h),
                        }
                    }
                }
            }
        }
    }

    // The batched kernel must match eight independent scalar lanes; a transpose or
    // index bug shows up as an O(1) error, far above the polynomial error.
    fn check_gru8(nin: usize, nh: usize) {
        let mut seed = 7u32;
        let wih = filled(3 * nh * nin, &mut seed);
        let whh = filled(3 * nh * nh, &mut seed);
        let bih = filled(3 * nh, &mut seed);
        let bhh = filled(3 * nh, &mut seed);
        let x = filled(nin * 8, &mut seed);
        let mut h = filled(nh * 8, &mut seed);
        let mut h_ref = h.clone();
        for lane in 0..8 {
            gru_lane(&x, nin, &mut h_ref, nh, &wih, &whh, &bih, &bhh, lane);
        }
        gru8(&x, nin, &mut h, nh, &wih, &whh, &bih, &bhh);
        for (a, b) in h.iter().zip(&h_ref) {
            assert!((a - b).abs() < 1e-3, "gru8 nin={nin} nh={nh}: {a} vs {b}");
        }
    }

    #[test]
    fn gru8_matches_scalar_lanes() {
        check_gru8(18, 16); // DAF per-bin controller
        check_gru8(10, 8); // DAF per-partition controller
    }
}
