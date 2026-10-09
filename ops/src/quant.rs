//! Per-vector symmetric f32 → i16 activation quantization of the W8A16 products.
//!
//! The scale is `max|x| / 16383`: 14-bit activations keep the int32 sum of up to
//! 1024 int8 × int16 products exact (127·16383·1024 < 2^31). The SIMD paths round
//! half away from zero by truncating and comparing the fraction, which reproduces
//! `f32::round`, so every tier emits the reference's bits. Input that is not
//! finite, or whose scale has no finite reciprocal, takes the reference.

use crate::simd_tier;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Quantizes `x` into `out[..x.len()]` and returns the scale.
pub fn quantize_i16(x: &[f32], out: &mut [i16]) -> f32 {
    quantize_on(simd_tier(), x, out)
}

fn quantize_on(tier: u8, x: &[f32], out: &mut [i16]) -> f32 {
    assert!(out.len() >= x.len(), "quantize_i16: output too short");
    #[cfg(target_arch = "x86_64")]
    if tier >= 1 {
        // Tests pass a tier the host supports, as `simd_tier` does.
        let amax = if tier >= 2 {
            // SAFETY: tier 2 and up means AVX.
            unsafe { abs_max_avx(x) }
        } else {
            // SAFETY: tier 1 means SSE4.1.
            unsafe { abs_max_sse(x) }
        };
        let scale = if amax > 0.0 { amax / 16383.0 } else { 1.0 };
        let inv = 1.0 / scale;
        if amax.is_finite() && inv.is_finite() {
            match tier {
                // SAFETY: tier 3 means AVX2; `out` holds `x.len()` values.
                3 => unsafe { quantize_avx2(x, out, inv) },
                // SAFETY: tier 2 means AVX; `out` holds `x.len()` values.
                2 => unsafe { quantize_avx(x, out, inv) },
                // SAFETY: tier 1 means SSE4.1; `out` holds `x.len()` values.
                _ => unsafe { quantize_sse(x, out, inv) },
            }
            return scale;
        }
    }
    let _ = tier;
    quantize_reference(x, out)
}

/// The scalar quantizer that defines every SIMD path's result.
fn quantize_reference(x: &[f32], out: &mut [i16]) -> f32 {
    assert!(out.len() >= x.len(), "quantize_i16: output too short");
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if amax > 0.0 { amax / 16383.0 } else { 1.0 };
    if scale == 0.0 {
        // `amax` is so small that the scale underflows: every value rounds to zero,
        // and `0 * inf` must not turn into NaN.
        out[..x.len()].fill(0);
        return 0.0;
    }
    let inv = 1.0 / scale;
    for (o, &v) in out.iter_mut().zip(x) {
        // A tiny scale can have an infinite reciprocal while the division stays finite.
        let q = if inv.is_finite() { v * inv } else { v / scale };
        *o = q.round().clamp(-16383.0, 16383.0) as i16;
    }
    scale
}

/// `max|x|`, or NaN if any value is not finite. Over finite values `max` is
/// order-independent, so this equals the reference fold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn abs_max_avx(x: &[f32]) -> f32 {
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut bad = _mm256_setzero_ps();
    let mut m = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`, so the load stays inside `x`.
        let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        bad = _mm256_or_ps(bad, _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
        m = _mm256_max_ps(m, _mm256_and_ps(v, absmask));
        i += 8;
    }
    if _mm256_movemask_ps(bad) != 0 {
        return f32::NAN;
    }
    let mut buf = [0f32; 8];
    // SAFETY: `buf` holds exactly 8 values.
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), m) };
    let mut r = buf.iter().copied().fold(0f32, f32::max);
    for &v in &x[i..] {
        if !v.is_finite() {
            return f32::NAN;
        }
        r = r.max(v.abs());
    }
    r
}

/// SSE4.1 counterpart of [`abs_max_avx`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
fn abs_max_sse(x: &[f32]) -> f32 {
    let absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let mut bad = _mm_setzero_ps();
    let mut m = _mm_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`, so the load stays inside `x`.
        let v = unsafe { _mm_loadu_ps(x.as_ptr().add(i)) };
        bad = _mm_or_ps(bad, _mm_cmpunord_ps(v, v));
        m = _mm_max_ps(m, _mm_and_ps(v, absmask));
        i += 4;
    }
    if _mm_movemask_ps(bad) != 0 {
        return f32::NAN;
    }
    let mut buf = [0f32; 4];
    // SAFETY: `buf` holds exactly 4 values.
    unsafe { _mm_storeu_ps(buf.as_mut_ptr(), m) };
    let mut r = buf.iter().copied().fold(0f32, f32::max);
    for &v in &x[i..] {
        if !v.is_finite() {
            return f32::NAN;
        }
        r = r.max(v.abs());
    }
    r
}

/// Rounds eight finite products half away from zero and clamps them to ±16383.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
fn round8(prod: __m256) -> __m256i {
    let trunc = _mm256_round_ps::<0x0B>(prod); // toward zero, no exception
    let frac = _mm256_sub_ps(prod, trunc);
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
    let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_and_ps(frac, abs), _mm256_set1_ps(0.5));
    // copysign(1, prod) where |frac| >= 0.5, else 0.
    let bump = _mm256_and_ps(
        _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(prod, sign)),
        ge,
    );
    let rounded = _mm256_min_ps(
        _mm256_max_ps(_mm256_add_ps(trunc, bump), _mm256_set1_ps(-16383.0)),
        _mm256_set1_ps(16383.0),
    );
    _mm256_cvtps_epi32(rounded) // integer-valued and in range: exact
}

/// SSE4.1 counterpart of [`round8`].
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
fn round4(prod: __m128) -> __m128i {
    let trunc = _mm_round_ps::<0x0B>(prod);
    let frac = _mm_sub_ps(prod, trunc);
    let abs = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let sign = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
    let ge = _mm_cmpge_ps(_mm_and_ps(frac, abs), _mm_set1_ps(0.5));
    let bump = _mm_and_ps(_mm_or_ps(_mm_set1_ps(1.0), _mm_and_ps(prod, sign)), ge);
    let rounded = _mm_min_ps(
        _mm_max_ps(_mm_add_ps(trunc, bump), _mm_set1_ps(-16383.0)),
        _mm_set1_ps(16383.0),
    );
    _mm_cvtps_epi32(rounded)
}

/// # Safety
/// The CPU must support AVX2, and `out` must be at least as long as `x`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(x: &[f32], out: &mut [i16], inv: f32) {
    // SAFETY: the loads read `x[i..i + 16]` with `i + 16 <= x.len()`, the stores
    // write the same range of `out`, which the caller makes at least as long, and
    // AVX2 implies the AVX the tail needs.
    unsafe {
        let scale = _mm256_set1_ps(inv);
        let mut i = 0;
        while i + 16 <= x.len() {
            let a = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), scale));
            let b = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), scale));
            // PACKSSDW works per 128-bit lane, giving a0..3 b0..3 a4..7 b4..7; swapping
            // the middle quadwords restores a0..7 b0..7. round8 already clamped, so the
            // pack never saturates.
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a, b));
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), packed);
            i += 16;
        }
        quantize_avx(&x[i..], &mut out[i..], inv);
    }
}

/// # Safety
/// The CPU must support AVX, and `out` must be at least as long as `x`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn quantize_avx(x: &[f32], out: &mut [i16], inv: f32) {
    // SAFETY: the load reads `x[i..i + 8]` with `i + 8 <= x.len()` and the store
    // writes the same range of `out`, which the caller makes at least as long.
    unsafe {
        let scale = _mm256_set1_ps(inv);
        let mut i = 0;
        while i + 8 <= x.len() {
            let r = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), scale));
            let packed =
                _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extractf128_si256::<1>(r));
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), packed);
            i += 8;
        }
        for (o, &v) in out[i..x.len()].iter_mut().zip(&x[i..]) {
            *o = (v * inv).round().clamp(-16383.0, 16383.0) as i16;
        }
    }
}

/// # Safety
/// The CPU must support SSE4.1, and `out` must be at least as long as `x`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn quantize_sse(x: &[f32], out: &mut [i16], inv: f32) {
    // SAFETY: the load reads `x[i..i + 4]` with `i + 4 <= x.len()` and the store
    // writes the same range of `out`, which the caller makes at least as long.
    unsafe {
        let scale = _mm_set1_ps(inv);
        let mut i = 0;
        while i + 4 <= x.len() {
            let r = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i)), scale));
            _mm_storel_epi64(out.as_mut_ptr().add(i).cast(), _mm_packs_epi32(r, r));
            i += 4;
        }
        for (o, &v) in out[i..x.len()].iter_mut().zip(&x[i..]) {
            *o = (v * inv).round().clamp(-16383.0, 16383.0) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_every_tier(x: &[f32]) {
        let mut want = vec![11i16; x.len() + 2];
        let ws = quantize_reference(x, &mut want[1..=x.len()]);
        for tier in 0..=simd_tier() {
            let mut got = vec![11i16; x.len() + 2];
            let gs = quantize_on(tier, x, &mut got[1..=x.len()]);
            assert_eq!(
                gs.to_bits(),
                ws.to_bits(),
                "tier {tier} scale, len {}",
                x.len()
            );
            assert_eq!(got, want, "tier {tier} values, len {}", x.len());
        }
    }

    #[test]
    fn every_tier_matches_the_reference_on_ties_extremes_and_tails() {
        let mut data = vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-39,
            -1e-39,
            16383.0,
            -16383.0,
        ];
        // Exact half-integer ties and their neighbours, both signs.
        for i in -16383..16383 {
            let x = i as f32 + 0.5;
            data.extend_from_slice(&[
                x,
                f32::from_bits(x.to_bits().wrapping_add(1)),
                f32::from_bits(x.to_bits().wrapping_sub(1)),
            ]);
        }
        for offset in 0..4 {
            for tail in 0..17 {
                check_every_tier(&data[offset..data.len() - tail]);
            }
        }
        for x in [
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, -1.0],
            vec![f32::NEG_INFINITY],
            vec![1e-40, -1e-40, 0.0],
            vec![],
            vec![-0.0; 512],
        ] {
            check_every_tier(&x);
        }
    }

    #[test]
    fn tiny_vectors_do_not_saturate_from_an_infinite_reciprocal() {
        let x = [1e-38f32, 0.5e-38, 0.0, -0.5e-38];
        let mut q = [0; 4];
        let scale = quantize_i16(&x, &mut q);
        assert!(scale > 0.0 && scale.is_finite());
        assert!(q[1] > 7500 && q[1] < 8500 && q[2] == 0 && q[3] == -q[1]);
        check_every_tier(&x);
    }

    #[test]
    fn an_underflowed_scale_quantizes_to_zero() {
        let x = [f32::from_bits(1); 4];
        let mut q = [1; 4];
        assert_eq!(quantize_i16(&x, &mut q), 0.0);
        assert_eq!(q, [0; 4]);
        check_every_tier(&x);
    }
}
