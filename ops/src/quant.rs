//! SIMD abs-max and round-half-away-from-zero quantization behind
//! `quantize_i16`. Non-finite input takes the scalar reference, which defines the
//! saturated result.
pub(crate) fn quantize(x: &[f32], out: &mut [i16]) -> f32 {
    assert!(out.len() >= x.len(), "quantize_i16: output too short");
    #[cfg(target_arch = "x86_64")]
    {
        let tier = crate::simd_tier();
        if tier >= 1 {
            // SAFETY: tier 1 guarantees SSE4.1, 2 AVX, 3 AVX2; `out` holds `x.len()`.
            let amax = unsafe {
                if tier >= 2 {
                    abs_max_avx(x)
                } else {
                    abs_max_sse(x)
                }
            };
            // `amax` is NaN when any input is not finite.
            let scale = if amax > 0.0 { amax / 16383.0 } else { 1.0 };
            let inv = 1.0 / scale;
            if amax.is_finite() && inv.is_finite() {
                // SAFETY: as above.
                unsafe {
                    match tier {
                        3 => quantize_avx2(x, out, inv),
                        2 => quantize_round_avx(x, out, inv),
                        _ => quantize_sse(x, out, inv),
                    }
                }
                return scale;
            }
        }
    }
    crate::quantize_i16_reference(x, out)
}
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn abs_max_avx(x: &[f32]) -> f32 {
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut bad = _mm256_setzero_ps();
    let mut m = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        bad = _mm256_or_ps(bad, _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
        m = _mm256_max_ps(m, _mm256_and_ps(v, absmask));
        i += 8;
    }
    if _mm256_movemask_ps(bad) != 0 {
        return f32::NAN;
    }
    let mut buf = [0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), m);
    let mut r = buf.iter().copied().fold(0f32, f32::max);
    while i < n {
        let v = *x.get_unchecked(i);
        if !v.is_finite() {
            return f32::NAN;
        }
        r = r.max(v.abs());
        i += 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn abs_max_sse(x: &[f32]) -> f32 {
    let absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let mut bad = _mm_setzero_ps();
    let mut m = _mm_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm_loadu_ps(x.as_ptr().add(i));
        bad = _mm_or_ps(bad, _mm_cmpunord_ps(v, v));
        m = _mm_max_ps(m, _mm_and_ps(v, absmask));
        i += 4;
    }
    if _mm_movemask_ps(bad) != 0 {
        return f32::NAN;
    }
    let mut buf = [0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), m);
    let mut r = buf.iter().copied().fold(0f32, f32::max);
    while i < n {
        let v = *x.get_unchecked(i);
        if !v.is_finite() {
            return f32::NAN;
        }
        r = r.max(v.abs());
        i += 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn quantize_round_avx(x: &[f32], out: &mut [i16], inv: f32) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn round8(prod: __m256) -> __m256i {
    let trunc = _mm256_round_ps::<0x0B>(prod);
    let frac = _mm256_sub_ps(prod, trunc);
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
    let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_and_ps(frac, abs), _mm256_set1_ps(0.5));
    let bump = _mm256_and_ps(
        _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(prod, sign)),
        ge,
    );
    let rounded = _mm256_min_ps(
        _mm256_max_ps(_mm256_add_ps(trunc, bump), _mm256_set1_ps(-16383.0)),
        _mm256_set1_ps(16383.0),
    );
    _mm256_cvtps_epi32(rounded)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn round4(prod: __m128) -> __m128i {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(x: &[f32], out: &mut [i16], inv: f32) {
    let mut i = 0;
    let scale = _mm256_set1_ps(inv);
    while i + 16 <= x.len() {
        let a = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), scale));
        let b = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), scale));
        // PACKSSDW is lane-local: [a0..3,b0..3,a4..7,b4..7]. Exchange middle
        // qwords so the stored i16 sequence is a0..7,b0..7. No saturation here:
        // round8 has already bounded every value to [-16383,16383].
        let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a, b));
        _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), packed);
        i += 16;
    }
    if i < x.len() {
        quantize_round_avx(&x[i..], &mut out[i..], inv);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn quantize_sse(x: &[f32], out: &mut [i16], inv: f32) {
    let scale = _mm_set1_ps(inv);
    let mut i = 0;
    while i + 8 <= x.len() {
        let a = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i)), scale));
        let b = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i + 4)), scale));
        _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_packs_epi32(a, b));
        i += 8;
    }
    if i < x.len() {
        quantize_round_sse(&x[i..], &mut out[i..], inv);
    }
}
