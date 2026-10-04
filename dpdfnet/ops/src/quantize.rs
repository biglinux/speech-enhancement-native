//! Vectorised f32 -> i16 activation quantization for AVX2 and SSE4.1, with the
//! scalar quantizer's scale, ties-away rounding and clipping.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
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
#[target_feature(enable = "avx2")]
pub(super) unsafe fn quantize_avx2(x: &[f32], out: &mut [i16], inv: f32) {
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
        super::quantize_round_avx(&x[i..], &mut out[i..], inv);
    }
}
#[cfg(target_arch = "x86_64")]
#[inline]
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
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn quantize_sse(x: &[f32], out: &mut [i16], inv: f32) {
    let scale = _mm_set1_ps(inv);
    let mut i = 0;
    while i + 8 <= x.len() {
        let a = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i)), scale));
        let b = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i + 4)), scale));
        _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_packs_epi32(a, b));
        i += 8;
    }
    if i < x.len() {
        super::quantize_round_sse(&x[i..], &mut out[i..], inv);
    }
}

// A preselected matvec function, NOT group fusion or a different GEMM schedule.

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn packed_rounding_matches_existing_for_tails_offsets_and_ties() {
        let mut v = Vec::new();
        for i in -16383..16383 {
            let x = i as f32 + 0.5;
            v.extend_from_slice(&[
                x,
                f32::from_bits(x.to_bits().wrapping_add(1)),
                f32::from_bits(x.to_bits().wrapping_sub(1)),
            ]);
        }
        v.extend_from_slice(&[0.0, -0.0, 1e-39, -1e-39, 20000.0, -20000.0]);
        for tail in 0..17 {
            let x = &v[1..v.len() - tail];
            let mut base = vec![99i16; x.len() + 2];
            let mut y = base.clone();
            if std::is_x86_feature_detected!("sse4.1") {
                unsafe {
                    crate::quantize_round_sse(x, &mut base[1..1 + x.len()], 1.0);
                    super::quantize_sse(x, &mut y[1..1 + x.len()], 1.0);
                }
                assert_eq!(y, base);
            }
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::quantize_round_avx(x, &mut base[1..1 + x.len()], 1.0);
                    super::quantize_avx2(x, &mut y[1..1 + x.len()], 1.0);
                }
                assert_eq!(y, base);
            }
        }
    }
}
