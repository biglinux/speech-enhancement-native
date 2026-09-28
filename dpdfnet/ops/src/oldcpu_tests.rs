//! Exact exponent construction versus deployed arithmetic.
use std::arch::x86_64::*;
#[target_feature(enable = "avx")]
unsafe fn exp8_reference(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
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
    let ri = _mm256_cvtps_epi32(r);
    let bias = _mm_set1_epi32(127);
    let lo = _mm_slli_epi32::<23>(_mm_add_epi32(_mm256_castsi256_si128(ri), bias));
    let hi = _mm_slli_epi32::<23>(_mm_add_epi32(_mm256_extractf128_si256::<1>(ri), bias));
    let pow2 = _mm256_set_m128(_mm_castsi128_ps(hi), _mm_castsi128_ps(lo));
    _mm256_mul_ps(p, pow2)
}

#[test]
fn exp_bits_keep_the_original_arithmetic() {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    // Includes all reachable exponent classes and narrow neighborhoods at zero.
    for i in 0..32768 {
        let mut x = [0.0f32; 8];
        for j in 0..8 {
            x[j] = ((i * 8 + j) as f32 - 131072.0) / 1024.0;
        }
        let mut a = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        unsafe {
            _mm256_storeu_ps(a.as_mut_ptr(), super::exp8(_mm256_loadu_ps(x.as_ptr())));
            _mm256_storeu_ps(b.as_mut_ptr(), exp8_reference(_mm256_loadu_ps(x.as_ptr())));
        }
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }
}
