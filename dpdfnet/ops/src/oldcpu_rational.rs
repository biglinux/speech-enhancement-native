//! Optional post-training numerical backend for old x86: rational tanh 9/8.
//! Copyright 2024 Google LLC: coefficients/evaluation based on XNNPACK's
//! src/f32-vtanh/rational-9-8.c.in (blob c2f6621fdfcade6c11a480d13fa3c8b51a44be40).
//! BSD-3-Clause, reproduced in licenses/XNNPACK-BSD.txt.
//! Rust adaptation: no FMA, reciprocal estimate, Newton step, LUT or AVX2.
//! Keeps DIVPS/VDIVPS. This is a PRECISION CHANGE, not a bit-exact optimization.
//! Nonlinear functions change; the original bias/reset-after/state ordering does not.
use std::arch::x86_64::*;

pub(super) fn try_update(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32]) -> bool {
    let hs = h.len();
    if hs == 0 || hs % 8 != 0 {
        return false;
    }
    match super::simd_tier() {
        2 => {
            unsafe { gate_avx(h, wx, rh, b, hs) };
            true
        }
        1 => {
            unsafe { gate_sse(h, wx, rh, b, hs) };
            true
        }
        _ => false, // unchanged modern/default scalar path
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn tanh_sse(x: __m128) -> __m128 {
    let x = _mm_max_ps(
        _mm_min_ps(x, _mm_set1_ps(7.8522667885)),
        _mm_set1_ps(-7.8522667885),
    );
    let u = _mm_mul_ps(x, x);
    let mut p = _mm_set1_ps(1.4248920266e-8);
    p = _mm_add_ps(_mm_mul_ps(u, p), _mm_set1_ps(2.1235626264e-5));
    p = _mm_add_ps(_mm_mul_ps(u, p), _mm_set1_ps(3.5330520477e-3));
    p = _mm_add_ps(_mm_mul_ps(u, p), _mm_set1_ps(1.3412411511e-1));
    p = _mm_add_ps(_mm_mul_ps(u, p), _mm_set1_ps(1.0));
    p = _mm_mul_ps(x, p);
    let mut q = _mm_set1_ps(8.1365948290e-7);
    q = _mm_add_ps(_mm_mul_ps(u, q), _mm_set1_ps(3.3472978976e-4));
    q = _mm_add_ps(_mm_mul_ps(u, q), _mm_set1_ps(2.6018999517e-2));
    q = _mm_add_ps(_mm_mul_ps(u, q), _mm_set1_ps(4.6745735407e-1));
    q = _mm_add_ps(_mm_mul_ps(u, q), _mm_set1_ps(1.0));
    _mm_div_ps(p, q)
}
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn sigmoid_sse(x: __m128) -> __m128 {
    _mm_mul_ps(
        _mm_add_ps(tanh_sse(_mm_mul_ps(x, _mm_set1_ps(0.5))), _mm_set1_ps(1.0)),
        _mm_set1_ps(0.5),
    )
}
#[target_feature(enable = "sse4.1")]
unsafe fn gate_sse(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    let one = _mm_set1_ps(1.0);
    for i in (0..hs).step_by(4) {
        let ld = |s: &[f32], o: usize| _mm_loadu_ps(s.as_ptr().add(o));
        let z = sigmoid_sse(_mm_add_ps(
            _mm_add_ps(ld(wx, i), ld(rh, i)),
            _mm_add_ps(ld(b, i), ld(b, 3 * hs + i)),
        ));
        let r = sigmoid_sse(_mm_add_ps(
            _mm_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
            _mm_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
        ));
        let inner = _mm_mul_ps(r, _mm_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
        let n = tanh_sse(_mm_add_ps(
            _mm_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
            inner,
        ));
        let y = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(one, z), n), _mm_mul_ps(z, ld(h, i)));
        _mm_storeu_ps(h.as_mut_ptr().add(i), y);
    }
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn tanh_avx(x: __m256) -> __m256 {
    let x = _mm256_max_ps(
        _mm256_min_ps(x, _mm256_set1_ps(7.8522667885)),
        _mm256_set1_ps(-7.8522667885),
    );
    let u = _mm256_mul_ps(x, x);
    let mut p = _mm256_set1_ps(1.4248920266e-8);
    p = _mm256_add_ps(_mm256_mul_ps(u, p), _mm256_set1_ps(2.1235626264e-5));
    p = _mm256_add_ps(_mm256_mul_ps(u, p), _mm256_set1_ps(3.5330520477e-3));
    p = _mm256_add_ps(_mm256_mul_ps(u, p), _mm256_set1_ps(1.3412411511e-1));
    p = _mm256_add_ps(_mm256_mul_ps(u, p), _mm256_set1_ps(1.0));
    p = _mm256_mul_ps(x, p);
    let mut q = _mm256_set1_ps(8.1365948290e-7);
    q = _mm256_add_ps(_mm256_mul_ps(u, q), _mm256_set1_ps(3.3472978976e-4));
    q = _mm256_add_ps(_mm256_mul_ps(u, q), _mm256_set1_ps(2.6018999517e-2));
    q = _mm256_add_ps(_mm256_mul_ps(u, q), _mm256_set1_ps(4.6745735407e-1));
    q = _mm256_add_ps(_mm256_mul_ps(u, q), _mm256_set1_ps(1.0));
    _mm256_div_ps(p, q)
}
#[inline]
#[target_feature(enable = "avx")]
unsafe fn sigmoid_avx(x: __m256) -> __m256 {
    _mm256_mul_ps(
        _mm256_add_ps(
            tanh_avx(_mm256_mul_ps(x, _mm256_set1_ps(0.5))),
            _mm256_set1_ps(1.0),
        ),
        _mm256_set1_ps(0.5),
    )
}
#[target_feature(enable = "avx")]
unsafe fn gate_avx(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    let one = _mm256_set1_ps(1.0);
    for i in (0..hs).step_by(8) {
        let ld = |s: &[f32], o: usize| _mm256_loadu_ps(s.as_ptr().add(o));
        let z = sigmoid_avx(_mm256_add_ps(
            _mm256_add_ps(ld(wx, i), ld(rh, i)),
            _mm256_add_ps(ld(b, i), ld(b, 3 * hs + i)),
        ));
        let r = sigmoid_avx(_mm256_add_ps(
            _mm256_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
            _mm256_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
        ));
        let inner = _mm256_mul_ps(r, _mm256_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
        let n = tanh_avx(_mm256_add_ps(
            _mm256_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
            inner,
        ));
        let y = _mm256_add_ps(
            _mm256_mul_ps(_mm256_sub_ps(one, z), n),
            _mm256_mul_ps(z, ld(h, i)),
        );
        _mm256_storeu_ps(h.as_mut_ptr().add(i), y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rational_local_error_and_widths() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for chunk in 0..8192 {
            let mut input = [0.0f32; 8];
            for (j, x) in input.iter_mut().enumerate() {
                *x = (chunk * 8 + j) as f32 / 2048.0 - 16.0;
            }
            let mut t = [0.0f32; 8];
            let mut s = [0.0f32; 8];
            unsafe {
                _mm_storeu_ps(t.as_mut_ptr(), tanh_sse(_mm_loadu_ps(input.as_ptr())));
                _mm_storeu_ps(
                    t.as_mut_ptr().add(4),
                    tanh_sse(_mm_loadu_ps(input.as_ptr().add(4))),
                );
                _mm_storeu_ps(s.as_mut_ptr(), sigmoid_sse(_mm_loadu_ps(input.as_ptr())));
                _mm_storeu_ps(
                    s.as_mut_ptr().add(4),
                    sigmoid_sse(_mm_loadu_ps(input.as_ptr().add(4))),
                );
            }
            for j in 0..8 {
                assert!((t[j] as f64 - (input[j] as f64).tanh()).abs() < 1.0e-6);
                assert!((s[j] as f64 - 1.0 / (1.0 + (-(input[j] as f64)).exp())).abs() < 1.0e-6);
            }
            if std::is_x86_feature_detected!("avx") {
                let mut at = [0.0f32; 8];
                let mut az = [0.0f32; 8];
                unsafe {
                    _mm256_storeu_ps(at.as_mut_ptr(), tanh_avx(_mm256_loadu_ps(input.as_ptr())));
                    _mm256_storeu_ps(
                        az.as_mut_ptr(),
                        sigmoid_avx(_mm256_loadu_ps(input.as_ptr())),
                    );
                }
                assert_eq!(t.map(f32::to_bits), at.map(f32::to_bits));
                assert_eq!(s.map(f32::to_bits), az.map(f32::to_bits));
            }
        }
    }
}
