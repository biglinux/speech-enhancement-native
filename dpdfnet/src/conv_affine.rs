//! AVX kernel for the 1x1 depthwise convolutions over 64 channels: a scale and
//! bias per channel, then the activation, with the generic path's FMA policy.
//! ReLU stays scalar at zero/NaN lanes to keep Rust's f32::max corner cases.
use super::super::Activation;
use super::{Conv, View};
use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
unsafe fn relu_store(y: *mut f32, x: __m256) {
    let z = _mm256_setzero_ps();
    let special = _mm256_movemask_ps(_mm256_or_ps(
        _mm256_cmp_ps::<{ _CMP_EQ_OQ }>(x, z),
        _mm256_cmp_ps::<{ _CMP_UNORD_Q }>(x, x),
    )) as u32;
    _mm256_storeu_ps(y, _mm256_max_ps(x, z));
    if special != 0 {
        let mut raw = [0.0f32; 8];
        _mm256_storeu_ps(raw.as_mut_ptr(), x);
        for (i, &v) in raw.iter().enumerate() {
            if special & (1 << i) != 0 {
                *y.add(i) = v.max(0.0);
            }
        }
    }
}

pub(super) fn affine_checked(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    // Shape 64 independent channels, kt=kf=1; AVX supported on tier 2 and 3.
    unsafe {
        affine(c, x, out, of, spacing, offset);
    }
}
#[target_feature(enable = "avx")]
unsafe fn affine(c: &Conv, x: View<'_>, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
    for j in (0..64).step_by(32) {
        let w0 = _mm256_loadu_ps(c.w.as_ptr().add(j));
        let b0 = _mm256_loadu_ps(c.b.as_ptr().add(j));
        let w1 = _mm256_loadu_ps(c.w.as_ptr().add(j + 8));
        let b1 = _mm256_loadu_ps(c.b.as_ptr().add(j + 8));
        let w2 = _mm256_loadu_ps(c.w.as_ptr().add(j + 16));
        let b2 = _mm256_loadu_ps(c.b.as_ptr().add(j + 16));
        let w3 = _mm256_loadu_ps(c.w.as_ptr().add(j + 24));
        let b3 = _mm256_loadu_ps(c.b.as_ptr().add(j + 24));
        for f in 0..of {
            let xi = f * c.stride;
            let y = out.as_mut_ptr().add(f * spacing + offset + j);
            let valid = xi >= c.pad && xi - c.pad < x.f;
            let row = if valid {
                x.data.as_ptr().add((xi - c.pad) * 64 + j)
            } else {
                std::ptr::null()
            };
            let a0 = if valid {
                _mm256_add_ps(b0, _mm256_mul_ps(_mm256_loadu_ps(row.add(0)), w0))
            } else {
                b0
            };
            let a1 = if valid {
                _mm256_add_ps(b1, _mm256_mul_ps(_mm256_loadu_ps(row.add(8)), w1))
            } else {
                b1
            };
            let a2 = if valid {
                _mm256_add_ps(b2, _mm256_mul_ps(_mm256_loadu_ps(row.add(16)), w2))
            } else {
                b2
            };
            let a3 = if valid {
                _mm256_add_ps(b3, _mm256_mul_ps(_mm256_loadu_ps(row.add(24)), w3))
            } else {
                b3
            };
            if matches!(c.act, Activation::Relu) {
                relu_store(y.add(0), a0);
                relu_store(y.add(8), a1);
                relu_store(y.add(16), a2);
                relu_store(y.add(24), a3);
            } else {
                _mm256_storeu_ps(y.add(0), a0);
                _mm256_storeu_ps(y.add(8), a1);
                _mm256_storeu_ps(y.add(16), a2);
                _mm256_storeu_ps(y.add(24), a3);
            }
        }
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;
    #[test]
    fn relu_epilogue_preserves_scalar_corner_cases() {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let values = [
            -0.0f32,
            0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
            -f32::from_bits(1),
            -2.0,
        ];
        let mut out = [0.0f32; 8];
        unsafe {
            relu_store(out.as_mut_ptr(), _mm256_loadu_ps(values.as_ptr()));
        }
        for i in 0..8 {
            assert_eq!(out[i].to_bits(), values[i].max(0.0).to_bits());
        }
    }
}
