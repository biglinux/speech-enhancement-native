//! Two independent sigmoid evaluations explicitly interleaved. Each lane keeps
//! the deployed operations/association. No reciprocal, rational or fast-math.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sigmoid_pair(x: __m256, y: __m256) -> (__m256, __m256) {
    let z = _mm256_setzero_ps();
    let lo = _mm256_set1_ps(-87.0);
    let hi = _mm256_set1_ps(88.0);
    let x = _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(z, x), lo), hi);
    let y = _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(z, y), lo), hi);
    let log2 = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let rx = _mm256_round_ps::<0x08>(_mm256_mul_ps(x, log2));
    let ry = _mm256_round_ps::<0x08>(_mm256_mul_ps(y, log2));
    let fx = _mm256_sub_ps(x, _mm256_mul_ps(rx, ln2));
    let fy = _mm256_sub_ps(y, _mm256_mul_ps(ry, ln2));
    let mut px = _mm256_set1_ps(1.0 / 120.0);
    let mut py = px;
    macro_rules! step {
        ($c:expr) => {{
            let c = _mm256_set1_ps($c);
            px = _mm256_add_ps(_mm256_mul_ps(px, fx), c);
            py = _mm256_add_ps(_mm256_mul_ps(py, fy), c);
        }};
    }
    step!(1.0 / 24.0);
    step!(1.0 / 6.0);
    step!(0.5);
    step!(1.0);
    step!(1.0);
    // r is integral in [-126,127]; these FP32 operations construct the same exponent bits.
    let off = _mm256_set1_ps(127.0);
    let shift = _mm256_set1_ps(8388608.0);
    let bx = _mm256_castsi256_ps(_mm256_cvttps_epi32(_mm256_mul_ps(
        _mm256_add_ps(rx, off),
        shift,
    )));
    let by = _mm256_castsi256_ps(_mm256_cvttps_epi32(_mm256_mul_ps(
        _mm256_add_ps(ry, off),
        shift,
    )));
    let one = _mm256_set1_ps(1.0);
    (
        _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(px, bx))),
        _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(py, by))),
    )
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub(super) unsafe fn gate(h: &mut [f32], wx: &[f32], rh: &[f32], b: &[f32], hs: usize) {
    let one = _mm256_set1_ps(1.0);
    for i in (0..hs).step_by(8) {
        let ld = |s: &[f32], o: usize| _mm256_loadu_ps(s.as_ptr().add(o));
        let zi = _mm256_add_ps(
            _mm256_add_ps(ld(wx, i), ld(rh, i)),
            _mm256_add_ps(ld(b, i), ld(b, 3 * hs + i)),
        );
        let ri = _mm256_add_ps(
            _mm256_add_ps(ld(wx, hs + i), ld(rh, hs + i)),
            _mm256_add_ps(ld(b, hs + i), ld(b, 4 * hs + i)),
        );
        let (z, rr) = sigmoid_pair(zi, ri);
        let inner = _mm256_mul_ps(rr, _mm256_add_ps(ld(rh, 2 * hs + i), ld(b, 5 * hs + i)));
        let hh = super::tanh8(_mm256_add_ps(
            _mm256_add_ps(ld(wx, 2 * hs + i), ld(b, 2 * hs + i)),
            inner,
        ));
        let newh = _mm256_add_ps(
            _mm256_mul_ps(_mm256_sub_ps(one, z), hh),
            _mm256_mul_ps(z, ld(h, i)),
        );
        _mm256_storeu_ps(h.as_mut_ptr().add(i), newh);
    }
}
#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    #[test]
    fn independent_schedule_matches_previous_gate() {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        for hs in [8, 64, 256, 512] {
            for scale in [0.001, 1.0, 10.0, 100.0] {
                let mut h: Vec<f32> = (0..hs).map(|i| (i as f32 % 17.0 - 8.0) / 8.0).collect();
                let mut old = h.clone();
                let wx: Vec<f32> = (0..3 * hs)
                    .map(|i| (i as f32 % 101.0 - 50.0) * scale)
                    .collect();
                let rh: Vec<f32> = (0..3 * hs)
                    .map(|i| (i as f32 % 31.0 - 15.0) * scale)
                    .collect();
                let b: Vec<f32> = (0..6 * hs)
                    .map(|i| (i as f32 % 13.0 - 6.0) * 0.01)
                    .collect();
                unsafe {
                    super::super::gate8_reference(&mut old, &wx, &rh, &b, hs);
                    super::gate(&mut h, &wx, &rh, &b, hs);
                }
                assert!(h.iter().zip(old).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
}
