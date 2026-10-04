//! f32 matrix-vector products over wide output tiles, for 64 or more outputs
//! (32 on SSE). More independent accumulators per pass; every output keeps the
//! reduction order and FMA policy of the narrow kernels, so results are equal.
use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
pub(super) unsafe fn matvec_avx(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    let mut j = 0usize;
    while j + 64 <= n {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut s4 = _mm256_setzero_ps();
        let mut s5 = _mm256_setzero_ps();
        let mut s6 = _mm256_setzero_ps();
        let mut s7 = _mm256_setzero_ps();
        for i in 0..m {
            let xx = _mm256_set1_ps(*x.get_unchecked(i));
            let row = a.as_ptr().add(i * n + j);
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(row.add(0)), xx));
            s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(row.add(8)), xx));
            s2 = _mm256_add_ps(s2, _mm256_mul_ps(_mm256_loadu_ps(row.add(16)), xx));
            s3 = _mm256_add_ps(s3, _mm256_mul_ps(_mm256_loadu_ps(row.add(24)), xx));
            s4 = _mm256_add_ps(s4, _mm256_mul_ps(_mm256_loadu_ps(row.add(32)), xx));
            s5 = _mm256_add_ps(s5, _mm256_mul_ps(_mm256_loadu_ps(row.add(40)), xx));
            s6 = _mm256_add_ps(s6, _mm256_mul_ps(_mm256_loadu_ps(row.add(48)), xx));
            s7 = _mm256_add_ps(s7, _mm256_mul_ps(_mm256_loadu_ps(row.add(56)), xx));
        }
        _mm256_storeu_ps(y.as_mut_ptr().add(j), s0);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 8), s1);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 16), s2);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 24), s3);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 32), s4);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 40), s5);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 48), s6);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 56), s7);
        j += 64;
    }
    while j + 8 <= n {
        let mut ss = _mm256_setzero_ps();
        for i in 0..m {
            let xx = _mm256_set1_ps(*x.get_unchecked(i));
            let ww = _mm256_loadu_ps(a.as_ptr().add(i * n + j));
            ss = _mm256_add_ps(ss, _mm256_mul_ps(ww, xx));
        }
        _mm256_storeu_ps(y.as_mut_ptr().add(j), ss);
        j += 8;
    }
    while j < n {
        let mut ss = 0.0f32;
        for i in 0..m {
            ss += *x.get_unchecked(i) * *a.get_unchecked(i * n + j);
        }
        *y.get_unchecked_mut(j) = ss;
        j += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn matvec_avx2(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    let mut j = 0usize;
    while j + 64 <= n {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut s4 = _mm256_setzero_ps();
        let mut s5 = _mm256_setzero_ps();
        let mut s6 = _mm256_setzero_ps();
        let mut s7 = _mm256_setzero_ps();
        for i in 0..m {
            let xx = _mm256_set1_ps(*x.get_unchecked(i));
            let row = a.as_ptr().add(i * n + j);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(0)), xx, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(8)), xx, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(16)), xx, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(24)), xx, s3);
            s4 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(32)), xx, s4);
            s5 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(40)), xx, s5);
            s6 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(48)), xx, s6);
            s7 = _mm256_fmadd_ps(_mm256_loadu_ps(row.add(56)), xx, s7);
        }
        _mm256_storeu_ps(y.as_mut_ptr().add(j), s0);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 8), s1);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 16), s2);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 24), s3);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 32), s4);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 40), s5);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 48), s6);
        _mm256_storeu_ps(y.as_mut_ptr().add(j + 56), s7);
        j += 64;
    }
    while j + 8 <= n {
        let mut ss = _mm256_setzero_ps();
        for i in 0..m {
            let xx = _mm256_set1_ps(*x.get_unchecked(i));
            let ww = _mm256_loadu_ps(a.as_ptr().add(i * n + j));
            ss = _mm256_fmadd_ps(ww, xx, ss);
        }
        _mm256_storeu_ps(y.as_mut_ptr().add(j), ss);
        j += 8;
    }
    while j < n {
        let mut ss = 0.0f32;
        for i in 0..m {
            ss += *x.get_unchecked(i) * *a.get_unchecked(i * n + j);
        }
        *y.get_unchecked_mut(j) = ss;
        j += 1;
    }
}

#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn matvec_sse(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    let mut j = 0usize;
    while j + 32 <= n {
        let mut s0 = _mm_setzero_ps();
        let mut s1 = _mm_setzero_ps();
        let mut s2 = _mm_setzero_ps();
        let mut s3 = _mm_setzero_ps();
        let mut s4 = _mm_setzero_ps();
        let mut s5 = _mm_setzero_ps();
        let mut s6 = _mm_setzero_ps();
        let mut s7 = _mm_setzero_ps();
        for i in 0..m {
            let xx = _mm_set1_ps(*x.get_unchecked(i));
            let row = a.as_ptr().add(i * n + j);
            s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(row.add(0)), xx));
            s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(row.add(4)), xx));
            s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_loadu_ps(row.add(8)), xx));
            s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_loadu_ps(row.add(12)), xx));
            s4 = _mm_add_ps(s4, _mm_mul_ps(_mm_loadu_ps(row.add(16)), xx));
            s5 = _mm_add_ps(s5, _mm_mul_ps(_mm_loadu_ps(row.add(20)), xx));
            s6 = _mm_add_ps(s6, _mm_mul_ps(_mm_loadu_ps(row.add(24)), xx));
            s7 = _mm_add_ps(s7, _mm_mul_ps(_mm_loadu_ps(row.add(28)), xx));
        }
        _mm_storeu_ps(y.as_mut_ptr().add(j), s0);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 4), s1);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 8), s2);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 12), s3);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 16), s4);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 20), s5);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 24), s6);
        _mm_storeu_ps(y.as_mut_ptr().add(j + 28), s7);
        j += 32;
    }
    while j + 4 <= n {
        let mut ss = _mm_setzero_ps();
        for i in 0..m {
            let xx = _mm_set1_ps(*x.get_unchecked(i));
            let ww = _mm_loadu_ps(a.as_ptr().add(i * n + j));
            ss = _mm_add_ps(ss, _mm_mul_ps(ww, xx));
        }
        _mm_storeu_ps(y.as_mut_ptr().add(j), ss);
        j += 4;
    }
    while j < n {
        let mut ss = 0.0f32;
        for i in 0..m {
            ss += *x.get_unchecked(i) * *a.get_unchecked(i * n + j);
        }
        *y.get_unchecked_mut(j) = ss;
        j += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Mat = unsafe fn(&mut [f32], &[f32], &[f32], usize, usize);
    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }
    #[test]
    fn wide_float_tiles_retain_per_output_order_and_isa_fma_policy() {
        let mut pairs: Vec<(Mat, Mat)> = Vec::new();
        if std::is_x86_feature_detected!("sse4.1") {
            pairs.push((matvec_sse, super::super::matvec_t_sse));
        }
        if std::is_x86_feature_detected!("avx") {
            pairs.push((matvec_avx, super::super::matvec_t_avx));
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            pairs.push((matvec_avx2, super::super::matvec_t_avx2));
        }
        for m in [0usize, 1, 3, 16, 64, 128, 256] {
            for n in [
                0usize, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 72, 96, 128, 192,
            ] {
                let a: Vec<f32> = (0..m * n)
                    .map(|i| ((i * 71 % 257) as f32 - 128.0) / 127.25)
                    .collect();
                let x: Vec<f32> = (0..m)
                    .map(|i| ((i * 193 % 127) as f32 - 63.0) / 63.75)
                    .collect();
                for &(new, old) in &pairs {
                    let mut y = vec![91.0; n + 2];
                    let mut reference = vec![0.0; n];
                    unsafe {
                        old(&mut reference, &a, &x, m, n);
                        new(&mut y[1..n + 1], &a, &x, m, n);
                    }
                    assert_eq!(bits(&y[1..n + 1]), bits(&reference), "m={m} n={n}");
                    assert_eq!(y[0], 91.0);
                    assert_eq!(y[n + 1], 91.0);
                }
            }
        }
    }
}
