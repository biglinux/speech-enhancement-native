//! Register-blocked f32 matvec over wide output tiles, ported from dpdfnet-ops;
//! every output keeps its per-column reduction order.
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
