//! Transposed f32 matrix-vector product `y = Aᵀ·x` (`A` row-major `[m][n]`) and
//! the grouped linear built on it. Every path sums each output column over
//! `i = 0..m` in order, starting from zero, so the SSE, AVX and scalar results are
//! bit-identical; AVX2 differs only by the single rounding of FMA.

use crate::simd_tier;

/// `y[..n] = A[m][n]ᵀ · x[..m]`.
#[inline]
pub fn matvec_t(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    matvec_on(simd_tier(), y, a, x, m, n);
}

#[inline]
fn matvec_on(tier: u8, y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m);
    assert!(a.len() >= m.checked_mul(n).expect("matvec_t dimensions overflow"));
    #[cfg(target_arch = "x86_64")]
    {
        // Tests pass a tier the host supports, as `simd_tier` does.
        match tier {
            // SAFETY: tier 3 means AVX2+FMA; the asserts above give the lengths.
            3 => return unsafe { matvec_avx2(y, a, x, m, n) },
            // SAFETY: tier 2 means AVX; the asserts above give the lengths.
            2 => return unsafe { matvec_avx(y, a, x, m, n) },
            // SAFETY: tier 1 means SSE4.1; the asserts above give the lengths.
            1 => return unsafe { matvec_sse(y, a, x, m, n) },
            _ => {}
        }
    }
    let _ = tier;
    y[..n].fill(0.0);
    for i in 0..m {
        let row = &a[i * n..i * n + n];
        for j in 0..n {
            y[j] += x[i] * row[j];
        }
    }
}

/// Grouped linear (einsum `g i -> g o`): per group, `y_g = W_gᵀ · x_g`, with `W_g`
/// laid out `[in_per_group][out_per_group]`.
pub fn grouped_linear(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    groups: usize,
    in_pg: usize,
    out_pg: usize,
) {
    for g in 0..groups {
        let xg = &x[g * in_pg..g * in_pg + in_pg];
        let wg = &w[g * in_pg * out_pg..g * in_pg * out_pg + in_pg * out_pg];
        let yg = &mut y[g * out_pg..g * out_pg + out_pg];
        matvec_t(yg, wg, xg, in_pg, out_pg);
    }
}

/// 256-bit body: 64 outputs stay in eight registers across the whole reduction,
/// then 32- and 8-wide tiles and a scalar tail. `$fma` selects FMA or a multiply
/// and add.
#[cfg(target_arch = "x86_64")]
macro_rules! matvec_256 {
    ($y:expr, $a:expr, $x:expr, $m:expr, $n:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (y, a, x, m, n) = ($y, $a, $x, $m, $n);
        let (ap, xp, yp) = (a.as_ptr(), x.as_ptr(), y.as_mut_ptr());
        let mut j = 0usize;
        while j + 64 <= n {
            let z = _mm256_setzero_ps();
            let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7) =
                (z, z, z, z, z, z, z, z);
            for i in 0..m {
                let xi = _mm256_set1_ps(*xp.add(i));
                let row = ap.add(i * n + j);
                s0 = fma_sel!($fma, _mm256_loadu_ps(row), xi, s0);
                s1 = fma_sel!($fma, _mm256_loadu_ps(row.add(8)), xi, s1);
                s2 = fma_sel!($fma, _mm256_loadu_ps(row.add(16)), xi, s2);
                s3 = fma_sel!($fma, _mm256_loadu_ps(row.add(24)), xi, s3);
                s4 = fma_sel!($fma, _mm256_loadu_ps(row.add(32)), xi, s4);
                s5 = fma_sel!($fma, _mm256_loadu_ps(row.add(40)), xi, s5);
                s6 = fma_sel!($fma, _mm256_loadu_ps(row.add(48)), xi, s6);
                s7 = fma_sel!($fma, _mm256_loadu_ps(row.add(56)), xi, s7);
            }
            _mm256_storeu_ps(yp.add(j), s0);
            _mm256_storeu_ps(yp.add(j + 8), s1);
            _mm256_storeu_ps(yp.add(j + 16), s2);
            _mm256_storeu_ps(yp.add(j + 24), s3);
            _mm256_storeu_ps(yp.add(j + 32), s4);
            _mm256_storeu_ps(yp.add(j + 40), s5);
            _mm256_storeu_ps(yp.add(j + 48), s6);
            _mm256_storeu_ps(yp.add(j + 56), s7);
            j += 64;
        }
        while j + 32 <= n {
            let z = _mm256_setzero_ps();
            let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);
            for i in 0..m {
                let xi = _mm256_set1_ps(*xp.add(i));
                let row = ap.add(i * n + j);
                s0 = fma_sel!($fma, _mm256_loadu_ps(row), xi, s0);
                s1 = fma_sel!($fma, _mm256_loadu_ps(row.add(8)), xi, s1);
                s2 = fma_sel!($fma, _mm256_loadu_ps(row.add(16)), xi, s2);
                s3 = fma_sel!($fma, _mm256_loadu_ps(row.add(24)), xi, s3);
            }
            _mm256_storeu_ps(yp.add(j), s0);
            _mm256_storeu_ps(yp.add(j + 8), s1);
            _mm256_storeu_ps(yp.add(j + 16), s2);
            _mm256_storeu_ps(yp.add(j + 24), s3);
            j += 32;
        }
        while j + 8 <= n {
            let mut s = _mm256_setzero_ps();
            for i in 0..m {
                let xi = _mm256_set1_ps(*xp.add(i));
                s = fma_sel!($fma, _mm256_loadu_ps(ap.add(i * n + j)), xi, s);
            }
            _mm256_storeu_ps(yp.add(j), s);
            j += 8;
        }
        while j < n {
            let mut s = 0.0f32;
            for i in 0..m {
                s += *xp.add(i) * *ap.add(i * n + j);
            }
            *yp.add(j) = s;
            j += 1;
        }
    }};
}

/// # Safety
/// The CPU must support AVX; `y` must hold `n` values, `x` `m` and `a` `m*n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn matvec_avx(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    // SAFETY: row `i` reads `a[i*n + j..]` for `j < n` and `x[i]` for `i < m`, and
    // the stores write `y[..n]`, all inside the caller's lengths.
    unsafe { matvec_256!(y, a, x, m, n, false) }
}

/// # Safety
/// The CPU must support AVX2 and FMA; `y` must hold `n` values, `x` `m` and `a` `m*n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    // SAFETY: row `i` reads `a[i*n + j..]` for `j < n` and `x[i]` for `i < m`, and
    // the stores write `y[..n]`, all inside the caller's lengths.
    unsafe { matvec_256!(y, a, x, m, n, true) }
}

/// 128-bit body of [`matvec_t`]: 32 outputs in eight registers, then 4-wide tiles.
///
/// # Safety
/// The CPU must support SSE4.1; `y` must hold `n` values, `x` `m` and `a` `m*n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn matvec_sse(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    // SAFETY: row `i` reads `a[i*n + j..]` for `j < n` and `x[i]` for `i < m`, and
    // the stores write `y[..n]`, all inside the caller's lengths.
    unsafe {
        use std::arch::x86_64::*;
        let (ap, xp, yp) = (a.as_ptr(), x.as_ptr(), y.as_mut_ptr());
        let mut j = 0usize;
        while j + 32 <= n {
            let mut s = [_mm_setzero_ps(); 8];
            for i in 0..m {
                let xi = _mm_set1_ps(*xp.add(i));
                let row = ap.add(i * n + j);
                for (k, v) in s.iter_mut().enumerate() {
                    *v = _mm_add_ps(*v, _mm_mul_ps(_mm_loadu_ps(row.add(4 * k)), xi));
                }
            }
            for (k, v) in s.iter().enumerate() {
                _mm_storeu_ps(yp.add(j + 4 * k), *v);
            }
            j += 32;
        }
        while j + 4 <= n {
            let mut s = _mm_setzero_ps();
            for i in 0..m {
                let xi = _mm_set1_ps(*xp.add(i));
                s = _mm_add_ps(s, _mm_mul_ps(_mm_loadu_ps(ap.add(i * n + j)), xi));
            }
            _mm_storeu_ps(yp.add(j), s);
            j += 4;
        }
        while j < n {
            let mut s = 0.0f32;
            for i in 0..m {
                s += *xp.add(i) * *ap.add(i * n + j);
            }
            *yp.add(j) = s;
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 71 + seed) % 257) as f32 / 128.0 - 1.0)
            .collect()
    }

    // Covers every tile boundary of both widths, and checks that nothing past `n`
    // is written.
    #[test]
    fn every_tier_matches_its_reference_bit_for_bit() {
        for m in [0usize, 1, 3, 16, 64, 128, 256] {
            for n in [0usize, 1, 4, 7, 8, 15, 31, 32, 33, 63, 64, 65, 96, 130, 192] {
                let a = values(m * n, 0);
                let x = values(m, 3);
                let mut plain = vec![0.0f32; n];
                let mut fused = vec![0.0f32; n];
                for j in 0..n {
                    for i in 0..m {
                        plain[j] += x[i] * a[i * n + j];
                        fused[j] = a[i * n + j].mul_add(x[i], fused[j]);
                    }
                }
                for tier in 0..=simd_tier() {
                    let mut y = vec![7.0f32; n + 1];
                    matvec_on(tier, &mut y, &a, &x, m, n);
                    let want = if tier == 3 { &fused } else { &plain };
                    let same = y[..n]
                        .iter()
                        .zip(want)
                        .all(|(g, w)| g.to_bits() == w.to_bits());
                    assert!(same, "tier {tier} m={m} n={n}");
                    assert_eq!(y[n], 7.0, "tier {tier} wrote past n");
                }
            }
        }
    }
}
