//! Parallel independent GROUPS, not frequencies, in small grouped linears.
//! Each output still reduces inputs from 0..ip using the same FMA policy.
//! No dense zero-filled expansion, transformed weights, or batching latency.
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn grouped(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    groups: usize,
    ip: usize,
    op: usize,
) {
    match op {
        8 => group8(y, x, w, groups, ip),
        16 => group16(y, x, w, groups, ip),
        32 => group32(y, x, w, groups, ip),
        _ => unreachable!("private validated output width"),
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn group8(y: &mut [f32], x: &[f32], w: &[f32], groups: usize, ip: usize) {
    let mut g = 0;
    while g + 8 <= groups {
        let mut a0_0 = _mm256_setzero_ps();
        let mut a1_0 = _mm256_setzero_ps();
        let mut a2_0 = _mm256_setzero_ps();
        let mut a3_0 = _mm256_setzero_ps();
        let mut a4_0 = _mm256_setzero_ps();
        let mut a5_0 = _mm256_setzero_ps();
        let mut a6_0 = _mm256_setzero_ps();
        let mut a7_0 = _mm256_setzero_ps();
        for i in 0..ip {
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 0) * ip + i));
            a0_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 8 + 0)),
                xx,
                a0_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 1) * ip + i));
            a1_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 8 + 0)),
                xx,
                a1_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 2) * ip + i));
            a2_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 2) * ip + i) * 8 + 0)),
                xx,
                a2_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 3) * ip + i));
            a3_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 3) * ip + i) * 8 + 0)),
                xx,
                a3_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 4) * ip + i));
            a4_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 4) * ip + i) * 8 + 0)),
                xx,
                a4_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 5) * ip + i));
            a5_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 5) * ip + i) * 8 + 0)),
                xx,
                a5_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 6) * ip + i));
            a6_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 6) * ip + i) * 8 + 0)),
                xx,
                a6_0,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 7) * ip + i));
            a7_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 7) * ip + i) * 8 + 0)),
                xx,
                a7_0,
            );
        }
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 8 + 0), a0_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 8 + 0), a1_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 2) * 8 + 0), a2_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 3) * 8 + 0), a3_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 4) * 8 + 0), a4_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 5) * 8 + 0), a5_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 6) * 8 + 0), a6_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 7) * 8 + 0), a7_0);
        g += 8;
    }
    while g < groups {
        super::matvec_t_avx2(
            &mut y[g * 8..(g + 1) * 8],
            &w[g * ip * 8..(g + 1) * ip * 8],
            &x[g * ip..(g + 1) * ip],
            ip,
            8,
        );
        g += 1;
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn group16(y: &mut [f32], x: &[f32], w: &[f32], groups: usize, ip: usize) {
    let mut g = 0;
    while g + 4 <= groups {
        let mut a0_0 = _mm256_setzero_ps();
        let mut a0_1 = _mm256_setzero_ps();
        let mut a1_0 = _mm256_setzero_ps();
        let mut a1_1 = _mm256_setzero_ps();
        let mut a2_0 = _mm256_setzero_ps();
        let mut a2_1 = _mm256_setzero_ps();
        let mut a3_0 = _mm256_setzero_ps();
        let mut a3_1 = _mm256_setzero_ps();
        for i in 0..ip {
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 0) * ip + i));
            a0_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 16 + 0)),
                xx,
                a0_0,
            );
            a0_1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 16 + 8)),
                xx,
                a0_1,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 1) * ip + i));
            a1_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 16 + 0)),
                xx,
                a1_0,
            );
            a1_1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 16 + 8)),
                xx,
                a1_1,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 2) * ip + i));
            a2_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 2) * ip + i) * 16 + 0)),
                xx,
                a2_0,
            );
            a2_1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 2) * ip + i) * 16 + 8)),
                xx,
                a2_1,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 3) * ip + i));
            a3_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 3) * ip + i) * 16 + 0)),
                xx,
                a3_0,
            );
            a3_1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 3) * ip + i) * 16 + 8)),
                xx,
                a3_1,
            );
        }
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 16 + 0), a0_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 16 + 8), a0_1);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 16 + 0), a1_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 16 + 8), a1_1);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 2) * 16 + 0), a2_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 2) * 16 + 8), a2_1);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 3) * 16 + 0), a3_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 3) * 16 + 8), a3_1);
        g += 4;
    }
    while g < groups {
        super::matvec_t_avx2(
            &mut y[g * 16..(g + 1) * 16],
            &w[g * ip * 16..(g + 1) * ip * 16],
            &x[g * ip..(g + 1) * ip],
            ip,
            16,
        );
        g += 1;
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn group32(y: &mut [f32], x: &[f32], w: &[f32], groups: usize, ip: usize) {
    let mut g = 0;
    while g + 2 <= groups {
        let mut a0_0 = _mm256_setzero_ps();
        let mut a0_1 = _mm256_setzero_ps();
        let mut a0_2 = _mm256_setzero_ps();
        let mut a0_3 = _mm256_setzero_ps();
        let mut a1_0 = _mm256_setzero_ps();
        let mut a1_1 = _mm256_setzero_ps();
        let mut a1_2 = _mm256_setzero_ps();
        let mut a1_3 = _mm256_setzero_ps();
        for i in 0..ip {
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 0) * ip + i));
            a0_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 32 + 0)),
                xx,
                a0_0,
            );
            a0_1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 32 + 8)),
                xx,
                a0_1,
            );
            a0_2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 32 + 16)),
                xx,
                a0_2,
            );
            a0_3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 0) * ip + i) * 32 + 24)),
                xx,
                a0_3,
            );
            let xx = _mm256_set1_ps(*x.get_unchecked((g + 1) * ip + i));
            a1_0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 32 + 0)),
                xx,
                a1_0,
            );
            a1_1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 32 + 8)),
                xx,
                a1_1,
            );
            a1_2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 32 + 16)),
                xx,
                a1_2,
            );
            a1_3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(((g + 1) * ip + i) * 32 + 24)),
                xx,
                a1_3,
            );
        }
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 32 + 0), a0_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 32 + 8), a0_1);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 32 + 16), a0_2);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 0) * 32 + 24), a0_3);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 32 + 0), a1_0);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 32 + 8), a1_1);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 32 + 16), a1_2);
        _mm256_storeu_ps(y.as_mut_ptr().add((g + 1) * 32 + 24), a1_3);
        g += 2;
    }
    while g < groups {
        super::matvec_t_avx2(
            &mut y[g * 32..(g + 1) * 32],
            &w[g * ip * 32..(g + 1) * ip * 32],
            &x[g * ip..(g + 1) * ip],
            ip,
            32,
        );
        g += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_match_individual_matvec_bitwise() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for op in [8usize, 16, 32] {
            for ip in [0usize, 1, 3, 8, 16, 32, 64, 80, 96] {
                for groups in [1usize, 2, 3, 4, 7, 8, 9, 16, 32] {
                    let x: Vec<f32> = (0..groups * ip)
                        .map(|i| ((i * 37 % 257) as f32 - 128.0) * 0.017)
                        .collect();
                    let w: Vec<f32> = (0..groups * ip * op)
                        .map(|i| ((i * 173 % 509) as f32 - 254.0) * 0.0031)
                        .collect();
                    let mut out = vec![91.25; groups * op + 2];
                    let mut want = vec![0.0; groups * op];
                    unsafe {
                        grouped(&mut out[1..groups * op + 1], &x, &w, groups, ip, op);
                        for g in 0..groups {
                            super::super::matvec_t_avx2(
                                &mut want[g * op..(g + 1) * op],
                                &w[g * ip * op..(g + 1) * ip * op],
                                &x[g * ip..(g + 1) * ip],
                                ip,
                                op,
                            );
                        }
                    }
                    assert_eq!(out[0], 91.25);
                    assert_eq!(out[groups * op + 1], 91.25);
                    assert_eq!(
                        out[1..groups * op + 1]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "groups={groups} ip={ip} op={op}"
                    );
                }
            }
        }
    }
}
