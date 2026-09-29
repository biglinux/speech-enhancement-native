//! R9 direct kernels. Nine taps stay in t-major/k-major order, with the
//! original FMA policy. No Winograd transform, lookahead change, or extra state.
//! Relu is kept scalar at zero/NaN lanes to preserve Rust f32::max corner cases.
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

#[cfg(feature = "r9-first-unroll")]
pub(super) fn first_checked(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    // Reached only through shape and tier-3 selection; x owns exactly 3 rows.
    unsafe {
        if c.ci == 1 {
            first::<1, 64>(c, x, out, of, spacing, offset)
        } else {
            first::<2, 32>(c, x, out, of, spacing, offset)
        }
    }
}
#[cfg(feature = "r9-first-unroll")]
#[target_feature(enable = "avx2,fma")]
unsafe fn first<const CI: usize, const OP: usize>(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    let mut rows = [std::ptr::null::<f32>(); 3];
    for (t, row) in rows.iter_mut().enumerate() {
        *row = x.data.as_ptr().add(((x.start + t) % 3) * x.f * CI);
    }
    for f in 0..of {
        let center = f * c.stride;
        let interior = center >= c.pad && center - c.pad + 2 < x.f;
        for g in 0..CI {
            if OP == 64 {
                first64(
                    c,
                    rows,
                    out.as_mut_ptr().add(f * spacing + offset + g * OP),
                    center,
                    x.f,
                    g,
                    interior,
                );
            } else {
                first32(
                    c,
                    rows,
                    out.as_mut_ptr().add(f * spacing + offset + g * OP),
                    center,
                    x.f,
                    g,
                    interior,
                );
            }
        }
    }
}
#[cfg(feature = "r9-first-unroll")]
#[target_feature(enable = "avx2,fma")]
unsafe fn first64(
    c: &Conv,
    rows: [*const f32; 3],
    y: *mut f32,
    center: usize,
    nf: usize,
    g: usize,
    interior: bool,
) {
    let base = g * 64;
    let mut a0 = _mm256_loadu_ps(c.b.as_ptr().add(base + 0));
    let mut a1 = _mm256_loadu_ps(c.b.as_ptr().add(base + 8));
    let mut a2 = _mm256_loadu_ps(c.b.as_ptr().add(base + 16));
    let mut a3 = _mm256_loadu_ps(c.b.as_ptr().add(base + 24));
    let mut a4 = _mm256_loadu_ps(c.b.as_ptr().add(base + 32));
    let mut a5 = _mm256_loadu_ps(c.b.as_ptr().add(base + 40));
    let mut a6 = _mm256_loadu_ps(c.b.as_ptr().add(base + 48));
    let mut a7 = _mm256_loadu_ps(c.b.as_ptr().add(base + 56));
    if interior {
        let f0 = center - c.pad;
        let x = _mm256_set1_ps(*rows[0].add((f0 + 0) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[0].add((f0 + 1) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[0].add((f0 + 2) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[1].add((f0 + 0) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[1].add((f0 + 1) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[1].add((f0 + 2) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[2].add((f0 + 0) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[2].add((f0 + 1) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 56)), a7);
        let x = _mm256_set1_ps(*rows[2].add((f0 + 2) * 1 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 24)), a3);
        a4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 32)), a4);
        a5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 40)), a5);
        a6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 48)), a6);
        a7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 56)), a7);
    } else {
        for (t, row) in rows.iter().enumerate() {
            for k in 0..3 {
                let xi = center + k;
                if xi < c.pad || xi - c.pad >= nf {
                    continue;
                }
                let x = _mm256_set1_ps(*(*row).add((xi - c.pad) * 1 + g));
                a0 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 0)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 8)),
                    a1,
                );
                a2 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 16)),
                    a2,
                );
                a3 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 24)),
                    a3,
                );
                a4 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 32)),
                    a4,
                );
                a5 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 40)),
                    a5,
                );
                a6 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 48)),
                    a6,
                );
                a7 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 56)),
                    a7,
                );
            }
        }
    }
    if matches!(c.act, Activation::Relu) {
        relu_store(y.add(0), a0);
        relu_store(y.add(8), a1);
        relu_store(y.add(16), a2);
        relu_store(y.add(24), a3);
        relu_store(y.add(32), a4);
        relu_store(y.add(40), a5);
        relu_store(y.add(48), a6);
        relu_store(y.add(56), a7);
    } else {
        _mm256_storeu_ps(y.add(0), a0);
        _mm256_storeu_ps(y.add(8), a1);
        _mm256_storeu_ps(y.add(16), a2);
        _mm256_storeu_ps(y.add(24), a3);
        _mm256_storeu_ps(y.add(32), a4);
        _mm256_storeu_ps(y.add(40), a5);
        _mm256_storeu_ps(y.add(48), a6);
        _mm256_storeu_ps(y.add(56), a7);
        c.act
            .apply_slice(std::slice::from_raw_parts_mut(y, 64), None);
    }
}
#[cfg(feature = "r9-first-unroll")]
#[target_feature(enable = "avx2,fma")]
unsafe fn first32(
    c: &Conv,
    rows: [*const f32; 3],
    y: *mut f32,
    center: usize,
    nf: usize,
    g: usize,
    interior: bool,
) {
    let base = g * 32;
    let mut a0 = _mm256_loadu_ps(c.b.as_ptr().add(base + 0));
    let mut a1 = _mm256_loadu_ps(c.b.as_ptr().add(base + 8));
    let mut a2 = _mm256_loadu_ps(c.b.as_ptr().add(base + 16));
    let mut a3 = _mm256_loadu_ps(c.b.as_ptr().add(base + 24));
    if interior {
        let f0 = center - c.pad;
        let x = _mm256_set1_ps(*rows[0].add((f0 + 0) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(0 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[0].add((f0 + 1) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(64 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[0].add((f0 + 2) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(128 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[1].add((f0 + 0) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(192 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[1].add((f0 + 1) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(256 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[1].add((f0 + 2) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(320 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[2].add((f0 + 0) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(384 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[2].add((f0 + 1) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(448 + base + 24)), a3);
        let x = _mm256_set1_ps(*rows[2].add((f0 + 2) * 2 + g));
        a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 0)), a0);
        a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 8)), a1);
        a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 16)), a2);
        a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c.w.as_ptr().add(512 + base + 24)), a3);
    } else {
        for (t, row) in rows.iter().enumerate() {
            for k in 0..3 {
                let xi = center + k;
                if xi < c.pad || xi - c.pad >= nf {
                    continue;
                }
                let x = _mm256_set1_ps(*(*row).add((xi - c.pad) * 2 + g));
                a0 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 0)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 8)),
                    a1,
                );
                a2 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 16)),
                    a2,
                );
                a3 = _mm256_fmadd_ps(
                    x,
                    _mm256_loadu_ps(c.w.as_ptr().add((t * 3 + k) * 64 + base + 24)),
                    a3,
                );
            }
        }
    }
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
        c.act
            .apply_slice(std::slice::from_raw_parts_mut(y, 32), None);
    }
}

#[cfg(feature = "r9-affine-epilogue")]
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
#[cfg(feature = "r9-affine-epilogue")]
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
