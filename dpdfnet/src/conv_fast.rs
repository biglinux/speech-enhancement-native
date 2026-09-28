//! Dedicated kernels for the remaining generic convolution shapes.
//! Child of layers: geometry/weights are validated by Conv::load/Pipeline::load.
//! Preserve the previous per-output reduction order and its FMA policy:
//! wide first convolutions use FMA only on tier 3; op=5 and depthwise never do.
use super::{Conv, View};
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "r9-first-unroll", feature = "r9-affine-epilogue")
))]
#[path = "conv_round9.rs"]
mod round9;
pub(super) type Kernel = fn(&Conv, View<'_>, &mut [f32], usize, usize, usize);

pub(super) fn select(c: &Conv) -> Option<Kernel> {
    if cfg!(feature = "scalar-reference") || !cfg!(feature = "specialized-conv") {
        return None;
    }
    // Keep the already register-blocked dense 1x1 and depthwise 1x3 paths intact.
    #[cfg(target_arch = "x86_64")]
    {
        let tier = dfn_ops::simd_tier();
        if c.kt == 3 && c.kf == 3 && c.ci == c.groups && c.co == 64 && matches!(c.ci, 1 | 2) {
            #[cfg(feature = "r9-first-unroll")]
            if tier == 3 {
                return Some(round9::first_checked);
            }
            return match tier {
                3 => Some(first_fma_checked),
                2 => Some(first_avx_checked),
                1 => Some(first_sse_checked),
                _ => None,
            };
        }
        if c.kt == 5 && c.kf == 1 && c.ci == 64 && c.co == 10 && c.groups == 2 {
            if tier == 2 && c.df5_padded.is_some() {
                return Some(df5_avx_checked);
            }
            return if tier > 0 { Some(df5_checked) } else { None };
        }
        if c.kt == 1 && c.kf == 1 && c.groups == c.ci && c.ci == c.co {
            #[cfg(feature = "r9-affine-epilogue")]
            if tier >= 2
                && c.co == 64
                && matches!(c.act, super::Activation::None | super::Activation::Relu)
            {
                return Some(round9::affine_checked);
            }
            return if tier >= 2 {
                Some(affine_avx_checked)
            } else if tier == 1 {
                Some(affine_sse_checked)
            } else {
                None
            };
        }
    }
    if c.kt == 1 && c.kf == 3 && c.co == 1 && c.groups == 1 {
        return Some(mask_head);
    }
    None
}

fn frames<'a>(x: View<'a>) -> [&'a [f32]; 5] {
    let mut rows = [&[][..]; 5];
    for (t, row) in rows.iter_mut().enumerate().take(x.t) {
        let base = ((x.start + t) % x.t) * x.f * x.c;
        *row = &x.data[base..base + x.f * x.c];
    }
    rows
}
fn activate(c: &Conv, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
    for f in 0..of {
        c.act.apply_slice(
            &mut out[f * spacing + offset..f * spacing + offset + c.co],
            None,
        );
    }
}
fn mask_head(c: &Conv, x: View<'_>, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
    let rows = frames(x);
    let row = rows[0];
    for f in 0..of {
        let mut sum = c.b[0];
        for k in 0..3 {
            let xi = f * c.stride + k;
            if xi < c.pad || xi - c.pad >= x.f {
                continue;
            }
            let start = (xi - c.pad) * c.ci;
            // Deliberately reuse the previous ISA-specific dot's reduction tree.
            sum +=
                crate::kernels::dot_f32(&c.w[k * c.ci..(k + 1) * c.ci], &row[start..start + c.ci]);
        }
        out[f * spacing + offset] = c.act.scalar(sum);
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! madd {
    (true, $a:expr, $b:expr, $s:expr) => {
        _mm256_fmadd_ps($a, $b, $s)
    };
    (false, $a:expr, $b:expr, $s:expr) => {
        _mm256_add_ps($s, _mm256_mul_ps($a, $b))
    };
}
#[cfg(target_arch = "x86_64")]
macro_rules! first_body {
    ($c:expr, $x:expr, $out:expr, $of:expr, $spacing:expr, $offset:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        let (c, x, out, of, spacing, offset) = ($c, $x, $out, $of, $spacing, $offset);
        let rows = frames(x);
        let op = c.co / c.groups; // 64 or 32. Exactly one input per group.
        for f in 0..of {
            for g in 0..c.groups {
                for o in (0..op).step_by(32) {
                    let base = g * op + o;
                    let mut a0 = _mm256_loadu_ps(c.b.as_ptr().add(base));
                    let mut a1 = _mm256_loadu_ps(c.b.as_ptr().add(base + 8));
                    let mut a2 = _mm256_loadu_ps(c.b.as_ptr().add(base + 16));
                    let mut a3 = _mm256_loadu_ps(c.b.as_ptr().add(base + 24));
                    for t in 0..3 {
                        for k in 0..3 {
                            let xi = f * c.stride + k;
                            if xi < c.pad || xi - c.pad >= x.f {
                                continue;
                            }
                            let v = _mm256_set1_ps(rows[t][(xi - c.pad) * c.ci + g]);
                            let w = c.w.as_ptr().add((t * 3 + k) * c.co + base);
                            a0 = madd!($fma, v, _mm256_loadu_ps(w), a0);
                            a1 = madd!($fma, v, _mm256_loadu_ps(w.add(8)), a1);
                            a2 = madd!($fma, v, _mm256_loadu_ps(w.add(16)), a2);
                            a3 = madd!($fma, v, _mm256_loadu_ps(w.add(24)), a3);
                        }
                    }
                    let y = out.as_mut_ptr().add(f * spacing + offset + base);
                    _mm256_storeu_ps(y, a0);
                    _mm256_storeu_ps(y.add(8), a1);
                    _mm256_storeu_ps(y.add(16), a2);
                    _mm256_storeu_ps(y.add(24), a3);
                }
            }
        }
        activate(c, out, of, spacing, offset);
    }};
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn first_fma(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    first_body!(c, x, out, of, spacing, offset, true)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn first_avx(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    first_body!(c, x, out, of, spacing, offset, false)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn first_sse(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    use std::arch::x86_64::*;
    let rows = frames(x);
    let op = c.co / c.groups;
    for f in 0..of {
        for g in 0..c.groups {
            for o in (0..op).step_by(16) {
                let base = g * op + o;
                let mut a0 = _mm_loadu_ps(c.b.as_ptr().add(base));
                let mut a1 = _mm_loadu_ps(c.b.as_ptr().add(base + 4));
                let mut a2 = _mm_loadu_ps(c.b.as_ptr().add(base + 8));
                let mut a3 = _mm_loadu_ps(c.b.as_ptr().add(base + 12));
                for t in 0..3 {
                    for k in 0..3 {
                        let xi = f * c.stride + k;
                        if xi < c.pad || xi - c.pad >= x.f {
                            continue;
                        }
                        let v = _mm_set1_ps(rows[t][(xi - c.pad) * c.ci + g]);
                        let w = c.w.as_ptr().add((t * 3 + k) * c.co + base);
                        a0 = _mm_add_ps(a0, _mm_mul_ps(v, _mm_loadu_ps(w)));
                        a1 = _mm_add_ps(a1, _mm_mul_ps(v, _mm_loadu_ps(w.add(4))));
                        a2 = _mm_add_ps(a2, _mm_mul_ps(v, _mm_loadu_ps(w.add(8))));
                        a3 = _mm_add_ps(a3, _mm_mul_ps(v, _mm_loadu_ps(w.add(12))));
                    }
                }
                let y = out.as_mut_ptr().add(f * spacing + offset + base);
                _mm_storeu_ps(y, a0);
                _mm_storeu_ps(y.add(4), a1);
                _mm_storeu_ps(y.add(8), a2);
                _mm_storeu_ps(y.add(12), a3);
            }
        }
    }
    activate(c, out, of, spacing, offset);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn df5(c: &Conv, x: View<'_>, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
    use std::arch::x86_64::*;
    let rows = frames(x);
    for f in 0..of {
        let xi = f * c.stride;
        for g in 0..2 {
            let mut a = _mm_loadu_ps(c.b.as_ptr().add(g * 5));
            let mut fifth = c.b[g * 5 + 4];
            if xi >= c.pad && xi - c.pad < x.f {
                for t in 0..5 {
                    let input = &rows[t][(xi - c.pad) * 64 + g * 32..];
                    let weights = c.w.as_ptr().add(t * 320 + g * 160);
                    for i in 0..32 {
                        let v = input[i];
                        let w = weights.add(i * 5);
                        a = _mm_add_ps(a, _mm_mul_ps(_mm_loadu_ps(w), _mm_set1_ps(v)));
                        // Must stay mul+add even on FMA hosts: original op<=8 loop.
                        fifth += *w.add(4) * v;
                    }
                }
            }
            let y = out.as_mut_ptr().add(f * spacing + offset + g * 5);
            _mm_storeu_ps(y, a);
            *y.add(4) = fifth;
        }
    }
    activate(c, out, of, spacing, offset);
}

#[cfg(target_arch = "x86_64")]
macro_rules! affine_body {
    ($c:expr, $x:expr, $out:expr, $of:expr, $spacing:expr, $offset:expr,
     $width:expr, $load:ident, $store:ident, $add:ident, $mul:ident) => {{
        use std::arch::x86_64::*;
        let (c, x, out, of, spacing, offset) = ($c, $x, $out, $of, $spacing, $offset);
        let rows = frames(x);
        for f in 0..of {
            let xi = f * c.stride;
            let y = &mut out[f * spacing + offset..f * spacing + offset + c.co];
            if xi < c.pad || xi - c.pad >= x.f {
                y.copy_from_slice(&c.b);
                continue;
            }
            let input = rows[0].as_ptr().add((xi - c.pad) * c.ci);
            let mut j = 0;
            while j + $width <= c.co {
                let v = $add(
                    $load(c.b.as_ptr().add(j)),
                    $mul($load(input.add(j)), $load(c.w.as_ptr().add(j))),
                );
                $store(y.as_mut_ptr().add(j), v);
                j += $width;
            }
            while j < c.co {
                y[j] = c.b[j] + *input.add(j) * c.w[j];
                j += 1;
            }
        }
        activate(c, out, of, spacing, offset);
    }};
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn affine_avx(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    affine_body!(
        c,
        x,
        out,
        of,
        spacing,
        offset,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps
    )
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn affine_sse(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    affine_body!(
        c,
        x,
        out,
        of,
        spacing,
        offset,
        4,
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_add_ps,
        _mm_mul_ps
    )
}

// The safe entry is reachable only via load-time geometry + ISA selection.
#[cfg(target_arch = "x86_64")]
macro_rules! wrapper {
    ($name:ident,$target:ident) => {
        fn $name(c: &Conv, x: View<'_>, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
            debug_assert_eq!(x.c, c.ci);
            debug_assert_eq!(x.t, c.kt);
            debug_assert!(of == 0 || out.len() >= (of - 1) * spacing + offset + c.co);
            // SAFETY: private kernel chosen after validating geometry and ISA;
            // Pipeline supplies a matching View and non-overlapping output allocation.
            unsafe { $target(c, x, out, of, spacing, offset) }
        }
    };
}
#[cfg(target_arch = "x86_64")]
wrapper!(first_fma_checked, first_fma);
#[cfg(target_arch = "x86_64")]
wrapper!(first_avx_checked, first_avx);
#[cfg(target_arch = "x86_64")]
wrapper!(first_sse_checked, first_sse);
#[cfg(target_arch = "x86_64")]
wrapper!(df5_checked, df5);
#[cfg(target_arch = "x86_64")]
wrapper!(affine_avx_checked, affine_avx);
#[cfg(target_arch = "x86_64")]
wrapper!(affine_sse_checked, affine_sse);

/// Pad ONLY the 5-output grouped DF matrix, not all FP32 weights. 6,400 ->
/// 10,240 bytes, so the additional allocation is 10,240 bytes per instance
/// (original immutable weights remain shared). No bundle/schema change.
/// The three padding lanes do not become model channels.
pub(super) fn prepare_df5(c: &Conv) -> Option<crate::weights::F32s> {
    #[cfg(all(
        feature = "oldcpu-df5-avx",
        target_arch = "x86_64",
        not(feature = "scalar-reference")
    ))]
    if dfn_ops::simd_tier() == 2
        && c.kt == 5
        && c.kf == 1
        && c.ci == 64
        && c.co == 10
        && c.groups == 2
    {
        let mut w = vec![0.0f32; 5 * 2 * 32 * 8];
        for t in 0..5 {
            for g in 0..2 {
                for i in 0..32 {
                    let src = (t * 2 * 32 + g * 32 + i) * 5;
                    let dst = (t * 2 * 32 + g * 32 + i) * 8;
                    w[dst..dst + 5].copy_from_slice(&c.w[src..src + 5]);
                }
            }
        }
        return Some(crate::weights::F32s::aligned_copy(&w));
    }
    let _ = c;
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn df5_avx(
    c: &Conv,
    x: View<'_>,
    out: &mut [f32],
    of: usize,
    spacing: usize,
    offset: usize,
) {
    use std::arch::x86_64::*;
    let rows = frames(x);
    let w = c
        .df5_padded
        .as_ref()
        .expect("load-time DF5 packing invariant");
    for f in 0..of {
        let xi = f * c.stride;
        for g in 0..2 {
            let b = c.b.as_ptr().add(g * 5);
            let mut a = _mm256_set_m128(_mm_set_ss(*b.add(4)), _mm_loadu_ps(b));
            if xi >= c.pad && xi - c.pad < x.f {
                for t in 0..5 {
                    let input = rows[t].as_ptr().add((xi - c.pad) * 64 + g * 32);
                    let weight = w.as_ptr().add(t * 512 + g * 256);
                    for i in 0..32 {
                        // MUL then ADD in the same t/i order, even on an FMA CPU.
                        a = _mm256_add_ps(
                            a,
                            _mm256_mul_ps(
                                _mm256_load_ps(weight.add(i * 8)),
                                _mm256_set1_ps(*input.add(i)),
                            ),
                        );
                    }
                }
            }
            let y = out.as_mut_ptr().add(f * spacing + offset + g * 5);
            _mm_storeu_ps(y, _mm256_castps256_ps128(a));
            _mm_store_ss(y.add(4), _mm256_extractf128_ps::<1>(a));
        }
    }
    activate(c, out, of, spacing, offset);
}
#[cfg(target_arch = "x86_64")]
wrapper!(df5_avx_checked, df5_avx);
