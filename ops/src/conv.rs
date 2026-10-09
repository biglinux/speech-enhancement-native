//! Convolutions over the frequency axis: depthwise 3-tap rows and pointwise
//! (1x1) mixes. [`depthwise_1x3`] and [`pointwise_conv2d`] give the same bits on
//! every tier.

use crate::simd_tier;

/// Depthwise convolution with one temporal and three frequency taps, one group per
/// channel, over a channel-contiguous frame `src[in_f][co]`. `w` holds the taps as
/// `[3][co]`. Writes bias plus the taps that fall inside the frame, with no
/// activation, to `out[f*spacing + offset ..][..co]` for each output position
/// `f < of`.
pub fn depthwise_1x3(
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    depthwise_on(
        simd_tier(),
        out,
        src,
        w,
        b,
        co,
        of,
        in_f,
        stride,
        pad,
        spacing,
        offset,
    );
}

fn depthwise_on(
    tier: u8,
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    assert!(
        w.len() >= 3 * co && b.len() >= co,
        "depthwise_1x3: weights too short"
    );
    if of == 0 {
        return;
    }
    let end = (of - 1)
        .checked_mul(spacing)
        .and_then(|v| v.checked_add(offset)?.checked_add(co))
        .expect("depthwise_1x3: output extent overflows");
    assert!(out.len() >= end, "depthwise_1x3: output too short");
    let frame = in_f
        .checked_mul(co)
        .expect("depthwise_1x3: input extent overflows");
    assert!(src.len() >= frame, "depthwise_1x3: input too short");
    #[cfg(target_arch = "x86_64")]
    {
        // Tests pass a tier the host supports, as `simd_tier` does.
        match tier {
            2 | 3 => {
                // SAFETY: tier 2 and up means AVX; the asserts above give the lengths.
                unsafe {
                    depthwise_avx(out, src, w, b, co, of, in_f, stride, pad, spacing, offset)
                };
                return;
            }
            1 => {
                // SAFETY: tier 1 means SSE4.1; the asserts above give the lengths.
                unsafe {
                    depthwise_sse(out, src, w, b, co, of, in_f, stride, pad, spacing, offset)
                };
                return;
            }
            _ => {}
        }
    }
    let _ = tier;
    for f in 0..of {
        let dst = &mut out[f * spacing + offset..f * spacing + offset + co];
        dst.copy_from_slice(&b[..co]);
        for k in 0..3 {
            let fr = f * stride + k;
            if fr < pad || fr - pad >= in_f {
                continue;
            }
            let xr = &src[(fr - pad) * co..(fr - pad) * co + co];
            let wk = &w[k * co..k * co + co];
            for c in 0..co {
                dst[c] += xr[c] * wk[c];
            }
        }
    }
}

/// The taps of output position `f` that fall inside `[0, in_f)`: pointers to the
/// input row and the tap weights. At most three.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dw_taps(
    src: &[f32],
    w: &[f32],
    co: usize,
    f: usize,
    stride: usize,
    pad: usize,
    in_f: usize,
) -> ([(*const f32, *const f32); 3], usize) {
    let mut taps = [(std::ptr::null(), std::ptr::null()); 3];
    let mut n = 0;
    for k in 0..3 {
        let fr = f * stride + k;
        if fr >= pad && fr - pad < in_f {
            taps[n] = (src[(fr - pad) * co..].as_ptr(), w[k * co..].as_ptr());
            n += 1;
        }
    }
    (taps, n)
}

/// Each 8-channel tile stays in a register across the taps, so `out` is written
/// once per position. Multiply then add, in the scalar order.
///
/// # Safety
/// The CPU must support AVX; `w` must hold `3*co` values, `b` `co`, `src`
/// `in_f*co`, and `out` `(of - 1)*spacing + offset + co` when `of > 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn depthwise_avx(
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    // SAFETY: each output position writes `out[f*spacing + offset..][..co]`, below
    // the caller's extent; `dw_taps` points only at rows `r < in_f` of `src` and
    // taps `k < 3` of `w`, each read for `co` values.
    unsafe {
        use std::arch::x86_64::*;
        let bp = b.as_ptr();
        for f in 0..of {
            let (taps, nt) = dw_taps(src, w, co, f, stride, pad, in_f);
            let dst = out.as_mut_ptr().add(f * spacing + offset);
            let mut c = 0;
            while c + 8 <= co {
                let mut acc = _mm256_loadu_ps(bp.add(c));
                for &(xr, wk) in &taps[..nt] {
                    acc = _mm256_add_ps(
                        acc,
                        _mm256_mul_ps(_mm256_loadu_ps(xr.add(c)), _mm256_loadu_ps(wk.add(c))),
                    );
                }
                _mm256_storeu_ps(dst.add(c), acc);
                c += 8;
            }
            while c < co {
                let mut s = *bp.add(c);
                for &(xr, wk) in &taps[..nt] {
                    s += *xr.add(c) * *wk.add(c);
                }
                *dst.add(c) = s;
                c += 1;
            }
        }
    }
}

/// [`depthwise_avx`] at 128-bit width.
///
/// # Safety
/// The CPU must support SSE4.1; `w` must hold `3*co` values, `b` `co`, `src`
/// `in_f*co`, and `out` `(of - 1)*spacing + offset + co` when `of > 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn depthwise_sse(
    out: &mut [f32],
    src: &[f32],
    w: &[f32],
    b: &[f32],
    co: usize,
    of: usize,
    in_f: usize,
    stride: usize,
    pad: usize,
    spacing: usize,
    offset: usize,
) {
    // SAFETY: each output position writes `out[f*spacing + offset..][..co]`, below
    // the caller's extent; `dw_taps` points only at rows `r < in_f` of `src` and
    // taps `k < 3` of `w`, each read for `co` values.
    unsafe {
        use std::arch::x86_64::*;
        let bp = b.as_ptr();
        for f in 0..of {
            let (taps, nt) = dw_taps(src, w, co, f, stride, pad, in_f);
            let dst = out.as_mut_ptr().add(f * spacing + offset);
            let mut c = 0;
            while c + 4 <= co {
                let mut acc = _mm_loadu_ps(bp.add(c));
                for &(xr, wk) in &taps[..nt] {
                    acc = _mm_add_ps(
                        acc,
                        _mm_mul_ps(_mm_loadu_ps(xr.add(c)), _mm_loadu_ps(wk.add(c))),
                    );
                }
                _mm_storeu_ps(dst.add(c), acc);
                c += 4;
            }
            while c < co {
                let mut s = *bp.add(c);
                for &(xr, wk) in &taps[..nt] {
                    s += *xr.add(c) * *wk.add(c);
                }
                *dst.add(c) = s;
                c += 1;
            }
        }
    }
}

/// Depthwise 1-D conv on the frequency axis, kernel 3, stride 1, pad 1.
/// Accumulates one temporal row (`w0,w1,w2` are that row's taps) into `out`.
pub fn dw_row_k3s1_accum(out: &mut [f32], src: &[f32], w0: f32, w1: f32, w2: f32) {
    let width = out.len();
    if width == 0 {
        return;
    }
    out[0] += w1 * src[0] + if width > 1 { w2 * src[1] } else { 0.0 };
    for ow in 1..width.saturating_sub(1) {
        out[ow] += w0 * src[ow - 1] + w1 * src[ow] + w2 * src[ow + 1];
    }
    if width > 1 {
        out[width - 1] += w0 * src[width - 2] + w1 * src[width - 1];
    }
}

/// Depthwise 1-D conv, kernel 3, stride 2, pad 1. `out.len() == src.len()/2`.
pub fn dw_row_k3s2_accum(out: &mut [f32], src: &[f32], w0: f32, w1: f32, w2: f32) {
    let w_in = src.len();
    for ow in 0..out.len() {
        let im = 2 * ow;
        let ir = 2 * ow + 1;
        let mut s = w1 * src[im];
        if ow > 0 {
            s += w0 * src[im - 1];
        }
        if ir < w_in {
            s += w2 * src[ir];
        }
        out[ow] += s;
    }
}

/// Pointwise (1x1) conv: `input[c_in][width]`, `weight[c_out][c_in]`,
/// `bias[c_out]` -> `out[c_out][width]`.
pub fn pointwise_conv2d(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        assert!(out.len() >= c_out.checked_mul(width).expect("pointwise output overflow"));
        assert!(input.len() >= c_in.checked_mul(width).expect("pointwise input overflow"));
        assert!(weight.len() >= c_out.checked_mul(c_in).expect("pointwise weights overflow"));
        assert!(bias.len() >= c_out);
        // SAFETY: tier 2 guarantees AVX; the asserts bound every access.
        unsafe { pointwise_conv2d_avx(out, input, weight, bias, c_in, c_out, width) };
        return;
    }
    pointwise_conv2d_scalar(out, input, weight, bias, c_in, c_out, width);
}

/// AVX body of [`pointwise_conv2d`], the largest single cost in DFN3. The scalar
/// loop rewrites `out` once per input channel; here up to 32 output columns stay
/// in registers across every input channel. Each lane still computes
/// `bias + w0*x0 + w1*x1 + ...` left to right with a separate multiply and add, so
/// the result is bit-identical to the scalar loop.
///
/// # Safety
/// The CPU must support AVX; `out` must hold `c_out*width` values, `input`
/// `c_in*width`, `weight` `c_out*c_in` and `bias` `c_out`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pointwise_conv2d_avx(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    // SAFETY: output row `co < c_out` spans `out[co*width..][..width]`, weight row
    // `weight[co*c_in..][..c_in]` and input row `ci < c_in` `input[ci*width..]`;
    // every column offset stays below `width`.
    unsafe {
        use std::arch::x86_64::*;
        let x = input.as_ptr();
        for co in 0..c_out {
            let wr = weight.as_ptr().add(co * c_in);
            let o = out.as_mut_ptr().add(co * width);
            let b = _mm256_set1_ps(bias[co]);
            let mut w0 = 0;
            while w0 + 32 <= width {
                let (mut a0, mut a1, mut a2, mut a3) = (b, b, b, b);
                for ci in 0..c_in {
                    let k = _mm256_set1_ps(*wr.add(ci));
                    let row = x.add(ci * width + w0);
                    a0 = _mm256_add_ps(a0, _mm256_mul_ps(k, _mm256_loadu_ps(row)));
                    a1 = _mm256_add_ps(a1, _mm256_mul_ps(k, _mm256_loadu_ps(row.add(8))));
                    a2 = _mm256_add_ps(a2, _mm256_mul_ps(k, _mm256_loadu_ps(row.add(16))));
                    a3 = _mm256_add_ps(a3, _mm256_mul_ps(k, _mm256_loadu_ps(row.add(24))));
                }
                _mm256_storeu_ps(o.add(w0), a0);
                _mm256_storeu_ps(o.add(w0 + 8), a1);
                _mm256_storeu_ps(o.add(w0 + 16), a2);
                _mm256_storeu_ps(o.add(w0 + 24), a3);
                w0 += 32;
            }
            while w0 + 8 <= width {
                let mut a = b;
                for ci in 0..c_in {
                    let k = _mm256_set1_ps(*wr.add(ci));
                    a = _mm256_add_ps(a, _mm256_mul_ps(k, _mm256_loadu_ps(x.add(ci * width + w0))));
                }
                _mm256_storeu_ps(o.add(w0), a);
                w0 += 8;
            }
            for w in w0..width {
                let mut s = bias[co];
                for ci in 0..c_in {
                    s += *wr.add(ci) * *x.add(ci * width + w);
                }
                *o.add(w) = s;
            }
        }
    }
}

/// Reference and non-AVX path of [`pointwise_conv2d`].
fn pointwise_conv2d_scalar(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    c_in: usize,
    c_out: usize,
    width: usize,
) {
    // co outer, ci inner: each output row stays hot in L1 while the small input
    // tile is re-read. The AVX body keeps exactly this summation order.
    for co in 0..c_out {
        let o = &mut out[co * width..co * width + width];
        o.fill(bias[co]);
        for ci in 0..c_in {
            let wt = weight[co * c_in + ci];
            let in_row = &input[ci * width..ci * width + width];
            for w in 0..width {
                o[w] += wt * in_row[w];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    #[test]
    fn depthwise_tiers_are_bit_identical_to_the_scalar_loop() {
        let mut r = Lcg(0xdeed_beef_0bad_c0de);
        // co covers 8-wide and 4-wide tiles with and without a remainder; spacing
        // and offset interleave the positions the way the DPDFNet encoder does.
        for (co, of, in_f, stride, pad, spacing, offset) in [
            (64usize, 40usize, 80usize, 1usize, 1usize, 64usize, 0usize),
            (64, 20, 40, 2, 1, 128, 64),
            (32, 13, 39, 3, 1, 32, 0),
            (12, 7, 7, 1, 0, 12, 0),
            (5, 9, 9, 1, 1, 7, 2),
        ] {
            let src = r.vec(in_f * co);
            let w = r.vec(3 * co);
            let b = r.vec(co);
            let len = (of - 1) * spacing + offset + co;
            let mut want = vec![9.0f32; len];
            depthwise_on(
                0, &mut want, &src, &w, &b, co, of, in_f, stride, pad, spacing, offset,
            );
            for tier in 1..=simd_tier() {
                let mut got = vec![9.0f32; len];
                depthwise_on(
                    tier, &mut got, &src, &w, &b, co, of, in_f, stride, pad, spacing, offset,
                );
                assert_eq!(got, want, "tier {tier} co={co} of={of} stride={stride}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "depthwise_1x3: output too short")]
    fn depthwise_rejects_a_short_output_before_dispatch() {
        let mut out = vec![0.0; 63];
        depthwise_1x3(
            &mut out, &[0.0; 64], &[0.0; 24], &[0.0; 8], 8, 8, 8, 1, 1, 8, 0,
        );
    }

    #[test]
    #[should_panic(expected = "depthwise_1x3: input too short")]
    fn depthwise_rejects_a_short_input_before_dispatch() {
        let mut out = vec![0.0; 64];
        depthwise_1x3(
            &mut out, &[0.0; 63], &[0.0; 24], &[0.0; 8], 8, 8, 8, 1, 1, 8, 0,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_pointwise_is_bit_identical_to_the_scalar_loop() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let mut r = Lcg(0x1234_5678);
        for (c_in, c_out, width) in [
            (64, 64, 96),
            (64, 64, 48),
            (10, 10, 96),
            (64, 64, 37),
            (3, 5, 7),
            (64, 64, 1),
        ] {
            let input = r.vec(c_in * width);
            let weight = r.vec(c_in * c_out);
            let bias = r.vec(c_out);
            let mut want = vec![0.0; c_out * width];
            let mut got = vec![0.0; c_out * width];
            pointwise_conv2d_scalar(&mut want, &input, &weight, &bias, c_in, c_out, width);
            // SAFETY: AVX was detected above.
            unsafe { pointwise_conv2d_avx(&mut got, &input, &weight, &bias, c_in, c_out, width) };
            let same = want
                .iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "shape {c_in}x{c_out}x{width}");
        }
    }
}
