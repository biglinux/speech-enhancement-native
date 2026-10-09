//! Layers both networks have in the same shape. Activations are `[CH][width]`.

use crate::{CH, NB_DF, NB_ERB};
use ops::{dw_row_k3s1_accum, pointwise_conv2d, relu_inplace};

/// `erb_conv0`: Conv2d(1, CH, 3x3) over the last three ERB feature frames, + ReLU.
/// `pad` holds the two previous frames.
pub fn erb_conv0(e0: &mut [f32], pad: &mut [f32], feat_erb: &[f32], w: &[f32], bias: &[f32]) {
    for co in 0..CH {
        let out_c = &mut e0[co * NB_ERB..co * NB_ERB + NB_ERB];
        out_c.fill(bias[co]);
        let wbase = co * 9;
        for kh in 0..3 {
            let src: &[f32] = if kh < 2 {
                &pad[kh * NB_ERB..kh * NB_ERB + NB_ERB]
            } else {
                feat_erb
            };
            let wk = &w[wbase + kh * 3..wbase + kh * 3 + 3];
            dw_row_k3s1_accum(out_c, src, wk[0], wk[1], wk[2]);
        }
        relu_inplace(out_c);
    }
    pad.copy_within(NB_ERB..2 * NB_ERB, 0);
    pad[NB_ERB..2 * NB_ERB].copy_from_slice(feat_erb);
}

/// `df_conv0`: a 3x3 conv in two groups over the last three complex feature frames
/// (real, imaginary), pointwise, ReLU. `pad` holds each group's two previous frames.
pub fn df_conv0(
    c0: &mut [f32],
    scratch: &mut [f32],
    pad: &mut [f32],
    feat_spec: &[f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
) {
    let cpg = CH / 2;
    for g in 0..2 {
        for co in 0..cpg {
            let co_abs = g * cpg + co;
            let out_c = &mut c0[co_abs * NB_DF..co_abs * NB_DF + NB_DF];
            out_c.fill(0.0);
            let wbase = co_abs * 9;
            for kh in 0..3 {
                let src: &[f32] = if kh < 2 {
                    &pad[g * 2 * NB_DF + kh * NB_DF..g * 2 * NB_DF + kh * NB_DF + NB_DF]
                } else {
                    &feat_spec[g * NB_DF..g * NB_DF + NB_DF]
                };
                let wk = &dw[wbase + kh * 3..wbase + kh * 3 + 3];
                dw_row_k3s1_accum(out_c, src, wk[0], wk[1], wk[2]);
            }
        }
    }
    for g in 0..2 {
        pad.copy_within(
            g * 2 * NB_DF + NB_DF..g * 2 * NB_DF + 2 * NB_DF,
            g * 2 * NB_DF,
        );
        let dst = g * 2 * NB_DF + NB_DF;
        pad[dst..dst + NB_DF].copy_from_slice(&feat_spec[g * NB_DF..g * NB_DF + NB_DF]);
    }
    pointwise_conv2d(scratch, c0, pw, pb, CH, CH, NB_DF);
    c0[..CH * NB_DF].copy_from_slice(&scratch[..CH * NB_DF]);
    relu_inplace(&mut c0[..CH * NB_DF]);
}

/// Per-channel 1x1 (scale + bias) + ReLU on `src`, then add `skip`, into `dst`.
pub fn conv_p_add(dst: &mut [f32], src: &[f32], skip: &[f32], w: &[f32], b: &[f32], width: usize) {
    for c in 0..CH {
        let wc = w[c];
        let bc = b[c];
        for f in 0..width {
            let mut v = src[c * width + f] * wc + bc;
            if v < 0.0 {
                v = 0.0;
            }
            dst[c * width + f] = v + skip[c * width + f];
        }
    }
}

/// Transposed depthwise conv, stride 2 (pads `[0,1,0,1]`, output_padding `[0,1]`),
/// reading `sio` at `w_in`, then pointwise + ReLU back into `sio` at `w_out`.
#[allow(clippy::too_many_arguments)]
pub fn convt_up(
    sio: &mut [f32],
    scr: &mut [f32],
    trans: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_in: usize,
    w_out: usize,
) {
    let w_trans = (w_in - 1) * 2 + 3;
    trans[..CH * w_trans].fill(0.0);
    for c in 0..CH {
        let wk = &dw[c * 3..c * 3 + 3];
        for i in 0..w_in {
            let v = sio[c * w_in + i];
            for k in 0..3 {
                trans[c * w_trans + i * 2 + k] += v * wk[k];
            }
        }
    }
    for c in 0..CH {
        for f in 0..w_out {
            scr[c * w_out + f] = trans[c * w_trans + f + 1];
        }
    }
    pointwise_conv2d(sio, scr, pw, pb, CH, CH, w_out);
    relu_inplace(&mut sio[..CH * w_out]);
}

const CONVP_TAPS: usize = 5;
const CONVP_OUT: usize = 10;

/// `df_convp`: a depthwise conv over the last five `c0` frames in two groups of
/// `CH / 2` channels to five outputs each, pointwise, ReLU. Its output joins the
/// deep-filter coefficients.
pub struct DfConvp {
    /// The four previous `c0` frames, a ring per channel starting at `head`.
    pad: Vec<f32>,
    head: usize,
    dw_out: Vec<f32>,
    pw_out: Vec<f32>,
}

impl Default for DfConvp {
    fn default() -> Self {
        Self::new()
    }
}

impl DfConvp {
    pub fn new() -> Self {
        Self {
            pad: vec![0.0; CH * (CONVP_TAPS - 1) * NB_DF],
            head: 0,
            dw_out: vec![0.0; CONVP_OUT * NB_DF],
            pw_out: vec![0.0; CONVP_OUT * NB_DF],
        }
    }

    /// Adds the layer's output on `c0` to `coefs` (`[NB_DF][CONVP_OUT]`).
    pub fn add_to(&mut self, coefs: &mut [f32], c0: &[f32], dw: &[f32], pw: &[f32], pb: &[f32]) {
        let cpg_in = CH / 2;
        let cpg_out = CONVP_OUT / 2;
        let kh = CONVP_TAPS;
        self.dw_out.fill(0.0);
        for g in 0..2 {
            for co in 0..cpg_out {
                let co_abs = g * cpg_out + co;
                let dst = &mut self.dw_out[co_abs * NB_DF..co_abs * NB_DF + NB_DF];
                #[cfg(target_arch = "x86_64")]
                if ops::simd_tier() >= 2 {
                    let taps = &dw[co_abs * cpg_in * kh..(co_abs + 1) * cpg_in * kh];
                    // SAFETY: tier 2 and up means AVX, the only requirement of this
                    // safe `#[target_feature]` function.
                    unsafe { convp_taps_avx(dst, &self.pad, c0, taps, g * cpg_in, self.head) };
                    continue;
                }
                for ci in 0..cpg_in {
                    let ci_abs = g * cpg_in + ci;
                    for k in 0..kh {
                        let src: &[f32] = if k < 4 {
                            let phys = (self.head + k) & 3;
                            &self.pad[ci_abs * 4 * NB_DF + phys * NB_DF
                                ..ci_abs * 4 * NB_DF + phys * NB_DF + NB_DF]
                        } else {
                            &c0[ci_abs * NB_DF..ci_abs * NB_DF + NB_DF]
                        };
                        let wval = dw[(co_abs * cpg_in + ci) * kh + k];
                        for f in 0..NB_DF {
                            dst[f] += src[f] * wval;
                        }
                    }
                }
            }
        }
        let old = self.head;
        for ci in 0..CH {
            let d = ci * 4 * NB_DF + old * NB_DF;
            self.pad[d..d + NB_DF].copy_from_slice(&c0[ci * NB_DF..ci * NB_DF + NB_DF]);
        }
        self.head = (old + 1) & 3;

        pointwise_conv2d(
            &mut self.pw_out,
            &self.dw_out,
            pw,
            pb,
            CONVP_OUT,
            CONVP_OUT,
            NB_DF,
        );
        relu_inplace(&mut self.pw_out);
        for ch in 0..CONVP_OUT {
            for f in 0..NB_DF {
                coefs[f * CONVP_OUT + ch] += self.pw_out[ch * NB_DF + f];
            }
        }
    }
}

/// AVX body of the `df_convp` taps for one output channel: `dst` (96 floats, 12
/// registers) stays in registers across all `cpg_in * 5` taps instead of being
/// loaded and stored per tap. Each lane adds `src * w` in the same (ci, k) order
/// with a separate multiply and add, so the sum is bit-identical to the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn convp_taps_avx(
    dst: &mut [f32],
    pad: &[f32],
    c0: &[f32],
    taps: &[f32],
    ci_base: usize,
    head: usize,
) {
    use std::arch::x86_64::*;
    const LANES: usize = NB_DF / 8;
    let cpg_in = CH / 2;
    assert!(dst.len() == NB_DF && taps.len() == cpg_in * CONVP_TAPS);
    let mut acc = [_mm256_setzero_ps(); LANES];
    for (l, a) in acc.iter_mut().enumerate() {
        // SAFETY: `l * 8 + 8 <= NB_DF`, the asserted length of `dst`.
        *a = unsafe { _mm256_loadu_ps(dst.as_ptr().add(l * 8)) };
    }
    for ci in 0..cpg_in {
        let ci_abs = ci_base + ci;
        for k in 0..CONVP_TAPS {
            let src: &[f32] = if k < 4 {
                let at = ci_abs * 4 * NB_DF + ((head + k) & 3) * NB_DF;
                &pad[at..at + NB_DF]
            } else {
                &c0[ci_abs * NB_DF..ci_abs * NB_DF + NB_DF]
            };
            let w = _mm256_set1_ps(taps[ci * CONVP_TAPS + k]);
            for (l, a) in acc.iter_mut().enumerate() {
                // SAFETY: `src` holds `NB_DF` values and `l * 8 + 8 <= NB_DF`.
                let x = unsafe { _mm256_loadu_ps(src.as_ptr().add(l * 8)) };
                *a = _mm256_add_ps(*a, _mm256_mul_ps(x, w));
            }
        }
    }
    for (l, a) in acc.iter().enumerate() {
        // SAFETY: `l * 8 + 8 <= NB_DF`, the asserted length of `dst`.
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add(l * 8), *a) };
    }
}
