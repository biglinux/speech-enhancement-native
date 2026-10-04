//! DeepFilterNet3-LL (low-latency) streaming denoiser — independent Rust port.
//!
//! Same DeepFilterNet3 pipeline as the standard model, with the LL hyper-params:
//! emb/GRU hidden 512, 3 DF GRU layers, grouped linears at 16/8 groups, separable
//! convs with temporal kernel 2 (one past frame), transposed convs depthwise, and
//! zero look-ahead (df_lookahead = conv_lookahead = 0). Weights are traced from the
//! upstream LL ONNX; this is our own code. Validated stage-by-stage against ORT.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design

mod ladspa;
mod weights;

use dfn_ops::gru_cell_packed as active_gru;
use dfn_ops::*;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
use weights::W;

/// Test hook: the LADSPA descriptor as a raw pointer (its layout is the LADSPA 1.1
/// ABI). Lets the integration test reach the C entry points via the rlib.
#[doc(hidden)]
pub fn ladspa_descriptor_ptr(index: usize) -> *const std::ffi::c_void {
    ladspa::ladspa_descriptor(index as std::os::raw::c_ulong) as *const std::ffi::c_void
}

/// A decoded, shareable weight table. Decode once with [`Weights::load`] and build
/// many [`Dfn3Ll`] instances from it via [`Dfn3Ll::with_weights`] to avoid
/// re-decoding the weight blob per instance (used by the LADSPA layer and fuzzing).
pub struct Weights(Arc<W>);

/// Shared dB→noisy-mix conversion, re-exported so the CLIs and tests attenuate
/// exactly as the LADSPA plugin does.
pub use dfn_ops::atten_lim_from_db;

impl Weights {
    pub fn load(bytes: &[u8]) -> Self {
        let w = W::load(bytes);
        // Pin the weights in RAM so realtime processing never faults on them
        // under memory pressure (a page fault in the audio callback is an xrun).
        w.mlock();
        Weights(Arc::new(w))
    }
}

pub const SR: usize = 48000;
pub const FFT: usize = 960;
pub const HOP: usize = 480;
pub const FREQ: usize = 481;
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;
pub const CH: usize = 64;
pub const EMB: usize = 512;
pub const HID: usize = 512;

const LSNR_MAX: f32 = 35.0;
const LSNR_MIN: f32 = -15.0;
const SILENCE_RMS: f32 = 1e-7;
const SILENCE_SKIP_MAX: i32 = 5;
const CONTEXT_FRAMES: u16 = (SR / HOP) as u16;

// LL: conv_lookahead = df_lookahead = 0.
const CONV_LOOKAHEAD: usize = 0;
const SPEC_Y: usize = DF_ORDER + CONV_LOOKAHEAD; // 5
const SPEC_X: usize = DF_ORDER; // 5
const LOOKAHEAD: usize = 0;

pub const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37,
    42, 50, 56, 67,
];
const ERB_DF_SKIP_BAND: usize = 21;
const ERB_DF_SKIP_OFFSET: usize = 93;

pub struct Dfn3Ll {
    w: Arc<W>,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    fft_r: Vec<f32>,
    fft_c: Vec<Complex32>,
    r2c_scratch: Vec<Complex32>,
    c2r_scratch: Vec<Complex32>,

    window: Vec<f32>,
    wnorm: f32,
    analysis_mem: Vec<f32>,
    synthesis_mem: Vec<f32>,

    spec_re: Vec<f32>,
    spec_im: Vec<f32>,
    roll_y: Vec<[f32; 2 * FREQ]>,
    roll_x: Vec<[f32; 2 * FREQ]>,
    y_head: usize,
    x_head: usize,

    erb_norm: Vec<f32>,
    unit_norm: Vec<f32>,
    alpha: f32,
    pub feat_erb: Vec<f32>,
    pub feat_spec: Vec<f32>,

    // conv temporal pads
    erb0_pad: Vec<f32>,     // erb_conv0 k3 -> 2 past [2,32]
    df0_pad: Vec<f32>,      // df_conv0 k3 g2 -> 2 past [2,2,96]
    erb1_pad: Vec<f32>,     // erb_conv1 k2 -> 1 past of e0 [64,32]
    erb2_pad: Vec<f32>,     // e1 [64,16]
    erb3_pad: Vec<f32>,     // e2 [64,8]
    df1_pad: Vec<f32>,      // c0 [64,96]
    convt3_pad: Vec<f32>,   // erb_dec convt3 k2 -> 1 past of its input [64,8]
    conv0out_pad: Vec<f32>, // conv0_out k2 -> 1 past of its input [64,32]

    pub e0: Vec<f32>,
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub e3: Vec<f32>,
    pub c0: Vec<f32>,
    pub emb: Vec<f32>,
    pub lsnr: f32,
    /// Per-ERB level (dB, 10·log10 mean bin power) of the *processed* output
    /// spectrum this hop — the same signal the GUI's spectrum graph shows. Read
    /// by the plugin to compare against a user-drawn per-frequency silence floor.
    /// Set to a deep floor on skipped/muted hops.
    pub band_db: [f32; NB_ERB],

    enc_h: Vec<f32>,
    erb0_h: Vec<f32>,
    erb1_h: Vec<f32>,
    df0_h: Vec<f32>,
    df1_h: Vec<f32>,
    df2_h: Vec<f32>,
    gru_scratch: Vec<f32>,
    i16q: Vec<i16>,

    pub erb_mask: Vec<f32>,
    convp_pad: Vec<f32>,
    convp_head: usize,
    coefs_t: Vec<f32>,
    pub coefs_dbg: Vec<f32>,

    s1: Vec<f32>,
    s2: Vec<f32>,
    // preallocated per-frame scratch (RT hygiene: no heap in process()).
    sc_flat: Vec<f32>, // 3072
    sc_512a: Vec<f32>,
    sc_512b: Vec<f32>,
    sc_512c: Vec<f32>,
    sc_960a: Vec<f32>,
    sc_960b: Vec<f32>,
    sc_960c: Vec<f32>,
    sc_960d: Vec<f32>,
    sc_trans: Vec<f32>, // CH*33
    enh_re: Vec<f32>,
    enh_im: Vec<f32>,
    silence: i32,
    speech_frames: u16,
    speech_lsnr_sum: f32,

    pub atten_lim: f32,
    pub post_filter_beta: f32,
    pub min_db: f32,
    pub max_db_erb: f32,
    pub max_db_df: f32,
}

impl Dfn3Ll {
    pub fn new(weights_bytes: &[u8]) -> Self {
        Self::from_shared(Arc::new(W::load(weights_bytes)))
    }

    /// Build an instance from an already-decoded shared weight table.
    pub fn with_weights(w: &Weights) -> Self {
        Self::from_shared(w.0.clone())
    }

    /// Build an instance sharing an already-loaded weight table, so multiple
    /// instances (and `activate()` resets) don't each re-copy the weight blob.
    pub(crate) fn from_shared(w: Arc<W>) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FFT);
        let c2r = planner.plan_fft_inverse(FFT);
        let fft_r = r2c.make_input_vec();
        let fft_c = r2c.make_output_vec();
        let r2c_scratch = r2c.make_scratch_vec();
        let c2r_scratch = c2r.make_scratch_vec();

        let wsh = FFT / 2;
        let mut window = vec![0.0f32; FFT];
        for (n, wv) in window.iter_mut().enumerate() {
            let s = (std::f64::consts::PI / 2.0 * (n as f64 + 0.5) / wsh as f64).sin();
            *wv = (std::f64::consts::PI / 2.0 * s * s).sin() as f32;
        }
        let wnorm = 2.0 * HOP as f32 / (FFT as f32 * FFT as f32);

        let mut erb_norm = vec![0.0f32; NB_ERB];
        for (i, v) in erb_norm.iter_mut().enumerate() {
            *v = -60.0 + i as f32 * (-90.0 - (-60.0)) / (NB_ERB - 1) as f32;
        }
        let mut unit_norm = vec![0.0f32; NB_DF];
        for (i, v) in unit_norm.iter_mut().enumerate() {
            *v = 0.001 + i as f32 * (0.0001 - 0.001) / (NB_DF - 1) as f32;
        }
        let alpha = (-(HOP as f32) / (SR as f32 * 1.0)).exp();

        Dfn3Ll {
            w,
            r2c,
            c2r,
            fft_r,
            fft_c,
            r2c_scratch,
            c2r_scratch,
            window,
            wnorm,
            analysis_mem: vec![0.0; HOP],
            synthesis_mem: vec![0.0; HOP],
            spec_re: vec![0.0; FREQ],
            spec_im: vec![0.0; FREQ],
            roll_y: vec![[0.0; 2 * FREQ]; SPEC_Y],
            roll_x: vec![[0.0; 2 * FREQ]; SPEC_X],
            y_head: 0,
            x_head: 0,
            erb_norm,
            unit_norm,
            alpha,
            feat_erb: vec![0.0; NB_ERB],
            feat_spec: vec![0.0; 2 * NB_DF],
            erb0_pad: vec![0.0; 2 * NB_ERB],
            df0_pad: vec![0.0; 2 * 2 * NB_DF],
            erb1_pad: vec![0.0; CH * NB_ERB],
            erb2_pad: vec![0.0; CH * (NB_ERB / 2)],
            erb3_pad: vec![0.0; CH * (NB_ERB / 4)],
            df1_pad: vec![0.0; CH * NB_DF],
            convt3_pad: vec![0.0; CH * (NB_ERB / 4)],
            conv0out_pad: vec![0.0; CH * NB_ERB],
            e0: vec![0.0; CH * NB_ERB],
            e1: vec![0.0; CH * (NB_ERB / 2)],
            e2: vec![0.0; CH * (NB_ERB / 4)],
            e3: vec![0.0; CH * (NB_ERB / 4)],
            c0: vec![0.0; CH * NB_DF],
            emb: vec![0.0; EMB],
            lsnr: 0.0,
            band_db: [-200.0; NB_ERB],
            enc_h: vec![0.0; HID],
            erb0_h: vec![0.0; HID],
            erb1_h: vec![0.0; HID],
            df0_h: vec![0.0; HID],
            df1_h: vec![0.0; HID],
            df2_h: vec![0.0; HID],
            gru_scratch: vec![0.0; 6 * HID],
            i16q: vec![0i16; HID],
            erb_mask: vec![0.0; NB_ERB],
            convp_pad: vec![0.0; CH * 4 * NB_DF],
            convp_head: 0,
            coefs_t: vec![0.0; DF_ORDER * 2 * NB_DF],
            coefs_dbg: vec![0.0; NB_DF * 10],
            // Largest live scratch planes: DF input (64x96) and strided DF (64x48).
            s1: vec![0.0; CH * NB_DF],
            s2: vec![0.0; CH * (NB_DF / 2)],
            sc_flat: vec![0.0; 48 * CH],
            sc_512a: vec![0.0; EMB],
            sc_512b: vec![0.0; EMB],
            sc_512c: vec![0.0; EMB],
            sc_960a: vec![0.0; NB_DF * 10],
            sc_960b: vec![0.0; NB_DF * 10],
            sc_960c: vec![0.0; NB_DF * 10],
            sc_960d: vec![0.0; NB_DF * 10],
            sc_trans: vec![0.0; CH * 33],
            enh_re: vec![0.0; FREQ],
            enh_im: vec![0.0; FREQ],
            silence: 0,
            speech_frames: 0,
            speech_lsnr_sum: 0.0,
            atten_lim: 0.0,
            post_filter_beta: 0.0,
            // Low-CPU defaults (upstream DFN): below min_db a frame is gated to
            // silence, above max_db_erb both decoders are skipped, above max_db_df
            // only the DF stage is skipped. The LADSPA layer exposes these as
            // control ports so they can be overridden in the filter-chain config.
            min_db: -10.0,
            max_db_erb: 30.0,
            max_db_df: 20.0,
        }
    }

    #[inline]
    fn yphys(&self, logical: usize) -> usize {
        (self.y_head + logical) % SPEC_Y
    }
    #[inline]
    fn xphys(&self, logical: usize) -> usize {
        (self.x_head + logical) % SPEC_X
    }

    fn frame_analysis(&mut self, input: &[f32]) {
        for i in 0..HOP {
            self.fft_r[i] = self.analysis_mem[i] * self.window[i];
            self.fft_r[HOP + i] = input[i] * self.window[HOP + i];
        }
        self.analysis_mem[..HOP].copy_from_slice(&input[..HOP]);
        self.r2c
            .process_with_scratch(&mut self.fft_r, &mut self.fft_c, &mut self.r2c_scratch)
            .expect("rfft");
        for i in 0..FREQ {
            self.spec_re[i] = self.fft_c[i].re * self.wnorm;
            self.spec_im[i] = self.fft_c[i].im * self.wnorm;
        }
    }

    fn frame_synthesis(&mut self, out: &mut [f32], enh_re: &[f32], enh_im: &[f32]) {
        for i in 0..FREQ {
            self.fft_c[i] = Complex32::new(enh_re[i], enh_im[i]);
        }
        self.fft_c[0].im = 0.0;
        self.fft_c[FREQ - 1].im = 0.0;
        self.c2r
            .process_with_scratch(&mut self.fft_c, &mut self.fft_r, &mut self.c2r_scratch)
            .expect("irfft");
        for i in 0..FFT {
            self.fft_r[i] *= self.window[i];
        }
        for i in 0..HOP {
            out[i] = self.fft_r[i] + self.synthesis_mem[i];
        }
        self.synthesis_mem[..HOP].copy_from_slice(&self.fft_r[HOP..FFT]);
    }

    fn extract_erb(&mut self) {
        let a = self.alpha;
        let mut off = 0;
        for b in 0..NB_ERB {
            let width = ERB_WIDTHS[b];
            let mut energy = 0.0f32;
            for i in off..off + width {
                energy += self.spec_re[i] * self.spec_re[i] + self.spec_im[i] * self.spec_im[i];
            }
            energy /= width as f32;
            let db = 10.0 * dfn_ops::log10_f(energy + 1e-10);
            self.erb_norm[b] = db * (1.0 - a) + self.erb_norm[b] * a;
            self.feat_erb[b] = (db - self.erb_norm[b]) / 40.0;
            off += width;
        }
    }

    fn extract_spec(&mut self) {
        let a = self.alpha;
        for i in 0..NB_DF {
            let mag =
                (self.spec_re[i] * self.spec_re[i] + self.spec_im[i] * self.spec_im[i]).sqrt();
            self.unit_norm[i] = mag * (1.0 - a) + self.unit_norm[i] * a;
            let inv = 1.0 / self.unit_norm[i].sqrt();
            self.feat_spec[i] = self.spec_re[i] * inv;
            self.feat_spec[NB_DF + i] = self.spec_im[i] * inv;
        }
    }

    fn encoder(&mut self) {
        // erb_conv0: Conv2d(1,64,3,3) causal 2-past + ReLU -> e0
        {
            let wt = self.w.enc_erb_conv0_dw_w();
            let bias = self.w.enc_erb_conv0_dw_b();
            for co in 0..CH {
                let out_c = &mut self.e0[co * NB_ERB..co * NB_ERB + NB_ERB];
                out_c.fill(bias[co]);
                let wbase = co * 9;
                for kh in 0..3 {
                    let src: &[f32] = if kh < 2 {
                        &self.erb0_pad[kh * NB_ERB..kh * NB_ERB + NB_ERB]
                    } else {
                        &self.feat_erb
                    };
                    let wk = &wt[wbase + kh * 3..wbase + kh * 3 + 3];
                    dw_row_k3s1_accum(out_c, src, wk[0], wk[1], wk[2]);
                }
                relu_inplace(out_c);
            }
            self.erb0_pad.copy_within(NB_ERB..2 * NB_ERB, 0);
            self.erb0_pad[NB_ERB..2 * NB_ERB].copy_from_slice(&self.feat_erb);
        }
        // erb_conv1/2 (k2 temporal, dw stride2 + pw + relu), erb_conv3 (k2, stride1)
        sep_k2_s2(
            &mut self.e1,
            &mut self.s1,
            &self.e0,
            &mut self.erb1_pad,
            self.w.enc_erb_conv1_dw_w(),
            self.w.enc_erb_conv1_pw_w(),
            self.w.enc_erb_conv1_pw_b(),
            NB_ERB,
            NB_ERB / 2,
        );
        sep_k2_s2(
            &mut self.e2,
            &mut self.s1,
            &self.e1,
            &mut self.erb2_pad,
            self.w.enc_erb_conv2_dw_w(),
            self.w.enc_erb_conv2_pw_w(),
            self.w.enc_erb_conv2_pw_b(),
            NB_ERB / 2,
            NB_ERB / 4,
        );
        sep_k2_s1(
            &mut self.e3,
            &mut self.s1,
            &self.e2,
            &mut self.erb3_pad,
            self.w.enc_erb_conv3_dw_w(),
            self.w.enc_erb_conv3_pw_w(),
            self.w.enc_erb_conv3_pw_b(),
            NB_ERB / 4,
        );

        // df_conv0: groups=2 causal 2-past + pw + relu -> c0
        {
            let wt = self.w.enc_df_conv0_dw_w();
            let cpg = CH / 2;
            for g in 0..2 {
                for co in 0..cpg {
                    let co_abs = g * cpg + co;
                    let out_c = &mut self.c0[co_abs * NB_DF..co_abs * NB_DF + NB_DF];
                    out_c.fill(0.0);
                    let wbase = co_abs * 9;
                    for kh in 0..3 {
                        let src: &[f32] = if kh < 2 {
                            &self.df0_pad
                                [g * 2 * NB_DF + kh * NB_DF..g * 2 * NB_DF + kh * NB_DF + NB_DF]
                        } else {
                            &self.feat_spec[g * NB_DF..g * NB_DF + NB_DF]
                        };
                        let wk = &wt[wbase + kh * 3..wbase + kh * 3 + 3];
                        dw_row_k3s1_accum(out_c, src, wk[0], wk[1], wk[2]);
                    }
                }
            }
            for g in 0..2 {
                self.df0_pad.copy_within(
                    g * 2 * NB_DF + NB_DF..g * 2 * NB_DF + 2 * NB_DF,
                    g * 2 * NB_DF,
                );
                let dst = g * 2 * NB_DF + NB_DF;
                self.df0_pad[dst..dst + NB_DF]
                    .copy_from_slice(&self.feat_spec[g * NB_DF..g * NB_DF + NB_DF]);
            }
            pointwise_conv2d(
                &mut self.s1,
                &self.c0,
                self.w.enc_df_conv0_pw_w(),
                self.w.enc_df_conv0_pw_b(),
                CH,
                CH,
                NB_DF,
            );
            self.c0[..CH * NB_DF].copy_from_slice(&self.s1[..CH * NB_DF]);
            relu_inplace(&mut self.c0[..CH * NB_DF]);
        }
        // df_conv1: k2 sep stride2 + pw + relu -> s2 [64,48]
        {
            let wout = NB_DF / 2;
            sep_k2_s2(
                &mut self.s2,
                &mut self.s1,
                &self.c0,
                &mut self.df1_pad,
                self.w.enc_df_conv1_dw_w(),
                self.w.enc_df_conv1_pw_w(),
                self.w.enc_df_conv1_pw_b(),
                NB_DF,
                wout,
            );
        }

        // embedding: df path transpose+einsum -> cemb; e3 flat -> erb_flat; emb = erb_flat + cemb
        {
            for ch in 0..CH {
                for f in 0..48 {
                    self.sc_flat[f * CH + ch] = self.s2[ch * 48 + f];
                }
            }
            grouped_linear(
                &mut self.sc_512a,
                &self.sc_flat,
                self.w.enc_df_fc_emb_w(),
                16,
                192,
                32,
            );
            relu_inplace(&mut self.sc_512a);
            for ch in 0..CH {
                for f in 0..8 {
                    self.emb[f * CH + ch] = self.e3[ch * 8 + f];
                }
            }
            vadd(&mut self.emb, &self.sc_512a);
        }

        // encoder squeezed GRU
        {
            grouped_linear(
                &mut self.sc_512b,
                &self.emb,
                self.w.enc_emb_gru_lin_in_w(),
                16,
                32,
                32,
            );
            relu_inplace(&mut self.sc_512b);
            active_gru(
                &mut self.enc_h,
                &self.sc_512b,
                self.w.enc_emb_gru_W_q(),
                self.w.enc_emb_gru_W_s(),
                self.w.enc_emb_gru_R_q(),
                self.w.enc_emb_gru_R_s(),
                self.w.enc_emb_gru_B(),
                HID,
                HID,
                &mut self.gru_scratch,
                &mut self.i16q,
            );
            grouped_linear(
                &mut self.sc_512c,
                &self.enc_h,
                self.w.enc_emb_gru_lin_out_w(),
                16,
                32,
                32,
            );
            relu_inplace(&mut self.sc_512c);
            self.emb.copy_from_slice(&self.sc_512c);
        }

        let val = self.w.enc_lsnr_fc_b()[0] + vdot(&self.emb, self.w.enc_lsnr_fc_w());
        self.lsnr = sigmoid(val) * (LSNR_MAX - LSNR_MIN) + LSNR_MIN;
    }

    fn erb_decoder(&mut self) {
        grouped_linear(
            &mut self.sc_512b,
            &self.emb,
            self.w.erb_emb_gru_lin_in_w(),
            16,
            32,
            32,
        );
        relu_inplace(&mut self.sc_512b);
        active_gru(
            &mut self.erb0_h,
            &self.sc_512b,
            self.w.erb_emb_gru0_W_q(),
            self.w.erb_emb_gru0_W_s(),
            self.w.erb_emb_gru0_R_q(),
            self.w.erb_emb_gru0_R_s(),
            self.w.erb_emb_gru0_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        active_gru(
            &mut self.erb1_h,
            &self.erb0_h,
            self.w.erb_emb_gru1_W_q(),
            self.w.erb_emb_gru1_W_s(),
            self.w.erb_emb_gru1_R_q(),
            self.w.erb_emb_gru1_R_s(),
            self.w.erb_emb_gru1_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        grouped_linear(
            &mut self.sc_512a,
            &self.erb1_h,
            self.w.erb_emb_gru_lin_out_w(),
            16,
            32,
            32,
        );
        relu_inplace(&mut self.sc_512a);
        for f in 0..8 {
            for ch in 0..CH {
                self.sc_512c[ch * 8 + f] = self.sc_512a[f * CH + ch];
            }
        }
        // C3p: s1 = relu(conv3p(e3)) + emb2d (sc_512c)
        conv_p_add(
            &mut self.s1,
            &self.e3,
            &self.sc_512c[..CH * 8],
            self.w.erb_conv3p_dw_w(),
            self.w.erb_conv3p_dw_b(),
            8,
        );
        // Ct3: k2 sep dw(s1, past) -> s2, pw -> s1
        convt3_k2(
            &mut self.s1,
            &mut self.s2,
            &mut self.convt3_pad,
            self.w.erb_convt3_dw_w(),
            self.w.erb_convt3_pw_w(),
            self.w.erb_convt3_pw_b(),
            8,
        );
        // C2p: s2 = relu(conv2p(e2)) + s1
        {
            let (dst, skip) = split_two(&mut self.s2, &self.s1, CH * 8);
            conv_p_add(
                dst,
                &self.e2,
                skip,
                self.w.erb_conv2p_dw_w(),
                self.w.erb_conv2p_dw_b(),
                8,
            );
        }
        // Ct2: transpose s2 8->16
        convt_up(
            &mut self.s2,
            &mut self.s1,
            &mut self.sc_trans,
            self.w.erb_convt2_dw_w(),
            self.w.erb_convt2_pw_w(),
            self.w.erb_convt2_pw_b(),
            8,
            16,
        );
        // C1p: s1 = relu(conv1p(e1)) + s2
        {
            let (dst, skip) = split_two(&mut self.s1, &self.s2, CH * 16);
            conv_p_add(
                dst,
                &self.e1,
                skip,
                self.w.erb_conv1p_dw_w(),
                self.w.erb_conv1p_dw_b(),
                16,
            );
        }
        // Ct1: transpose s1 16->32
        convt_up(
            &mut self.s1,
            &mut self.s2,
            &mut self.sc_trans,
            self.w.erb_convt1_dw_w(),
            self.w.erb_convt1_pw_w(),
            self.w.erb_convt1_pw_b(),
            16,
            32,
        );
        // C0p: s2 = relu(conv0p(e0)) + s1
        {
            let (dst, skip) = split_two(&mut self.s2, &self.s1, CH * NB_ERB);
            conv_p_add(
                dst,
                &self.e0,
                skip,
                self.w.erb_conv0p_dw_w(),
                self.w.erb_conv0p_dw_b(),
                NB_ERB,
            );
        }
        // conv0_out: k2 temporal [1,64,2,3] group1 + sigmoid -> erb_mask
        {
            let wt = self.w.erb_conv0_out_w();
            let b = self.w.erb_conv0_out_b()[0];
            self.erb_mask.fill(b);
            // kh=0 past, kh=1 current: weight [1,64,2,3] = per (ci, kh, kw)
            for ci in 0..CH {
                // wt index: ci occupies 2*3=6 per channel; layout [C_out=1, C_in=64, 2, 3]
                let base = ci * 6;
                let past = &self.conv0out_pad[ci * NB_ERB..ci * NB_ERB + NB_ERB];
                let cur = &self.s2[ci * NB_ERB..ci * NB_ERB + NB_ERB];
                dw_row_k3s1_accum(
                    &mut self.erb_mask,
                    past,
                    wt[base],
                    wt[base + 1],
                    wt[base + 2],
                );
                dw_row_k3s1_accum(
                    &mut self.erb_mask,
                    cur,
                    wt[base + 3],
                    wt[base + 4],
                    wt[base + 5],
                );
            }
            for v in self.erb_mask.iter_mut() {
                *v = sigmoid(*v);
            }
            self.conv0out_pad.copy_from_slice(&self.s2[..CH * NB_ERB]);
        }
    }

    fn df_decoder(&mut self) {
        grouped_linear(
            &mut self.sc_512b,
            &self.emb,
            self.w.df_gru_lin_in_w(),
            8,
            64,
            64,
        );
        relu_inplace(&mut self.sc_512b);
        active_gru(
            &mut self.df0_h,
            &self.sc_512b,
            self.w.df_gru0_W_q(),
            self.w.df_gru0_W_s(),
            self.w.df_gru0_R_q(),
            self.w.df_gru0_R_s(),
            self.w.df_gru0_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        active_gru(
            &mut self.df1_h,
            &self.df0_h,
            self.w.df_gru1_W_q(),
            self.w.df_gru1_W_s(),
            self.w.df_gru1_R_q(),
            self.w.df_gru1_R_s(),
            self.w.df_gru1_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        active_gru(
            &mut self.df2_h,
            &self.df1_h,
            self.w.df_gru2_W_q(),
            self.w.df_gru2_W_s(),
            self.w.df_gru2_R_q(),
            self.w.df_gru2_R_s(),
            self.w.df_gru2_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );

        // df_comb (sc_512a) = df2_h + df_skip(emb)
        self.sc_512a.copy_from_slice(&self.df2_h);
        grouped_linear(&mut self.sc_512c, &self.emb, self.w.df_skip_w(), 16, 32, 32);
        vadd(&mut self.sc_512a, &self.sc_512c);
        // coefs (sc_960a)
        grouped_linear(
            &mut self.sc_960a,
            &self.sc_512a,
            self.w.df_out_w(),
            16,
            32,
            60,
        );
        for v in self.sc_960a.iter_mut() {
            *v = dfn_ops::tanh_f(*v);
        }

        // df_convp: dw (5,1) groups=2 on c0, pw, relu, transpose -> c0_proj (sc_960b)
        {
            let cpg_in = CH / 2;
            let cpg_out = 5;
            let kh = 5;
            let wt = self.w.df_convp_dw_w();
            self.sc_960c.fill(0.0); // dw_out
            for g in 0..2 {
                for co in 0..cpg_out {
                    let co_abs = g * cpg_out + co;
                    let dst = &mut self.sc_960c[co_abs * NB_DF..co_abs * NB_DF + NB_DF];
                    for ci in 0..cpg_in {
                        let ci_abs = g * cpg_in + ci;
                        for k in 0..kh {
                            let src: &[f32] = if k < 4 {
                                let phys = (self.convp_head + k) & 3;
                                &self.convp_pad[ci_abs * 4 * NB_DF + phys * NB_DF
                                    ..ci_abs * 4 * NB_DF + phys * NB_DF + NB_DF]
                            } else {
                                &self.c0[ci_abs * NB_DF..ci_abs * NB_DF + NB_DF]
                            };
                            let wval = wt[(co_abs * cpg_in + ci) * kh + k];
                            for f in 0..NB_DF {
                                dst[f] += src[f] * wval;
                            }
                        }
                    }
                }
            }
            let old = self.convp_head;
            for ci in 0..CH {
                let d = ci * 4 * NB_DF + old * NB_DF;
                self.convp_pad[d..d + NB_DF]
                    .copy_from_slice(&self.c0[ci * NB_DF..ci * NB_DF + NB_DF]);
            }
            self.convp_head = (old + 1) & 3;

            pointwise_conv2d(
                &mut self.sc_960d,
                &self.sc_960c,
                self.w.df_convp_pw_w(),
                self.w.df_convp_pw_b(),
                10,
                10,
                NB_DF,
            );
            relu_inplace(&mut self.sc_960d[..10 * NB_DF]);
            for ch in 0..10 {
                for f in 0..NB_DF {
                    self.sc_960b[f * 10 + ch] = self.sc_960d[ch * NB_DF + f];
                }
            }
        }

        // coefs (sc_960a) += c0_proj (sc_960b)
        for i in 0..NB_DF * 10 {
            self.sc_960a[i] += self.sc_960b[i];
        }
        self.coefs_dbg.copy_from_slice(&self.sc_960a[..NB_DF * 10]);
        for n in 0..DF_ORDER {
            for f in 0..NB_DF {
                self.coefs_t[n * 2 * NB_DF + f] = self.sc_960a[f * 10 + n * 2];
                self.coefs_t[n * 2 * NB_DF + NB_DF + f] = self.sc_960a[f * 10 + n * 2 + 1];
            }
        }
    }

    fn apply_df(&self, enh_re: &mut [f32], enh_im: &mut [f32]) {
        enh_re[..NB_DF].fill(0.0);
        enh_im[..NB_DF].fill(0.0);
        for n in 0..DF_ORDER {
            let phys = self.xphys(n);
            let frame = &self.roll_x[phys];
            let cre = &self.coefs_t[n * 2 * NB_DF..n * 2 * NB_DF + NB_DF];
            let cim = &self.coefs_t[n * 2 * NB_DF + NB_DF..n * 2 * NB_DF + 2 * NB_DF];
            for f in 0..NB_DF {
                let sr = frame[f];
                let si = frame[FREQ + f];
                enh_re[f] += cre[f] * sr - cim[f] * si;
                enh_im[f] += cre[f] * si + cim[f] * sr;
            }
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut [f32]) {
        let e: f32 = input[..HOP].iter().map(|v| v * v).sum();
        if e / (HOP as f32) < SILENCE_RMS {
            self.silence += 1;
        } else {
            self.silence = 0;
        }
        if self.silence > SILENCE_SKIP_MAX {
            if self.speech_frames < CONTEXT_FRAMES {
                self.speech_frames = 0;
                self.speech_lsnr_sum = 0.0;
            }
            out[..HOP].fill(0.0);
            self.lsnr = LSNR_MIN;
            self.band_db = [-200.0; NB_ERB];
            return;
        }

        self.frame_analysis(input);
        {
            let d = &mut self.roll_y[self.y_head];
            d[..FREQ].copy_from_slice(&self.spec_re);
            d[FREQ..].copy_from_slice(&self.spec_im);
            self.y_head = (self.y_head + 1) % SPEC_Y;
            let d = &mut self.roll_x[self.x_head];
            d[..FREQ].copy_from_slice(&self.spec_re);
            d[FREQ..].copy_from_slice(&self.spec_im);
            self.x_head = (self.x_head + 1) % SPEC_X;
        }

        self.extract_erb();
        self.extract_spec();
        self.encoder();
        // Confirm a full second of accepted audio with speech-dominant mean
        // SNR. Averaging permits short unvoiced sounds inside real speech;
        // merely keeping the gate open also admitted sustained background noise.
        // Once armed, the context stays warm through pauses without any mute.
        if self.speech_frames < CONTEXT_FRAMES {
            if self.lsnr >= self.min_db {
                self.speech_frames += 1;
                self.speech_lsnr_sum += self.lsnr;
                if self.speech_frames == CONTEXT_FRAMES && self.speech_lsnr_sum <= 0.0 {
                    self.speech_frames = 0;
                    self.speech_lsnr_sum = 0.0;
                }
            } else {
                self.speech_frames = 0;
                self.speech_lsnr_sum = 0.0;
            }
        }
        // The encoder GRU state is kept on every processed hop — a continuous
        // recurrence. The prior signal-dependent rollback to the old state when
        // `lsnr < min_db` discarded gated-frame evidence an onset needs (external
        // audit, Rank 1). LL leans harder on the GRU (no lookahead), so it matters
        // more here.
        // Advance BOTH decoder recurrences on every hop, so their GRU and temporal
        // convolution histories (LL adds convt3_pad / conv0out_pad) stay
        // continuous. They used to be skipped whenever the output was gated,
        // feeding the next onset stale decoder state (external audit, Rank 2).
        self.erb_decoder();
        self.df_decoder();

        let (apply_gains, apply_gain_zeros, apply_df) = if self.lsnr < self.min_db {
            (false, true, false)
        } else if self.lsnr > self.max_db_erb {
            (false, false, false)
        } else if self.lsnr > self.max_db_df {
            (true, false, false)
        } else {
            (true, false, true)
        };

        // `erb_mask` and the DF coefs are already computed; the flags only choose
        // what reaches the output.
        let mut has_gains = false;
        if apply_gains {
            has_gains = true;
        } else if apply_gain_zeros {
            self.erb_mask.fill(0.0);
            has_gains = true;
        }
        let has_coefs = apply_df;

        let delayed = DF_ORDER - 1;
        {
            let phys = self.yphys(delayed);
            if has_gains {
                let (start_band, mut off) = if has_coefs {
                    (ERB_DF_SKIP_BAND, ERB_DF_SKIP_OFFSET)
                } else {
                    (0usize, 0usize)
                };
                let buf = &mut self.roll_y[phys];
                for b in start_band..NB_ERB {
                    let g = self.erb_mask[b];
                    let bw = ERB_WIDTHS[b];
                    for i in 0..bw {
                        buf[off + i] *= g;
                        buf[FREQ + off + i] *= g;
                    }
                    off += bw;
                }
                self.silence = 0;
            } else {
                self.silence += 1;
            }
        }

        // move the preallocated enh buffers out (no heap; restored at end).
        let mut enh_re = std::mem::take(&mut self.enh_re);
        let mut enh_im = std::mem::take(&mut self.enh_im);
        {
            let phys = self.yphys(delayed);
            let src = &self.roll_y[phys];
            enh_re.copy_from_slice(&src[..FREQ]);
            enh_im.copy_from_slice(&src[FREQ..]);
        }
        if has_coefs {
            self.apply_df(&mut enh_re, &mut enh_im);
        }

        let noisy_logical = SPEC_X - LOOKAHEAD - 1;
        if apply_gains && self.post_filter_beta > 0.0 {
            let phys = self.xphys(noisy_logical);
            let (nr, ni) = self.roll_x[phys].split_at(FREQ);
            post_filter(nr, ni, &mut enh_re, &mut enh_im, self.post_filter_beta);
        }
        if self.atten_lim > 0.0 {
            let phys = self.xphys(noisy_logical);
            let one_m = 1.0 - self.atten_lim;
            for i in 0..FREQ {
                enh_re[i] = enh_re[i] * one_m + self.roll_x[phys][i] * self.atten_lim;
                enh_im[i] = enh_im[i] * one_m + self.roll_x[phys][FREQ + i] * self.atten_lim;
            }
        }

        // Per-ERB level of the processed spectrum (same definition as extract_erb,
        // but on the enhanced output bins), for the plugin's per-frequency floor.
        {
            let mut off = 0usize;
            for b in 0..NB_ERB {
                let width = ERB_WIDTHS[b];
                // Peak bin power in the band, matching the GUI spectrum bars' scale.
                let mut peak = 0.0f32;
                for i in off..off + width {
                    let m = enh_re[i] * enh_re[i] + enh_im[i] * enh_im[i];
                    if m > peak {
                        peak = m;
                    }
                }
                self.band_db[b] = 10.0 * dfn_ops::log10_f(peak + 1e-12);
                off += width;
            }
        }

        self.frame_synthesis(out, &enh_re, &enh_im);
        self.enh_re = enh_re;
        self.enh_im = enh_im;
    }
}

fn split_two<'a>(dst: &'a mut [f32], skip: &'a [f32], n: usize) -> (&'a mut [f32], &'a [f32]) {
    (&mut dst[..n], &skip[..n])
}

/// Separable conv, temporal kernel 2 (1 past frame) + freq kernel 3 stride 2, pw, ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_k2_s2(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    pad: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_in: usize,
    w_out: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_out..c * w_out + w_out];
        out_c.fill(0.0);
        let wk = &dw[c * 6..c * 6 + 6];
        let past = &pad[c * w_in..c * w_in + w_in];
        let cur = &src[c * w_in..c * w_in + w_in];
        dw_row_k3s2_accum(out_c, past, wk[0], wk[1], wk[2]);
        dw_row_k3s2_accum(out_c, cur, wk[3], wk[4], wk[5]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_out);
    relu_inplace(&mut dst[..CH * w_out]);
    pad[..CH * w_in].copy_from_slice(&src[..CH * w_in]);
}

/// Separable conv, temporal kernel 2 + freq kernel 3 stride 1, pw, ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_k2_s1(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    pad: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_io: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 6..c * 6 + 6];
        let past = &pad[c * w_io..c * w_io + w_io];
        let cur = &src[c * w_io..c * w_io + w_io];
        dw_row_k3s1_accum(out_c, past, wk[0], wk[1], wk[2]);
        dw_row_k3s1_accum(out_c, cur, wk[3], wk[4], wk[5]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut dst[..CH * w_io]);
    pad[..CH * w_io].copy_from_slice(&src[..CH * w_io]);
}

fn conv_p_add(dst: &mut [f32], src: &[f32], skip: &[f32], w: &[f32], b: &[f32], width: usize) {
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

/// convt3: temporal kernel 2 depthwise (1 past) freq kernel 3 stride 1 into `scr`,
/// pointwise back into `sio`, ReLU.
fn convt3_k2(
    sio: &mut [f32],
    scr: &mut [f32],
    pad: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_io: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 6..c * 6 + 6];
        let past = &pad[c * w_io..c * w_io + w_io];
        let cur = &sio[c * w_io..c * w_io + w_io];
        dw_row_k3s1_accum(out_c, past, wk[0], wk[1], wk[2]);
        dw_row_k3s1_accum(out_c, cur, wk[3], wk[4], wk[5]);
    }
    pad[..CH * w_io].copy_from_slice(&sio[..CH * w_io]);
    pointwise_conv2d(sio, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut sio[..CH * w_io]);
}

#[allow(clippy::too_many_arguments)]
fn convt_up(
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

fn post_filter(nr: &[f32], ni: &[f32], er: &mut [f32], ei: &mut [f32], beta: f32) {
    let beta_p1 = beta + 1.0;
    let eps = 1e-12f32;
    for i in 0..FREQ {
        let em = (er[i] * er[i] + ei[i] * ei[i]).sqrt();
        let nm = (nr[i] * nr[i] + ni[i] * ni[i]).sqrt();
        let mut g = em / (nm + eps);
        g = g.min(1.0).max(eps);
        let gs = g * dfn_ops::sin_unit(g * std::f32::consts::FRAC_PI_2);
        let ratio = g / gs;
        let pf = beta_p1 / (1.0 + beta * ratio * ratio);
        er[i] *= pf;
        ei[i] *= pf;
    }
}

#[cfg(test)]
mod scratch_budget_tests {
    use super::*;
    #[test]
    fn continuous_encoder_still_gates_noise_and_voice_resumes() {
        let mut engine = Dfn3Ll::new(include_bytes!("../dfn3ll_weights.bin"));
        // eSpeak NG, en-us, 150 wpm: "Hello. This is a microphone test."
        // Mono signed little-endian PCM at 48 kHz; no recorded user speech.
        let speech: Vec<f32> = include_bytes!("../../dfn3-ladspa/tests/fixtures/speech.pcm")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| f32::from(i16::from_le_bytes(v)) / 32768.0)
            .collect();
        let mut out = [0.0; HOP];
        for _ in 0..2 {
            for frame in speech.as_chunks::<HOP>().0.iter().take(50) {
                engine.process(frame, &mut out);
            }
            assert!(
                engine.speech_frames < CONTEXT_FRAMES,
                "half-second utterance established speech context"
            );
            for _ in 0..60 {
                engine.process(&[0.0; HOP], &mut out);
            }
            assert_eq!(engine.speech_frames, 0, "pause did not reset validation");
        }
        for frame in speech.as_chunks::<HOP>().0 {
            engine.process(frame, &mut out);
        }
        assert!(
            engine.speech_frames == CONTEXT_FRAMES,
            "voice did not establish a one-second context"
        );
        let mut seed = 17_u32;
        let mut gated = 0;
        for _ in 0..6000 {
            let input = std::array::from_fn::<_, HOP, _>(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                0.001 * (2.0 * (seed >> 9) as f32 / 8_388_608.0 - 1.0)
            });
            engine.process(&input, &mut out);
            assert_eq!(engine.silence, 0, "white noise bypassed neural inference");
            // The encoder recurrence now advances on every hop (no rollback); only
            // count how often noise drove the output gate.
            if engine.lsnr < engine.min_db {
                gated += 1;
            }
        }
        for _ in 0..60 {
            engine.process(&[0.0; HOP], &mut out);
        }
        let mut burst_energy = 0.0_f64;
        for frame in speech.as_chunks::<HOP>().0.iter().take(50) {
            engine.process(frame, &mut out);
            burst_energy += out.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        }
        assert!(burst_energy > 0.1, "validation muted a short utterance");
        let mut resumed = 0.0_f64;
        for frame in speech.as_chunks::<HOP>().0 {
            engine.process(frame, &mut out);
            resumed += out.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        }
        assert!(
            gated > 100,
            "noise did not exercise the gate: {gated} frames"
        );
        assert!(resumed > 1.0, "voice failed to resume: energy {resumed}");
    }

    #[test]
    fn scratch_matches_network_geometry_and_processes_multiple_hops() {
        let mut engine = Dfn3Ll::new(include_bytes!("../dfn3ll_weights.bin"));
        assert_eq!(engine.s1.len(), CH * NB_DF);
        assert_eq!(engine.s2.len(), CH * (NB_DF / 2));
        let saved_bytes =
            (2 * CH * FREQ - engine.s1.len() - engine.s2.len()) * std::mem::size_of::<f32>();
        assert_eq!(saved_bytes, 209_408);
        let mut out = [0.0; HOP];
        for hop in 0..100 {
            let input = std::array::from_fn::<_, HOP, _>(|i| {
                let t = (hop * HOP + i) as f32 / SR as f32;
                0.15 * (std::f32::consts::TAU * 173.0 * t).sin()
            });
            engine.process(&input, &mut out);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
