//! DeepFilterNet3 streaming speech denoiser — independent Rust implementation.
//!
//! Reimplements the DeepFilterNet3 real-time inference pipeline; the weights are
//! derived from the upstream DeepFilterNet3 ONNX. The algorithm — STFT, ERB +
//! complex features, encoder/decoder with squeezed GRUs, deep filtering, ISTFT —
//! follows the published libDF design. This is our own code: standard NN ops,
//! `realfft` for the transform, exact libm math. Native only (no WASM); the
//! shipped artifact is a mono LADSPA plugin.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design

mod ladspa;
mod weights;

#[cfg(feature = "r11-packed")]
use dfn_ops::gru_cell_packed as active_gru;
#[cfg(not(feature = "r11-packed"))]
use dfn_ops::gru_cell_q as active_gru;
use dfn_ops::*;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
use weights::W;

/// Test hook: the LADSPA descriptor as a raw pointer (LADSPA 1.1 ABI layout).
#[doc(hidden)]
pub fn ladspa_descriptor_ptr(index: usize) -> *const std::ffi::c_void {
    ladspa::ladspa_descriptor(index as std::os::raw::c_ulong) as *const std::ffi::c_void
}

/// Shared dB→noisy-mix conversion, re-exported so the CLIs and tests attenuate
/// exactly as the LADSPA plugin does.
pub use dfn_ops::atten_lim_from_db;

/// A decoded, shareable weight table (decode once, build many `Dfn3`).
pub struct Weights(Arc<W>);

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
pub const HID: usize = 256;

const LSNR_MAX: f32 = 35.0;
const LSNR_MIN: f32 = -15.0;
const SILENCE_RMS: f32 = 1e-7;
const SILENCE_SKIP_MAX: i32 = 5;
const CONTEXT_FRAMES: u16 = (SR / HOP) as u16;

const SPEC_Y: usize = DF_ORDER + 2; // 7
const SPEC_X: usize = DF_ORDER; // 5
const LOOKAHEAD: usize = 2;

pub const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37,
    42, 50, 56, 67,
];
const ERB_DF_SKIP_BAND: usize = 21;
const ERB_DF_SKIP_OFFSET: usize = 93;

pub struct Dfn3 {
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
    feat_erb: Vec<f32>,
    feat_spec: Vec<f32>,

    erb0_pad: Vec<f32>,
    df0_pad: Vec<f32>,
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    e3: Vec<f32>,
    c0: Vec<f32>,
    emb: Vec<f32>,
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
    gru_scratch: Vec<f32>,
    i16q: Vec<i16>,

    erb_mask: Vec<f32>,
    convp_pad: Vec<f32>,
    convp_head: usize,
    coefs_t: Vec<f32>,

    s1: Vec<f32>,
    s2: Vec<f32>,
    // preallocated per-frame scratch (RT hygiene: no heap in process()).
    sc_hid: Vec<f32>,    // HID: gru input `gin`
    h_tmp: Vec<f32>,     // HID: stacked-GRU hidden copy
    sc_dfcomb: Vec<f32>, // HID: df comb
    sc_skip: Vec<f32>,   // HID: df skip
    sc_emb: Vec<f32>,    // EMB: grouped-linear outputs
    sc_flat: Vec<f32>,   // 48*CH: df transpose
    sc_emb2d: Vec<f32>,  // CH*8: erb dec emb2d
    sc_960a: Vec<f32>,   // NB_DF*10: coefs
    sc_960b: Vec<f32>,   // NB_DF*10: c0_proj
    sc_960c: Vec<f32>,   // NB_DF*10: dw_out
    sc_960d: Vec<f32>,   // NB_DF*10: pw_out
    sc_trans: Vec<f32>,  // CH*33: convt_up transpose scratch
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
    /// Diagnostic only: round-trip the deep-filter head's input through int8
    /// (per-vector amax/127) before its projection, to measure what an A8
    /// activation would cost on the tensor that shapes the output coefficients.
    /// Not part of the shipped signal path (defaults off).
    pub a8_df_head: bool,
}

impl Dfn3 {
    pub fn new(weights_bytes: &[u8]) -> Self {
        Self::from_shared(Arc::new(W::load(weights_bytes)))
    }

    /// Build an instance from an already-decoded shared weight table.
    pub fn with_weights(w: &Weights) -> Self {
        Self::from_shared(w.0.clone())
    }

    /// Build an instance from an already-decoded shared weight table, so multiple
    /// instances and `activate()` resets don't each re-copy the blob.
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

        Dfn3 {
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
            gru_scratch: vec![0.0; 6 * HID],
            i16q: vec![0i16; HID],
            erb_mask: vec![0.0; NB_ERB],
            convp_pad: vec![0.0; CH * 4 * NB_DF],
            convp_head: 0,
            coefs_t: vec![0.0; DF_ORDER * 2 * NB_DF],
            // Largest live scratch planes: DF input (64x96) and strided DF (64x48).
            s1: vec![0.0; CH * NB_DF],
            s2: vec![0.0; CH * (NB_DF / 2)],
            sc_hid: vec![0.0; HID],
            h_tmp: vec![0.0; HID],
            sc_dfcomb: vec![0.0; HID],
            sc_skip: vec![0.0; HID],
            sc_emb: vec![0.0; EMB],
            sc_flat: vec![0.0; 48 * CH],
            sc_emb2d: vec![0.0; CH * 8],
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
            min_db: -10.0,
            max_db_erb: 30.0,
            max_db_df: 20.0,
            a8_df_head: false,
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
        // A real signal has zero imaginary part at DC and Nyquist; the deep-filter
        // complex MAC can leave a residue at bin 0 that realfft rejects.
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
        // erb_conv0: Conv2d(1,64,3,3) causal-pad + ReLU -> e0
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
        // erb_conv1/2 (dw s2 + pw + relu), erb_conv3 (dw s1 + pw + relu)
        sep_s2(
            &mut self.e1,
            &mut self.s1,
            &self.e0,
            self.w.enc_erb_conv1_dw_w(),
            self.w.enc_erb_conv1_pw_w(),
            self.w.enc_erb_conv1_pw_b(),
            NB_ERB,
            NB_ERB / 2,
        );
        sep_s2(
            &mut self.e2,
            &mut self.s1,
            &self.e1,
            self.w.enc_erb_conv2_dw_w(),
            self.w.enc_erb_conv2_pw_w(),
            self.w.enc_erb_conv2_pw_b(),
            NB_ERB / 2,
            NB_ERB / 4,
        );
        sep_s1(
            &mut self.e3,
            &mut self.s1,
            &self.e2,
            self.w.enc_erb_conv3_dw_w(),
            self.w.enc_erb_conv3_pw_w(),
            self.w.enc_erb_conv3_pw_b(),
            NB_ERB / 4,
        );

        // df_conv0: groups=2 causal + pw + relu -> c0
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
        // df_conv1 dw s2 + pw + relu -> s2 [64,48]
        {
            let win = NB_DF;
            let wout = NB_DF / 2;
            let wt = self.w.enc_df_conv1_dw_w();
            for c in 0..CH {
                let out_c = &mut self.s1[c * wout..c * wout + wout];
                out_c.fill(0.0);
                let wk = &wt[c * 3..c * 3 + 3];
                dw_row_k3s2_accum(out_c, &self.c0[c * win..c * win + win], wk[0], wk[1], wk[2]);
            }
            pointwise_conv2d(
                &mut self.s2,
                &self.s1,
                self.w.enc_df_conv1_pw_w(),
                self.w.enc_df_conv1_pw_b(),
                CH,
                CH,
                wout,
            );
            relu_inplace(&mut self.s2[..CH * wout]);
        }

        // embedding: transpose df [64,48]->[48,64] flat, grouped einsum -> cemb; e3 transpose -> erb_flat; emb = erb_flat + cemb
        {
            for ch in 0..CH {
                for f in 0..48 {
                    self.sc_flat[f * CH + ch] = self.s2[ch * 48 + f];
                }
            }
            grouped_linear(
                &mut self.sc_emb,
                &self.sc_flat,
                self.w.enc_df_fc_emb_w(),
                32,
                96,
                16,
            );
            relu_inplace(&mut self.sc_emb);
            for ch in 0..CH {
                for f in 0..8 {
                    self.emb[f * CH + ch] = self.e3[ch * 8 + f];
                }
            }
            vadd(&mut self.emb, &self.sc_emb);
        }

        // encoder squeezed GRU
        {
            grouped_linear(
                &mut self.sc_hid,
                &self.emb,
                self.w.enc_emb_gru_lin_in_w(),
                16,
                32,
                16,
            );
            relu_inplace(&mut self.sc_hid);
            active_gru(
                &mut self.enc_h,
                &self.sc_hid,
                self.w.enc_emb_gru0_W_q(),
                self.w.enc_emb_gru0_W_s(),
                self.w.enc_emb_gru0_R_q(),
                self.w.enc_emb_gru0_R_s(),
                self.w.enc_emb_gru0_B(),
                HID,
                HID,
                &mut self.gru_scratch,
                &mut self.i16q,
            );
            grouped_linear(
                &mut self.sc_emb,
                &self.enc_h,
                self.w.enc_emb_gru_lin_out_w(),
                16,
                16,
                32,
            );
            relu_inplace(&mut self.sc_emb);
            self.emb.copy_from_slice(&self.sc_emb);
        }

        // lsnr
        let val = self.w.enc_lsnr_fc_b()[0] + vdot(&self.emb, self.w.enc_lsnr_fc_w());
        self.lsnr = sigmoid(val) * (LSNR_MAX - LSNR_MIN) + LSNR_MIN;
    }

    fn erb_decoder(&mut self) {
        grouped_linear(
            &mut self.sc_hid,
            &self.emb,
            self.w.erb_emb_gru_lin_in_w(),
            16,
            32,
            16,
        );
        relu_inplace(&mut self.sc_hid);
        active_gru(
            &mut self.erb0_h,
            &self.sc_hid,
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
        self.h_tmp.copy_from_slice(&self.erb0_h);
        active_gru(
            &mut self.erb1_h,
            &self.h_tmp,
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
            &mut self.sc_emb,
            &self.erb1_h,
            self.w.erb_emb_gru_lin_out_w(),
            16,
            16,
            32,
        );
        relu_inplace(&mut self.sc_emb);
        for f in 0..8 {
            for ch in 0..CH {
                self.sc_emb2d[ch * 8 + f] = self.sc_emb[f * CH + ch];
            }
        }
        // C3p: s1 = relu(conv3p(e3)) + emb2d
        conv_p_add(
            &mut self.s1,
            &self.e3,
            &self.sc_emb2d,
            self.w.erb_conv3p_dw_w(),
            self.w.erb_conv3p_dw_b(),
            8,
        );
        // Ct3: dw s1 -> s2, pw -> s1
        convt_dw_pw(
            &mut self.s1,
            &mut self.s2,
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
        // conv0_out: [1,64,1,3] group1 + sigmoid -> erb_mask
        {
            let wt = self.w.erb_conv0_out_w();
            let b = self.w.erb_conv0_out_b()[0];
            self.erb_mask.fill(b);
            for ci in 0..CH {
                let wc = &wt[ci * 3..ci * 3 + 3];
                dw_row_k3s1_accum(
                    &mut self.erb_mask,
                    &self.s2[ci * NB_ERB..ci * NB_ERB + NB_ERB],
                    wc[0],
                    wc[1],
                    wc[2],
                );
            }
            for v in self.erb_mask.iter_mut() {
                *v = sigmoid(*v);
            }
        }
    }

    fn df_decoder(&mut self) {
        grouped_linear(
            &mut self.sc_hid,
            &self.emb,
            self.w.df_gru_lin_in_w(),
            8,
            64,
            32,
        );
        relu_inplace(&mut self.sc_hid);
        active_gru(
            &mut self.df0_h,
            &self.sc_hid,
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
        self.h_tmp.copy_from_slice(&self.df0_h);
        active_gru(
            &mut self.df1_h,
            &self.h_tmp,
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

        self.sc_dfcomb.copy_from_slice(&self.df1_h);
        grouped_linear(&mut self.sc_skip, &self.emb, self.w.df_skip_w(), 16, 32, 16);
        vadd(&mut self.sc_dfcomb, &self.sc_skip);
        if self.a8_df_head {
            // Diagnostic: what an A8 activation would cost on the head's input.
            let amax = self.sc_dfcomb.iter().fold(0f32, |a, &v| a.max(v.abs()));
            if amax > 0.0 {
                let inv = 127.0 / amax;
                let scale = amax / 127.0;
                for v in self.sc_dfcomb.iter_mut() {
                    *v = (*v * inv).round().clamp(-127.0, 127.0) * scale;
                }
            }
        }
        grouped_linear(
            &mut self.sc_960a,
            &self.sc_dfcomb,
            self.w.df_out_w(),
            16,
            16,
            60,
        );
        for v in self.sc_960a.iter_mut() {
            *v = dfn_ops::tanh_f(*v);
        }

        // df_convp: dw (5,1) groups=2 on c0 with 4-frame causal pad, pw, relu,
        // transpose -> sc_960b (c0_proj[96,10]).
        {
            let cpg_in = CH / 2;
            let cpg_out = 5;
            let kh = 5;
            let wt = self.w.df_convp_dw_w();
            self.sc_960c.fill(0.0);
            for g in 0..2 {
                for co in 0..cpg_out {
                    let co_abs = g * cpg_out + co;
                    let dst = &mut self.sc_960c[co_abs * NB_DF..co_abs * NB_DF + NB_DF];
                    #[cfg(target_arch = "x86_64")]
                    if dfn_ops::simd_tier() >= 2 {
                        let taps = &wt[co_abs * cpg_in * kh..(co_abs + 1) * cpg_in * kh];
                        // SAFETY: AVX was detected at run time; every slice is
                        // bounds-checked inside.
                        unsafe {
                            convp_taps_avx(
                                dst,
                                &self.convp_pad,
                                &self.c0,
                                taps,
                                g * cpg_in,
                                cpg_in,
                                self.convp_head,
                            )
                        };
                        continue;
                    }
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

        for i in 0..NB_DF * 10 {
            self.sc_960a[i] += self.sc_960b[i];
        }
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
        // Retained encoder context stays warm through pauses without any mute.
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
        // recurrence h_t = F(x_t, h_{t-1}). It used to be rolled back to the prior
        // state whenever `lsnr < min_db`, which discarded the gated frame's
        // contribution to the recurrent evidence an onset needs and could retain
        // speech-conditioned state across changing noise (external audit, Rank 1).
        // Advance BOTH decoder recurrences on every hop, so their GRU and temporal
        // convolution histories stay continuous. They used to be skipped whenever
        // the output was gated (noise/pass-through branches), which fed the next
        // speech onset stale decoder state — a residual/transition source
        // (external audit, Rank 2). Only the OUTPUT stage below is gated.
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
                // Peak bin power in the band, so the level matches the GUI spectrum
                // bars (which take the peak bin per band, 20·log10 magnitude). Using
                // the peak (not the mean) keeps the plugin and the on-screen bars on
                // the same scale, so the user's drawn curve lines up with what they see.
                let mut peak = 0.0f32;
                for i in off..off + width {
                    let m = enh_re[i] * enh_re[i] + enh_im[i] * enh_im[i];
                    if m > peak {
                        peak = m;
                    }
                }
                // 10·log10(peak power) == 20·log10(peak magnitude), plus a fixed
                // calibration offset that aligns this 960-pt engine spectrum with the
                // GUI's 4096-pt windowed FFT dB (see FLOOR_CALIB in the plugin).
                self.band_db[b] = 10.0 * dfn_ops::log10_f(peak + 1e-12);
                off += width;
            }
        }

        self.frame_synthesis(out, &enh_re, &enh_im);
        self.enh_re = enh_re;
        self.enh_im = enh_im;
    }
}

/// Borrow two disjoint prefixes of two distinct buffers as (&mut dst, &skip).
fn split_two<'a>(dst: &'a mut [f32], skip: &'a [f32], n: usize) -> (&'a mut [f32], &'a [f32]) {
    (&mut dst[..n], &skip[..n])
}

/// Separable conv, stride-2 depthwise (kernel 3, pad 1) + pointwise + ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_s2(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_in: usize,
    w_out: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_out..c * w_out + w_out];
        out_c.fill(0.0);
        let wk = &dw[c * 3..c * 3 + 3];
        dw_row_k3s2_accum(out_c, &src[c * w_in..c * w_in + w_in], wk[0], wk[1], wk[2]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_out);
    relu_inplace(&mut dst[..CH * w_out]);
}

/// Separable conv, stride-1 depthwise (kernel 3, pad 1) + pointwise + ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_s1(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_io: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 3..c * 3 + 3];
        dw_row_k3s1_accum(out_c, &src[c * w_io..c * w_io + w_io], wk[0], wk[1], wk[2]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut dst[..CH * w_io]);
}

/// Per-channel 1x1 (scale+bias) + ReLU on `src`, then add `skip`, into `dst`.
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

/// Depthwise (kernel 3, stride 1) into `scr`, pointwise back into `sio`, ReLU.
fn convt_dw_pw(sio: &mut [f32], scr: &mut [f32], dw: &[f32], pw: &[f32], pb: &[f32], w_io: usize) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 3..c * 3 + 3];
        dw_row_k3s1_accum(out_c, &sio[c * w_io..c * w_io + w_io], wk[0], wk[1], wk[2]);
    }
    pointwise_conv2d(sio, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut sio[..CH * w_io]);
}

/// Transposed depthwise conv stride 2 (pads [0,1,0,1], output_padding [0,1])
/// reading `sio` (`w_in`), pointwise + ReLU back into `sio` (`w_out`). `scr` scratch.
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
        let mut engine = Dfn3::new(include_bytes!("../dfn3_weights.bin"));
        // First 3 s of eSpeak NG, en-us, 130 wpm: "We are evaluating the
        // microphone recording quality continuously during this conversation."
        // Mono signed little-endian PCM at 48 kHz; no recorded user speech.
        let speech: Vec<f32> = include_bytes!("../tests/fixtures/continuous-speech.pcm")
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
            let previous_y_head = engine.y_head;
            engine.process(&input, &mut out);
            assert_eq!(
                engine.y_head,
                (previous_y_head + 1) % SPEC_Y,
                "white noise bypassed neural inference"
            );
            // The encoder recurrence now advances on every hop (no rollback); we
            // only count how often noise drove the output gate.
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
        let mut engine = Dfn3::new(include_bytes!("../dfn3_weights.bin"));
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

/// AVX body of the `df_convp` depthwise taps for one output channel: `dst` (96
/// floats, 12 registers) stays in registers across all `cpg_in * 5` taps instead
/// of being loaded and stored per tap as the SSE2 loop does. Each lane adds
/// `src * w` in the same (ci, k) order with a separate multiply and add, so the
/// sum is bit-identical to the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn convp_taps_avx(
    dst: &mut [f32],
    convp_pad: &[f32],
    c0: &[f32],
    taps: &[f32],
    ci_base: usize,
    cpg_in: usize,
    head: usize,
) {
    use std::arch::x86_64::*;
    const LANES: usize = NB_DF / 8;
    assert!(dst.len() == NB_DF && taps.len() == cpg_in * 5);
    let mut acc = [_mm256_setzero_ps(); LANES];
    for (l, a) in acc.iter_mut().enumerate() {
        *a = _mm256_loadu_ps(dst.as_ptr().add(l * 8));
    }
    for ci in 0..cpg_in {
        let ci_abs = ci_base + ci;
        for k in 0..5 {
            let src: &[f32] = if k < 4 {
                let at = ci_abs * 4 * NB_DF + ((head + k) & 3) * NB_DF;
                &convp_pad[at..at + NB_DF]
            } else {
                &c0[ci_abs * NB_DF..ci_abs * NB_DF + NB_DF]
            };
            let w = _mm256_set1_ps(taps[ci * 5 + k]);
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm256_add_ps(
                    *a,
                    _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(l * 8)), w),
                );
            }
        }
    }
    for (l, a) in acc.iter().enumerate() {
        _mm256_storeu_ps(dst.as_mut_ptr().add(l * 8), *a);
    }
}
