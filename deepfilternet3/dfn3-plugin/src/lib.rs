//! The part of the DeepFilterNet3 streaming pipeline both models share, and the
//! LADSPA plugin built on it.
//!
//! An independent Rust implementation of the published libDF design: STFT, ERB and
//! complex features, a network (encoder and two decoders, see [`Network`]), an ERB
//! mask and a deep filter on the lowest bins, iSTFT. Standard NN ops, `realfft` for
//! the transform, exact libm math.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design

mod expander;
pub mod ladspa;
pub mod layers;
mod weights;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub use ops::atten_lim_from_db;
pub use weights::{AlignedBlob, Tensors};

pub const SR: usize = 48000;
pub const FFT: usize = 960;
pub const HOP: usize = 480;
pub const FREQ: usize = 481;
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;
/// Deep-filter coefficients per hop, laid out `[NB_DF][DF_ORDER][re, im]`.
pub const DF_COEFS: usize = NB_DF * DF_ORDER * 2;
/// Convolution channels of both networks.
pub const CH: usize = 64;
pub const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37,
    42, 50, 56, 67,
];

const LSNR_MAX: f32 = 35.0;
const LSNR_MIN: f32 = -15.0;
/// Mean-square level below which a hop is silence (about -70 dBFS RMS).
const SILENCE_MEAN_SQUARE: f32 = 1e-7;
/// Silent hops still processed before the rest are skipped (2 s), so the
/// recurrent state settles before it freezes.
pub const SILENCE_SKIP_MAX: u32 = 200;
/// With the deep filter on, the ERB mask only covers the bands above its bins.
const ERB_DF_SKIP_BAND: usize = 21;
const ERB_DF_SKIP_OFFSET: usize = 93;
/// `band_db` on hops that were not processed.
const BAND_DB_FLOOR: f32 = -200.0;

/// The model-specific part of a DeepFilterNet3 variant.
pub trait Network {
    /// Hops the model looks ahead; the output lags the input by as many hops more.
    const LOOKAHEAD: usize;
    /// A fresh network over the weights embedded in the plugin.
    fn embedded() -> Self;
    /// Runs the encoder and both decoders on one hop's features: fills the ERB mask
    /// and the deep-filter coefficients and returns the local SNR estimate in dB.
    fn infer(
        &mut self,
        feat_erb: &[f32; NB_ERB],
        feat_spec: &[f32; 2 * NB_DF],
        erb_mask: &mut [f32; NB_ERB],
        coefs: &mut [f32; DF_COEFS],
    ) -> f32;
}

/// The local SNR estimate the encoder's `lsnr_fc` head maps to dB.
pub fn lsnr(emb: &[f32], w: &[f32], b: f32) -> f32 {
    ops::sigmoid(b + ops::vdot(emb, w)) * (LSNR_MAX - LSNR_MIN) + LSNR_MIN
}

/// A streaming denoiser: 480 samples in, 480 out per [`Denoiser::process`] call.
pub struct Denoiser<N> {
    net: N,
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
    /// The last `DF_ORDER + LOOKAHEAD` spectra; the ERB mask applies to the one
    /// `LOOKAHEAD` hops old.
    roll_y: Vec<[f32; 2 * FREQ]>,
    /// The last `DF_ORDER` spectra, which the deep filter combines.
    roll_x: Vec<[f32; 2 * FREQ]>,
    y_head: usize,
    x_head: usize,

    erb_norm: Vec<f32>,
    unit_norm: Vec<f32>,
    alpha: f32,
    feat_erb: [f32; NB_ERB],
    feat_spec: [f32; 2 * NB_DF],
    erb_mask: [f32; NB_ERB],
    coefs: [f32; DF_COEFS],
    /// `coefs` regrouped per tap: `[DF_ORDER][re: NB_DF, im: NB_DF]`.
    coefs_t: Vec<f32>,
    enh_re: Vec<f32>,
    enh_im: Vec<f32>,
    /// Consecutive silent input hops.
    silence: u32,

    /// Local SNR estimate of the last hop (dB).
    pub lsnr: f32,
    /// Per-ERB level of the processed spectrum (dB of the peak bin power), on the
    /// scale of the GUI's spectrum bars. `BAND_DB_FLOOR` on skipped hops.
    pub band_db: [f32; NB_ERB],
    /// Fraction of the noisy input mixed back in: 0 is full enhancement.
    pub atten_lim: f32,
    /// DeepFilterNet post-filter strength; 0 is off.
    pub post_filter_beta: f32,
    /// Below this LSNR a hop is treated as noise only and muted.
    pub min_db: f32,
    /// Above this LSNR the hop passes without mask or deep filter.
    pub max_db_erb: f32,
    /// Above this LSNR only the deep filter is skipped.
    pub max_db_df: f32,
}

impl<N: Network> Default for Denoiser<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Network> Denoiser<N> {
    const SPEC_Y: usize = DF_ORDER + N::LOOKAHEAD;
    const SPEC_X: usize = DF_ORDER;

    pub fn new() -> Self {
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

        Self {
            net: N::embedded(),
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
            roll_y: vec![[0.0; 2 * FREQ]; Self::SPEC_Y],
            roll_x: vec![[0.0; 2 * FREQ]; Self::SPEC_X],
            y_head: 0,
            x_head: 0,
            erb_norm,
            unit_norm,
            alpha,
            feat_erb: [0.0; NB_ERB],
            feat_spec: [0.0; 2 * NB_DF],
            erb_mask: [0.0; NB_ERB],
            coefs: [0.0; DF_COEFS],
            coefs_t: vec![0.0; DF_COEFS],
            enh_re: vec![0.0; FREQ],
            enh_im: vec![0.0; FREQ],
            silence: 0,
            lsnr: 0.0,
            band_db: [BAND_DB_FLOOR; NB_ERB],
            atten_lim: 0.0,
            post_filter_beta: 0.0,
            min_db: -10.0,
            max_db_erb: 40.0,
            max_db_df: 40.0,
        }
    }

    #[inline]
    fn yphys(&self, logical: usize) -> usize {
        (self.y_head + logical) % Self::SPEC_Y
    }
    #[inline]
    fn xphys(&self, logical: usize) -> usize {
        (self.x_head + logical) % Self::SPEC_X
    }

    fn frame_analysis(&mut self, input: &[f32; HOP]) {
        for i in 0..HOP {
            self.fft_r[i] = self.analysis_mem[i] * self.window[i];
            self.fft_r[HOP + i] = input[i] * self.window[HOP + i];
        }
        self.analysis_mem.copy_from_slice(input);
        self.r2c
            .process_with_scratch(&mut self.fft_r, &mut self.fft_c, &mut self.r2c_scratch)
            .expect("rfft");
        for i in 0..FREQ {
            self.spec_re[i] = self.fft_c[i].re * self.wnorm;
            self.spec_im[i] = self.fft_c[i].im * self.wnorm;
        }
    }

    fn frame_synthesis(&mut self, out: &mut [f32; HOP], enh_re: &[f32], enh_im: &[f32]) {
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
            let db = 10.0 * (energy + 1e-10).log10();
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

    /// Denoises one hop. The output lags the input by `FFT - HOP` samples plus
    /// `N::LOOKAHEAD` hops.
    pub fn process(&mut self, input: &[f32; HOP], out: &mut [f32; HOP]) {
        let e: f32 = input.iter().map(|v| v * v).sum();
        if e / (HOP as f32) < SILENCE_MEAN_SQUARE {
            self.silence = self.silence.saturating_add(1);
        } else {
            self.silence = 0;
        }
        if self.silence > SILENCE_SKIP_MAX {
            out.fill(0.0);
            self.lsnr = LSNR_MIN;
            self.band_db = [BAND_DB_FLOOR; NB_ERB];
            return;
        }

        self.frame_analysis(input);
        {
            let d = &mut self.roll_y[self.y_head];
            d[..FREQ].copy_from_slice(&self.spec_re);
            d[FREQ..].copy_from_slice(&self.spec_im);
            self.y_head = (self.y_head + 1) % Self::SPEC_Y;
            let d = &mut self.roll_x[self.x_head];
            d[..FREQ].copy_from_slice(&self.spec_re);
            d[FREQ..].copy_from_slice(&self.spec_im);
            self.x_head = (self.x_head + 1) % Self::SPEC_X;
        }

        self.extract_erb();
        self.extract_spec();
        // The network runs on every hop the gate below mutes too, so its recurrent
        // and temporal-convolution state stays continuous into the next onset.
        self.lsnr = self.net.infer(
            &self.feat_erb,
            &self.feat_spec,
            &mut self.erb_mask,
            &mut self.coefs,
        );
        for n in 0..DF_ORDER {
            for f in 0..NB_DF {
                self.coefs_t[n * 2 * NB_DF + f] = self.coefs[f * 10 + n * 2];
                self.coefs_t[n * 2 * NB_DF + NB_DF + f] = self.coefs[f * 10 + n * 2 + 1];
            }
        }

        let (apply_gains, apply_gain_zeros, apply_df) = if self.lsnr < self.min_db {
            (false, true, false)
        } else if self.lsnr > self.max_db_erb {
            (false, false, false)
        } else if self.lsnr > self.max_db_df {
            (true, false, false)
        } else {
            (true, false, true)
        };

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
            }
        }

        // Moved out and back so `self` stays borrowable; no allocation.
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

        let noisy_logical = Self::SPEC_X - N::LOOKAHEAD - 1;
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

        // The GUI's spectrum bars show the peak bin of each band; matching that
        // keeps a floor curve drawn against them on the same scale.
        let mut off = 0usize;
        for b in 0..NB_ERB {
            let width = ERB_WIDTHS[b];
            let mut peak = 0.0f32;
            for i in off..off + width {
                let m = enh_re[i] * enh_re[i] + enh_im[i] * enh_im[i];
                if m > peak {
                    peak = m;
                }
            }
            self.band_db[b] = 10.0 * (peak + 1e-12).log10();
            off += width;
        }

        self.frame_synthesis(out, &enh_re, &enh_im);
        self.enh_re = enh_re;
        self.enh_im = enh_im;
    }
}

fn post_filter(nr: &[f32], ni: &[f32], er: &mut [f32], ei: &mut [f32], beta: f32) {
    let beta_p1 = beta + 1.0;
    let eps = 1e-12f32;
    for i in 0..FREQ {
        let em = (er[i] * er[i] + ei[i] * ei[i]).sqrt();
        let nm = (nr[i] * nr[i] + ni[i] * ni[i]).sqrt();
        let mut g = em / (nm + eps);
        g = g.min(1.0).max(eps);
        let gs = g * (g * std::f32::consts::FRAC_PI_2).sin();
        let ratio = g / gs;
        let pf = beta_p1 / (1.0 + beta * ratio * ratio);
        er[i] *= pf;
        ei[i] *= pf;
    }
}
