//! STFT / iSTFT as the model stores them: pre-baked cos/sin analysis and
//! synthesis matrices (window folded in), matching LocalVQE `gtcrn.cpp`.
//! N = 512, hop = 256, 257 bins, centered with reflect padding.

pub const N_FFT: usize = 512;
pub const HOP: usize = 256;
pub const N_FREQ: usize = 257;

/// Frames of a centered STFT. `L` should be a multiple of `HOP`.
#[must_use]
pub fn n_frames(l: usize) -> usize {
    l / HOP + 1
}

fn reflect_pad(sig: &[f32]) -> Vec<f32> {
    let l = sig.len();
    let mut pad = vec![0.0f32; l + N_FFT];
    pad[HOP..HOP + l].copy_from_slice(sig);
    if l == 0 {
        return pad;
    }
    if l == 1 {
        pad.fill(sig[0]);
        return pad;
    }
    if l <= HOP {
        // Repeated reflect padding, excluding the endpoints: the direct
        // indices below need l > HOP.
        let period = (2 * (l - 1)) as isize;
        for (i, sample) in pad.iter_mut().enumerate() {
            let k = (i as isize - HOP as isize).rem_euclid(period) as usize;
            *sample = sig[if k < l { k } else { period as usize - k }];
        }
        return pad;
    }
    for j in 0..HOP {
        pad[j] = sig[HOP - j];
        pad[HOP + l + j] = sig[l - 2 - j];
    }
    pad
}

/// Analysis STFT. `wcos`/`wsin` are `stft.wcos`/`stft.wsin` (shape 257×512,
/// window baked in). Returns time-major `re`/`im`, each `T*N_FREQ`.
#[must_use]
pub fn stft(sig: &[f32], wcos: &[f32], wsin: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let l = sig.len();
    let t_n = n_frames(l);
    let pad = reflect_pad(sig);
    let mut re = vec![0.0f32; t_n * N_FREQ];
    let mut im = vec![0.0f32; t_n * N_FREQ];
    for t in 0..t_n {
        let fr = &pad[t * HOP..t * HOP + N_FFT];
        let o = t * N_FREQ;
        stft_frame(
            fr,
            wcos,
            wsin,
            &mut re[o..o + N_FREQ],
            &mut im[o..o + N_FREQ],
        );
    }
    (re, im)
}

/// One analysis frame: 512 samples (caller owns the sliding window) -> 257 re/im.
/// No padding/envelope — the streaming caller does overlap-add. Matches the
/// reference `stft_frame` (window baked into `wcos`/`wsin`).
pub fn stft_frame(fr: &[f32], wcos: &[f32], wsin: &[f32], re: &mut [f32], im: &mut [f32]) {
    for f in 0..N_FREQ {
        let wc = &wcos[f * N_FFT..f * N_FFT + N_FFT];
        let ws = &wsin[f * N_FFT..f * N_FFT + N_FFT];
        re[f] = dfn_ops::vdot_f32(wc, fr);
        im[f] = dfn_ops::vdot_f32(ws, fr);
    }
}

/// One synthesis frame: 257 re/im -> 512 samples (pre-overlap-add). The caller
/// overlap-adds and divides by the window-power envelope (`win2`).
pub fn istft_frame(re: &[f32], im: &[f32], icos: &[f32], isin: &[f32], ft: &mut [f32]) {
    for n in 0..N_FFT {
        let row = &icos[n * N_FREQ..n * N_FREQ + N_FREQ];
        let rows = &isin[n * N_FREQ..n * N_FREQ + N_FREQ];
        ft[n] = dfn_ops::vdot_f32(row, re) + dfn_ops::vdot_f32(rows, im);
    }
}

/// The baked matrices are a sine-windowed 512-point real DFT: `wcos = w·cos`,
/// `wsin = -w·sin`, and the synthesis rows are `w/N` times the Hermitian inverse.
/// This runs them as a real FFT, ~50x fewer multiply-adds per hop than the two
/// 257x512 matrix products. Built only when the loaded model's matrices match
/// that form, so any other model keeps the matrix path. Plans and scratch are
/// allocated here, never in the audio callback.
pub struct RealFft {
    forward: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    inverse: std::sync::Arc<dyn realfft::ComplexToReal<f32>>,
    win: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<realfft::num_complex::Complex<f32>>,
    scratch: Vec<realfft::num_complex::Complex<f32>>,
}

impl RealFft {
    /// `None` when the matrices are not the windowed DFT this implements.
    #[must_use]
    pub fn from_matrices(wcos: &[f32], wsin: &[f32], icos: &[f32], isin: &[f32]) -> Option<Self> {
        if wcos.len() != N_FREQ * N_FFT
            || wsin.len() != wcos.len()
            || icos.len() != wcos.len()
            || isin.len() != wcos.len()
        {
            return None;
        }
        let win: Vec<f32> = wcos[..N_FFT].to_vec();
        let step = 2.0 * std::f64::consts::PI / N_FFT as f64;
        for f in 0..N_FREQ {
            let k = if f == 0 || f == N_FREQ - 1 { 1.0 } else { 2.0 };
            for n in 0..N_FFT {
                let (s, c) = (step * ((f * n) % N_FFT) as f64).sin_cos();
                let w = f64::from(win[n]);
                let a = w / N_FFT as f64 * k;
                let close = |got: f32, want: f64, scale: f64| {
                    (f64::from(got) - want).abs() <= 1e-6 * scale.max(1e-3)
                };
                if !close(wcos[f * N_FFT + n], w * c, 1.0)
                    || !close(wsin[f * N_FFT + n], -w * s, 1.0)
                    || !close(icos[n * N_FREQ + f], a * c, 2.0 / N_FFT as f64)
                    || !close(isin[n * N_FREQ + f], -a * s, 2.0 / N_FFT as f64)
                {
                    return None;
                }
            }
        }
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(N_FFT);
        let inverse = planner.plan_fft_inverse(N_FFT);
        let scratch_len = forward.get_scratch_len().max(inverse.get_scratch_len());
        Some(Self {
            spectrum: forward.make_output_vec(),
            scratch: vec![realfft::num_complex::Complex::default(); scratch_len],
            frame: vec![0.0; N_FFT],
            forward,
            inverse,
            win,
        })
    }

    /// Same result as [`stft_frame`] (to FFT rounding).
    pub fn analyze(&mut self, fr: &[f32], re: &mut [f32], im: &mut [f32]) {
        for ((d, &x), &w) in self.frame.iter_mut().zip(fr).zip(&self.win) {
            *d = x * w;
        }
        let _ = self.forward.process_with_scratch(
            &mut self.frame,
            &mut self.spectrum,
            &mut self.scratch,
        );
        for (f, c) in self.spectrum.iter().enumerate() {
            re[f] = c.re;
            im[f] = c.im;
        }
    }

    /// Same result as [`istft_frame`] (to FFT rounding).
    pub fn synthesize(&mut self, re: &[f32], im: &[f32], ft: &mut [f32]) {
        for (f, c) in self.spectrum.iter_mut().enumerate() {
            *c = realfft::num_complex::Complex::new(re[f], im[f]);
        }
        // The synthesis rows weigh the DC and Nyquist imaginary parts by sin(0)
        // and sin(pi n): zero, as the inverse real FFT requires.
        self.spectrum[0].im = 0.0;
        self.spectrum[N_FREQ - 1].im = 0.0;
        let _ = self
            .inverse
            .process_with_scratch(&mut self.spectrum, ft, &mut self.scratch);
        let scale = 1.0 / N_FFT as f32;
        for (v, &w) in ft.iter_mut().zip(&self.win) {
            *v *= w * scale;
        }
    }
}

/// Synthesis iSTFT with overlap-add normalized by the window-power envelope.
/// `icos`/`isin` are `stft.icos`/`stft.isin` (shape 512×257), `win2` is `stft.win2`.
#[must_use]
pub fn istft(
    re: &[f32],
    im: &[f32],
    t_n: usize,
    l: usize,
    icos: &[f32],
    isin: &[f32],
    win2: &[f32],
) -> Vec<f32> {
    let mut ytmp = vec![0.0f32; l + N_FFT];
    let mut wenv = vec![0.0f32; l + N_FFT];
    let mut ft = vec![0.0f32; N_FFT];
    for t in 0..t_n {
        ft.iter_mut().for_each(|v| *v = 0.0);
        for n in 0..N_FFT {
            let row = &icos[n * N_FREQ..n * N_FREQ + N_FREQ];
            let rows = &isin[n * N_FREQ..n * N_FREQ + N_FREQ];
            let mut v = 0.0f32;
            for f in 0..N_FREQ {
                v += row[f] * re[t * N_FREQ + f] + rows[f] * im[t * N_FREQ + f];
            }
            ft[n] = v;
        }
        for n in 0..N_FFT {
            ytmp[t * HOP + n] += ft[n];
            wenv[t * HOP + n] += win2[n];
        }
    }
    let mut y = vec![0.0f32; l];
    for i in 0..l {
        let w = wenv[HOP + i];
        y[i] = if w > 1e-11 { ytmp[HOP + i] / w } else { 0.0 };
    }
    y
}

#[cfg(test)]
mod padding_tests {
    use super::*;
    #[test]
    fn short_clips_have_defined_padding() {
        assert!(reflect_pad(&[]).iter().all(|&x| x == 0.0));
        assert!(reflect_pad(&[0.25]).iter().all(|&x| x == 0.25));
        for len in [2, 3, 128, 256, 257, 512, 1024] {
            let x: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let y = reflect_pad(&x);
            assert_eq!(&y[HOP..HOP + len], &x);
            assert_eq!(y[HOP - 1], x[1]);
            assert_eq!(y[HOP + len], x[len - 2]);
            if len > HOP {
                for j in 0..HOP {
                    assert_eq!(y[j], x[HOP - j]);
                    assert_eq!(y[HOP + len + j], x[len - 2 - j]);
                }
            }
        }
    }
}
