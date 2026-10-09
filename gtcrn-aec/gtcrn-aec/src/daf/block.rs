//! One DAF block: predict the echo, update the controller GRUs and the
//! Kalman step, then adapt the partitioned filter.

use super::{A_DECAY, Daf, K_ITER, M, N, NB, NBP, NFFT};
use crate::fft::{fill_twiddles, irfft, rfft};
use crate::model::gru_cell;

/// LayerNorm over the six log features; the remaining `k - 6` pass through.
fn cos_ln(inp: &[f32], out: &mut [f32], w: &[f32], b: &[f32], k: usize) {
    let mut mean = 0.0;
    for i in 0..6 {
        mean += inp[i];
    }
    mean /= 6.0;
    let mut var = 0.0;
    for i in 0..6 {
        let d = inp[i] - mean;
        var += d * d;
    }
    var /= 6.0;
    let inv = 1.0 / (var + 1e-5f32).sqrt();
    for i in 0..6 {
        out[i] = (inp[i] - mean) * inv * w[i] + b[i];
    }
    out[6..k].copy_from_slice(&inp[6..k]);
}

/// Predict the block echo spectrum: `y = sum_p H_p * X_p` (complex, per bin).
#[inline(always)]
fn predict(hr: &[f32], hi: &[f32], xr: &[f32], xi: &[f32], yr: &mut [f32], yi: &mut [f32]) {
    yr.fill(0.0);
    yi.fill(0.0);
    // Partition-outer, bin-inner: the bin loop is contiguous and autovectorizes,
    // and each bin still sums in partition order.
    for p in 0..N {
        let hrp = &hr[p * NB..p * NB + NB];
        let hip = &hi[p * NB..p * NB + NB];
        let xrp = &xr[p * NB..p * NB + NB];
        let xip = &xi[p * NB..p * NB + NB];
        for k in 0..NB {
            yr[k] += hrp[k] * xrp[k] - hip[k] * xip[k];
            yi[k] += hrp[k] * xip[k] + hip[k] * xrp[k];
        }
    }
}

/// Scratch for `block`, sized once so the audio path never allocates. It is
/// moved out of `Daf` for the call so the other fields stay borrowable.
#[derive(Default)]
pub(super) struct Scratch {
    fft_re: Vec<f32>,
    fft_im: Vec<f32>,
    xn: Vec<f32>,
    buf: Vec<f32>,
    xr_: Vec<f32>,
    xi_: Vec<f32>,
    yr: Vec<f32>,
    yi: Vec<f32>,
    er: Vec<f32>,
    ei: Vec<f32>,
    dr: Vec<f32>,
    di: Vec<f32>,
    x2: Vec<f32>,
    meanp: Vec<f32>,
    gbin: Vec<f32>,
    e2r: Vec<f32>,
    e2i: Vec<f32>,
    hcr: Vec<f32>,
    hci: Vec<f32>,
    spart: Vec<f32>,
    mu: Vec<f32>,
    fbins: Vec<f32>,
    logbuf: Vec<f32>,
    plog: Vec<f32>,
    // Feature-major GRU inputs and states, 8 bins or partitions per SIMD lane.
    bx8: Vec<f32>,
    bh8: Vec<f32>,
    px8: Vec<f32>,
    ph8: Vec<f32>,
    // Twiddles for the NFFT real FFT and its inner NFFT/2 complex FFT.
    tw_cos: Vec<f32>,
    tw_sin: Vec<f32>,
    tw_h_cos: Vec<f32>,
    tw_h_sin: Vec<f32>,
}

impl Scratch {
    pub(super) fn new() -> Self {
        let mut s = Self {
            fft_re: vec![0.0; NFFT],
            fft_im: vec![0.0; NFFT],
            xn: vec![0.0; NFFT],
            buf: vec![0.0; NFFT],
            xr_: vec![0.0; NB],
            xi_: vec![0.0; NB],
            yr: vec![0.0; NB],
            yi: vec![0.0; NB],
            er: vec![0.0; NB],
            ei: vec![0.0; NB],
            dr: vec![0.0; NB],
            di: vec![0.0; NB],
            x2: vec![0.0; NB],
            meanp: vec![0.0; NB],
            gbin: vec![0.0; NB],
            e2r: vec![0.0; NB],
            e2i: vec![0.0; NB],
            hcr: vec![0.0; NB],
            hci: vec![0.0; NB],
            spart: vec![0.0; N],
            mu: vec![0.0; N * NB],
            fbins: vec![0.0; NB * 10],
            logbuf: vec![0.0; NB * 6],
            plog: vec![0.0; N * 2],
            bx8: vec![0.0; 18 * 8],
            bh8: vec![0.0; 16 * 8],
            px8: vec![0.0; 10 * 8],
            ph8: vec![0.0; 8 * 8],
            tw_cos: vec![0.0; NFFT / 2],
            tw_sin: vec![0.0; NFFT / 2],
            tw_h_cos: vec![0.0; NFFT / 4],
            tw_h_sin: vec![0.0; NFFT / 4],
        };
        fill_twiddles(NFFT, &mut s.tw_cos, &mut s.tw_sin);
        fill_twiddles(NFFT / 2, &mut s.tw_h_cos, &mut s.tw_h_sin);
        s
    }
}

impl Daf {
    pub(super) fn block(&mut self, d_cur: &[f32], x_cur: &[f32], e_out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if ops::simd_tier() >= 2 {
            // SAFETY: AVX was detected at run time.
            unsafe { self.block_avx(d_cur, x_cur, e_out) };
            return;
        }
        self.block_body(d_cur, x_cur, e_out);
    }

    /// The same body compiled for AVX, so its loops vectorize 256 bits wide
    /// instead of SSE2's 128. FMA stays off: every element sees the same
    /// multiplies and adds, and the output is bit-identical to the SSE2 build.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn block_avx(&mut self, d_cur: &[f32], x_cur: &[f32], e_out: &mut [f32]) {
        self.block_body(d_cur, x_cur, e_out);
    }

    #[inline(always)]
    fn block_body(&mut self, d_cur: &[f32], x_cur: &[f32], e_out: &mut [f32]) {
        let nf = NFFT;
        let mut sc = std::mem::take(&mut self.sc);

        sc.xn[..M].copy_from_slice(&self.x_old);
        sc.xn[M..2 * M].copy_from_slice(&x_cur[..M]);
        self.x_old.copy_from_slice(&x_cur[..M]);
        rfft(
            &sc.xn,
            nf,
            &mut sc.xr_,
            &mut sc.xi_,
            &mut sc.fft_re,
            &mut sc.fft_im,
            &sc.tw_cos,
            &sc.tw_sin,
            &sc.tw_h_cos,
            &sc.tw_h_sin,
        );

        // ring push (newest at partition 0)
        self.xr_r.copy_within(0..(N - 1) * NB, NB);
        self.xr_i.copy_within(0..(N - 1) * NB, NB);
        self.xr_r[..NB].copy_from_slice(&sc.xr_);
        self.xr_i[..NB].copy_from_slice(&sc.xi_);

        // predict echo
        predict(
            &self.hr, &self.hi, &self.xr_r, &self.xr_i, &mut sc.yr, &mut sc.yi,
        );
        irfft(
            &sc.yr,
            &sc.yi,
            nf,
            &mut sc.buf,
            &mut sc.fft_re,
            &mut sc.fft_im,
            &sc.tw_cos,
            &sc.tw_sin,
            &sc.tw_h_cos,
            &sc.tw_h_sin,
        );
        for i in 0..M {
            e_out[i] = d_cur[i] - sc.buf[M + i];
        }

        // spectra of e and d (zero-padded first half)
        for v in sc.xn[..M].iter_mut() {
            *v = 0.0;
        }
        sc.xn[M..2 * M].copy_from_slice(&e_out[..M]);
        rfft(
            &sc.xn,
            nf,
            &mut sc.er,
            &mut sc.ei,
            &mut sc.fft_re,
            &mut sc.fft_im,
            &sc.tw_cos,
            &sc.tw_sin,
            &sc.tw_h_cos,
            &sc.tw_h_sin,
        );
        sc.xn[M..2 * M].copy_from_slice(&d_cur[..M]);
        rfft(
            &sc.xn,
            nf,
            &mut sc.dr,
            &mut sc.di,
            &mut sc.fft_re,
            &mut sc.fft_im,
            &sc.tw_cos,
            &sc.tw_sin,
            &sc.tw_h_cos,
            &sc.tw_h_sin,
        );

        // X2 and meanP, summed in partition order like `predict`.
        for v in sc.x2.iter_mut() {
            *v = 0.0;
        }
        for v in sc.meanp.iter_mut() {
            *v = 0.0;
        }
        for p in 0..N {
            let xrp = &self.xr_r[p * NB..p * NB + NB];
            let xip = &self.xr_i[p * NB..p * NB + NB];
            let pp = &self.p[p * NB..p * NB + NB];
            for k in 0..NB {
                sc.x2[k] += xrp[k] * xrp[k] + xip[k] * xip[k];
                sc.meanp[k] += pp[k];
            }
        }
        for v in sc.meanp.iter_mut() {
            *v /= N as f32;
        }

        // Controller features: six log10 powers, staged contiguously for one
        // batched log10, and four normalized correlations.
        let ce = 1e-9f32;
        let eps = 1e-10f32;
        for k in 0..NB {
            let xa = (sc.xr_[k] * sc.xr_[k] + sc.xi_[k] * sc.xi_[k]).sqrt();
            let ea = (sc.er[k] * sc.er[k] + sc.ei[k] * sc.ei[k]).sqrt();
            let ya = (sc.yr[k] * sc.yr[k] + sc.yi[k] * sc.yi[k]).sqrt();
            let da = (sc.dr[k] * sc.dr[k] + sc.di[k] * sc.di[k]).sqrt();
            let lb = &mut sc.logbuf[k * 6..k * 6 + 6];
            lb[0] = xa * xa + eps;
            lb[1] = sc.x2[k] + eps;
            lb[2] = da * da + eps;
            lb[3] = ea * ea + eps;
            lb[4] = ya * ya + eps;
            lb[5] = sc.meanp[k] + eps;
            let f = &mut sc.fbins[k * 10..k * 10 + 10];
            f[6] = (sc.er[k] * sc.dr[k] + sc.ei[k] * sc.di[k]) / (ea * da + ce);
            f[7] = (sc.dr[k] * sc.yr[k] + sc.di[k] * sc.yi[k]) / (da * ya + ce);
            f[8] = (sc.er[k] * sc.yr[k] + sc.ei[k] * sc.yi[k]) / (ea * ya + ce);
            f[9] = (sc.xr_[k] * sc.dr[k] + sc.xi_[k] * sc.di[k]) / (xa * da + ce);
        }
        ops::log10_slice(&mut sc.logbuf);
        for k in 0..NB {
            sc.fbins[k * 10..k * 10 + 6].copy_from_slice(&sc.logbuf[k * 6..k * 6 + 6]);
        }
        // global branch
        let mut fg = [0.0f32; 10];
        for k in 0..NB {
            for j in 0..10 {
                fg[j] += sc.fbins[k * 10 + j];
            }
        }
        for j in 0..10 {
            fg[j] /= NB as f32;
        }
        let mut fb = [0.0f32; 10];
        cos_ln(&fg, &mut fb, &self.g_ln_w, &self.g_ln_b, 10);
        gru_cell(
            &fb,
            &mut self.hg,
            &self.g_wih,
            &self.g_whh,
            &self.g_bih,
            &self.g_bhh,
        );
        // Per-bin controller GRU, 8 bins per SIMD lane. NBP pads NB to 17 groups;
        // pad lanes carry zero input and state and are never read.
        for grp in 0..NBP / 8 {
            for v in sc.bx8.iter_mut() {
                *v = 0.0;
            }
            for v in sc.bh8.iter_mut() {
                *v = 0.0;
            }
            for lane in 0..8 {
                let k = grp * 8 + lane;
                if k >= NB {
                    continue;
                }
                let mut c10 = [0.0f32; 10];
                cos_ln(
                    &sc.fbins[k * 10..k * 10 + 10],
                    &mut c10,
                    &self.b_ln_w,
                    &self.b_ln_b,
                    10,
                );
                for i in 0..10 {
                    sc.bx8[i * 8 + lane] = c10[i];
                }
                for j in 0..8 {
                    sc.bx8[(10 + j) * 8 + lane] = self.hg[j];
                }
                for j in 0..16 {
                    sc.bh8[j * 8 + lane] = self.hb[k * 16 + j];
                }
            }
            ops::gru8(
                &sc.bx8,
                18,
                &mut sc.bh8,
                16,
                &self.b_wih,
                &self.b_whh,
                &self.b_bih,
                &self.b_bhh,
            );
            for lane in 0..8 {
                let k = grp * 8 + lane;
                if k >= NB {
                    continue;
                }
                let mut z = self.head_b[0];
                for j in 0..16 {
                    let hv = sc.bh8[j * 8 + lane];
                    self.hb[k * 16 + j] = hv;
                    z += self.head_w[j] * hv;
                }
                sc.gbin[k] = 2.0 / (1.0 + (-z).exp());
            }
        }
        // Per-partition log10 features (filter power, mean P), batched like the
        // per-bin ones.
        for p in 0..N {
            let base = p * NB;
            let hrp = &self.hr[base..base + NB];
            let hip = &self.hi[base..base + NB];
            let pp = &self.p[base..base + NB];
            let mut h2 = 0.0f32;
            let mut pm = 0.0f32;
            for k in 0..NB {
                h2 += hrp[k] * hrp[k] + hip[k] * hip[k];
                pm += pp[k];
            }
            sc.plog[p * 2] = h2 / NB as f32 + 1e-10;
            sc.plog[p * 2 + 1] = pm / NB as f32 + 1e-10;
        }
        ops::log10_slice(&mut sc.plog);
        // Per-partition controller GRU, 8 partitions per SIMD lane.
        for grp in 0..N / 8 {
            for v in sc.px8.iter_mut() {
                *v = 0.0;
            }
            for v in sc.ph8.iter_mut() {
                *v = 0.0;
            }
            for lane in 0..8 {
                let p = grp * 8 + lane;
                let f2 = [sc.plog[p * 2], sc.plog[p * 2 + 1]];
                let mean = 0.5 * (f2[0] + f2[1]);
                let var = 0.5 * ((f2[0] - mean) * (f2[0] - mean) + (f2[1] - mean) * (f2[1] - mean));
                let inv = 1.0 / (var + 1e-5f32).sqrt();
                sc.px8[lane] = (f2[0] - mean) * inv * self.p_ln_w[0] + self.p_ln_b[0];
                sc.px8[8 + lane] = (f2[1] - mean) * inv * self.p_ln_w[1] + self.p_ln_b[1];
                for j in 0..8 {
                    sc.px8[(2 + j) * 8 + lane] = self.hg[j];
                }
                for j in 0..8 {
                    sc.ph8[j * 8 + lane] = self.hp[p * 8 + j];
                }
            }
            ops::gru8(
                &sc.px8,
                10,
                &mut sc.ph8,
                8,
                &self.p_wih,
                &self.p_whh,
                &self.p_bih,
                &self.p_bhh,
            );
            for lane in 0..8 {
                let p = grp * 8 + lane;
                let mut z = self.p_head_b[0];
                for j in 0..8 {
                    let hv = sc.ph8[j * 8 + lane];
                    self.hp[p * 8 + j] = hv;
                    z += self.p_head_w[j] * hv;
                }
                sc.spart[p] = 2.0 / (1.0 + (-z).exp());
            }
        }

        // Kalman update. The `s > 2.0` step clamp is the rounding-sensitive edge
        // described in docs/gtcrn-aec.md ("DAF numerical note").
        let a2c = A_DECAY * A_DECAY;
        for p in 0..N {
            let sp = sc.spart[p];
            let base = p * NB;
            let prow = &mut self.p[base..base + NB];
            let hrp = &self.hr[base..base + NB];
            let hip = &self.hi[base..base + NB];
            let murow = &mut sc.mu[base..base + NB];
            for k in 0..NB {
                let rt = (sc.er[k] * sc.er[k] + sc.ei[k] * sc.ei[k]) / N as f32;
                let pe = 0.5 * prow[k] * sc.x2[k] + rt;
                let mut m_ = prow[k] / (pe + 1e-10);
                m_ *= sc.gbin[k] * sp;
                let s = m_ * sc.x2[k];
                if s > 2.0 {
                    m_ *= 2.0 / s;
                }
                let fac = (1.0 - 0.5 * m_ * sc.x2[k]).max(0.0);
                prow[k] = (a2c * fac * prow[k] + (1.0 - a2c) * (hrp[k] * hrp[k] + hip[k] * hip[k]))
                    .max(1e-12);
                murow[k] = m_;
            }
        }
        // H += E * mu * conj(X), repeated K_ITER times on the refreshed error.
        for it in 0..K_ITER {
            let (er_ref, ei_ref): (&[f32], &[f32]);
            if it > 0 {
                predict(
                    &self.hr, &self.hi, &self.xr_r, &self.xr_i, &mut sc.yr, &mut sc.yi,
                );
                irfft(
                    &sc.yr,
                    &sc.yi,
                    nf,
                    &mut sc.buf,
                    &mut sc.fft_re,
                    &mut sc.fft_im,
                    &sc.tw_cos,
                    &sc.tw_sin,
                    &sc.tw_h_cos,
                    &sc.tw_h_sin,
                );
                for v in sc.xn[..M].iter_mut() {
                    *v = 0.0;
                }
                for i in 0..M {
                    sc.xn[M + i] = d_cur[i] - sc.buf[M + i];
                }
                rfft(
                    &sc.xn,
                    nf,
                    &mut sc.e2r,
                    &mut sc.e2i,
                    &mut sc.fft_re,
                    &mut sc.fft_im,
                    &sc.tw_cos,
                    &sc.tw_sin,
                    &sc.tw_h_cos,
                    &sc.tw_h_sin,
                );
                er_ref = &sc.e2r;
                ei_ref = &sc.e2i;
            } else {
                er_ref = &sc.er;
                ei_ref = &sc.ei;
            }
            for p in 0..N {
                let base = p * NB;
                let xrp = &self.xr_r[base..base + NB];
                let xip = &self.xr_i[base..base + NB];
                let mup = &sc.mu[base..base + NB];
                let hrp = &mut self.hr[base..base + NB];
                let hip = &mut self.hi[base..base + NB];
                for k in 0..NB {
                    let (xr, xi, m) = (xrp[k], xip[k], mup[k]);
                    hrp[k] += m * (er_ref[k] * xr + ei_ref[k] * xi);
                    hip[k] += m * (ei_ref[k] * xr - er_ref[k] * xi);
                }
            }
        }
        // round-robin overlap-save constraint
        {
            let pi = self.p_idx;
            sc.hcr.copy_from_slice(&self.hr[pi * NB..pi * NB + NB]);
            sc.hci.copy_from_slice(&self.hi[pi * NB..pi * NB + NB]);
            irfft(
                &sc.hcr,
                &sc.hci,
                nf,
                &mut sc.buf,
                &mut sc.fft_re,
                &mut sc.fft_im,
                &sc.tw_cos,
                &sc.tw_sin,
                &sc.tw_h_cos,
                &sc.tw_h_sin,
            );
            for v in sc.buf[M..].iter_mut() {
                *v = 0.0;
            }
            rfft(
                &sc.buf,
                nf,
                &mut sc.hcr,
                &mut sc.hci,
                &mut sc.fft_re,
                &mut sc.fft_im,
                &sc.tw_cos,
                &sc.tw_sin,
                &sc.tw_h_cos,
                &sc.tw_h_sin,
            );
            self.hr[pi * NB..pi * NB + NB].copy_from_slice(&sc.hcr);
            self.hi[pi * NB..pi * NB + NB].copy_from_slice(&sc.hci);
            self.p_idx = (self.p_idx + 1) % N;
        }
        self.sc = sc;
    }
}
