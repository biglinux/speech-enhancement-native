//! DAF (delay-and-filter) AEC front-end — scalar Rust port of LocalVQE
//! `daf_frontend.cpp`. A frequency-domain block Kalman/RLS adaptive filter
//! (N=128 partitions, M=128 block) steered by three tiny GRU controllers, with a
//! GCC-PHAT coarse delay estimate. Produces the error `e` and echo estimate
//! `yhat` that feed the GTCRN core. Ports the reference's own radix-2 FFT so the
//! result is bit-identical.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::gguf::Gguf;

const M: usize = 128;
const N: usize = 128;
const NFFT: usize = 256;
const NB: usize = 129;
const NBP: usize = 136; // NB padded to a multiple of 8 for the batched per-bin GRU
const K_ITER: usize = 2;
const A_DECAY: f32 = 0.999;
const P_INIT: f32 = 1e3;
const G_WIN: usize = 16384;
const G_HOP: usize = 8000;
const G_NFFT: usize = 32768;
const G_MAXLAG: usize = 16384;
const G_GATE_DB: f32 = 26.0;
const G_CONF_THR: f32 = 8.0;
const LOCK_AFTER: i64 = 56000;

// Radix-2 FFT. `twc`/`tws` are the precomputed forward twiddle table (length n/2,
// `twc[m]=cos(-2π m/n)`, `tws[m]=sin(-2π m/n)`); the inverse conjugates it. Replacing
// the old per-butterfly f64 recurrence with a table lookup removes the serial
// dependency (the k-loop autovectorizes) and the f64 math; f32 twiddle error is
// ~1e-7 (>120 dB), far under the DAF's ERLE gate.
#[inline(always)]
fn fft_inplace(re: &mut [f32], im: &mut [f32], inv: bool, twc: &[f32], tws: &[f32]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let h = len / 2;
        let stride = n / len;
        let mut i = 0;
        while i < n {
            for k in 0..h {
                let m = k * stride;
                let cr = twc[m];
                let ci = if inv { -tws[m] } else { tws[m] };
                let ur = re[i + k];
                let ui = im[i + k];
                let br = re[i + k + h];
                let bi = im[i + k + h];
                let vr = br * cr - bi * ci;
                let vi = br * ci + bi * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + h] = ur - vr;
                im[i + k + h] = ui - vi;
            }
            i += len;
        }
        len <<= 1;
    }
    if inv {
        let s = 1.0 / n as f32;
        for i in 0..n {
            re[i] *= s;
            im[i] *= s;
        }
    }
}

/// Fill a forward twiddle table for an `n`-point FFT: `[cos(-2πm/n), sin(-2πm/n)]`
/// for `m in 0..n/2`, computed in f64 then stored f32.
fn fill_twiddles(n: usize, twc: &mut [f32], tws: &mut [f32]) {
    for m in 0..n / 2 {
        let th = -2.0 * std::f64::consts::PI * m as f64 / n as f64;
        twc[m] = th.cos() as f32;
        tws[m] = th.sin() as f32;
    }
}

// `sre`/`sim` are caller-owned scratch of length >= n (moved out of these fns so
// the RT path never allocates). Values are fully overwritten before use.
// Real FFT of `x[0..n]` → half spectrum `outr/outi[0..=n/2]`. Packs the real input
// as n/2 complex points, runs one n/2-point complex FFT (`twc_h`/`tws_h`, the n/2
// twiddle table), then recombines with the n-point table (`twc`/`tws`) — about half
// the work of a full n-point FFT of the zero-padded-imaginary input. `sre`/`sim` are
// scratch of length >= n/2.
#[inline(always)]
fn rfft(
    x: &[f32],
    n: usize,
    outr: &mut [f32],
    outi: &mut [f32],
    sre: &mut [f32],
    sim: &mut [f32],
    twc: &[f32],
    tws: &[f32],
    twc_h: &[f32],
    tws_h: &[f32],
) {
    let n2 = n / 2;
    for j in 0..n2 {
        sre[j] = x[2 * j];
        sim[j] = x[2 * j + 1];
    }
    fft_inplace(&mut sre[..n2], &mut sim[..n2], false, twc_h, tws_h);
    for k in 0..=n2 {
        let (zr, zi) = (sre[k % n2], sim[k % n2]);
        let (mr, mi) = (sre[(n2 - k) % n2], sim[(n2 - k) % n2]);
        let xer = 0.5 * (zr + mr);
        let xei = 0.5 * (zi - mi);
        let xo_r = 0.5 * (zi + mi);
        let xo_i = -0.5 * (zr - mr);
        let (wr, wi) = if k < n2 {
            (twc[k], tws[k])
        } else {
            (-1.0, 0.0)
        };
        outr[k] = xer + wr * xo_r - wi * xo_i;
        outi[k] = xei + wr * xo_i + wi * xo_r;
    }
}

// Inverse of `rfft`: half spectrum `br/bi[0..=n/2]` → real `out[0..n]`. Recovers the
// n/2 complex sequence, one inverse n/2-point FFT, unpack.
#[inline(always)]
fn irfft(
    br: &[f32],
    bi: &[f32],
    n: usize,
    out: &mut [f32],
    sre: &mut [f32],
    sim: &mut [f32],
    twc: &[f32],
    tws: &[f32],
    twc_h: &[f32],
    tws_h: &[f32],
) {
    let n2 = n / 2;
    for k in 0..n2 {
        let (xr, xi) = (br[k], bi[k]);
        let (yr, yi) = (br[n2 - k], bi[n2 - k]); // X[n2-k]
        let xer = 0.5 * (xr + yr);
        let xei = 0.5 * (xi - yi);
        let dr = xr - yr;
        let di = xi + yi;
        let (wr, wi) = (twc[k], tws[k]); // k < n2 here
        // Xo = 0.5 * conj(W_k) * (X[k] - conj(X[n2-k]))
        let xo_r = 0.5 * (wr * dr + wi * di);
        let xo_i = 0.5 * (wr * di - wi * dr);
        // Z[k] = Xe + i*Xo
        sre[k] = xer - xo_i;
        sim[k] = xei + xo_r;
    }
    fft_inplace(&mut sre[..n2], &mut sim[..n2], true, twc_h, tws_h);
    for j in 0..n2 {
        out[2 * j] = sre[j];
        out[2 * j + 1] = sim[j];
    }
}

fn gru_cell(
    x: &[f32],
    nin: usize,
    h: &mut [f32],
    nh: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
) {
    // The bound is a model contract, checked before streaming. Fixed stack
    // storage also works on a data-loop thread different from the init thread.
    assert!(nh <= 16, "DAF GRU hidden size exceeds fixed scratch");
    let mut gi = [0.0f32; 3 * 16];
    let mut gh = [0.0f32; 3 * 16];
    for g in 0..3 * nh {
        let mut s = bih[g];
        let w = &wih[g * nin..g * nin + nin];
        for i in 0..nin {
            s += w[i] * x[i];
        }
        gi[g] = s;
        let mut t = bhh[g];
        let v = &whh[g * nh..g * nh + nh];
        for i in 0..nh {
            t += v[i] * h[i];
        }
        gh[g] = t;
    }
    for i in 0..nh {
        let r = 1.0 / (1.0 + (-(gi[i] + gh[i])).exp());
        let z = 1.0 / (1.0 + (-(gi[nh + i] + gh[nh + i])).exp());
        let n = (gi[2 * nh + i] + r * gh[2 * nh + i]).tanh();
        h[i] = (1.0 - z) * n + z * h[i];
    }
}

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

/// Predict the block echo spectrum: `y = sum_p H_p * X_p` (complex, per bin). One
/// function for the two identical passes in `block` (the main predict and the
/// K_ITER refinement), previously duplicated inline.
#[inline(always)]
fn predict(hr: &[f32], hi: &[f32], xr: &[f32], xi: &[f32], yr: &mut [f32], yi: &mut [f32]) {
    yr.fill(0.0);
    yi.fill(0.0);
    // partition-outer / bin-inner: the k-loop is contiguous and the autovectorizer
    // already takes it to SSE. An explicit AVX complex-MAC kernel was measured to be
    // a wash here (per-partition call overhead offsets the wider lanes), so this
    // stays a plain loop — see docs/gtcrn-aec.md, DAF numerical note.
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

/// Per-call scratch for `block`/`gcc_update`, owned by `Daf` and reused so the RT
/// path allocates nothing. Sized in `reset`. `mem::take`n out at the top of the
/// two callers so `self`'s other fields stay independently borrowable.
#[derive(Default)]
struct Scratch {
    // block-path radix-2 FFT working buffers (length NFFT)
    fft_re: Vec<f32>,
    fft_im: Vec<f32>,
    xn: Vec<f32>,
    buf: Vec<f32>,
    // NB-wide spectra / features
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
    // feature-major GRU batching buffers (8 bins/partitions per SIMD lane)
    bx8: Vec<f32>,
    bh8: Vec<f32>,
    px8: Vec<f32>,
    ph8: Vec<f32>,
    // precomputed forward twiddle tables (length n/2) + half-size tables (length n/4)
    // for the packed real-FFT's inner n/2-point complex transform.
    tw_cos: Vec<f32>,
    tw_sin: Vec<f32>,
    gc_tw_cos: Vec<f32>,
    gc_tw_sin: Vec<f32>,
    tw_h_cos: Vec<f32>,
    tw_h_sin: Vec<f32>,
    gc_tw_h_cos: Vec<f32>,
    gc_tw_h_sin: Vec<f32>,
    // gcc-path (length G_NFFT / G_NFFT/2+1)
    gc_fft_re: Vec<f32>,
    gc_fft_im: Vec<f32>,
    gc_pad: Vec<f32>,
    gc_xr: Vec<f32>,
    gc_xi: Vec<f32>,
    gc_dr: Vec<f32>,
    gc_di: Vec<f32>,
    gc_c: Vec<f32>,
}

#[derive(Clone, Copy, Default)]
struct DelayEstimate {
    shift: i64,
    confidence: f32,
    locked: bool,
    reliable: bool,
    observed_at: i64,
}

struct GccEstimator {
    gcc_sr: Vec<f32>,
    gcc_si: Vec<f32>,
    gcc_maxrms: f32,
    estimate: DelayEstimate,
    measure_reliability: bool,
    sc: Scratch,
}

impl GccEstimator {
    fn new() -> Self {
        let mut sc = Scratch::default();
        let s = &mut sc;
        let nh = G_NFFT / 2 + 1;
        s.gc_fft_re = vec![0.0; G_NFFT];
        s.gc_fft_im = vec![0.0; G_NFFT];
        s.gc_pad = vec![0.0; G_NFFT];
        s.gc_xr = vec![0.0; nh];
        s.gc_xi = vec![0.0; nh];
        s.gc_dr = vec![0.0; nh];
        s.gc_di = vec![0.0; nh];
        s.gc_c = vec![0.0; G_NFFT];
        s.gc_tw_cos = vec![0.0; G_NFFT / 2];
        s.gc_tw_sin = vec![0.0; G_NFFT / 2];
        fill_twiddles(G_NFFT, &mut s.gc_tw_cos, &mut s.gc_tw_sin);
        s.gc_tw_h_cos = vec![0.0; G_NFFT / 4];
        s.gc_tw_h_sin = vec![0.0; G_NFFT / 4];
        fill_twiddles(G_NFFT / 2, &mut s.gc_tw_h_cos, &mut s.gc_tw_h_sin);
        Self {
            gcc_sr: vec![0.0; nh],
            gcc_si: vec![0.0; nh],
            gcc_maxrms: 1e-12,
            estimate: DelayEstimate::default(),
            measure_reliability: false,
            sc,
        }
    }
    fn update(&mut self, mic: &[f32], reference: &[f32], seen: i64) -> DelayEstimate {
        if self.estimate.locked {
            return self.estimate;
        }
        let (w, nf) = (G_WIN, G_NFFT);
        let mut rms = 0.0f32;
        for i in 0..w {
            rms += reference[i] * reference[i];
        }
        rms = (rms / w as f32).sqrt();
        // A decaying peak lets the gate recover after a loud transient. This
        // time constant is per GCC observation (~0.5 s), not per audio sample.
        self.gcc_maxrms = (0.98 * self.gcc_maxrms).max(rms);
        let keep = rms > 1e-9 && rms > self.gcc_maxrms * 10f32.powf(-G_GATE_DB / 20.0);
        if !keep {
            // No new evidence: neither rescore old correlation nor lock a
            // stale hypothesis. Keep the current alignment unchanged.
            return self.estimate;
        }
        let mut sc = std::mem::take(&mut self.sc);
        {
            sc.gc_pad[..w].copy_from_slice(&reference[..w]);
            rfft(
                &sc.gc_pad,
                nf,
                &mut sc.gc_xr,
                &mut sc.gc_xi,
                &mut sc.gc_fft_re,
                &mut sc.gc_fft_im,
                &sc.gc_tw_cos,
                &sc.gc_tw_sin,
                &sc.gc_tw_h_cos,
                &sc.gc_tw_h_sin,
            );
            for (i, v) in sc.gc_pad[..w].iter_mut().enumerate() {
                *v = mic[i];
            }
            rfft(
                &sc.gc_pad,
                nf,
                &mut sc.gc_dr,
                &mut sc.gc_di,
                &mut sc.gc_fft_re,
                &mut sc.gc_fft_im,
                &sc.gc_tw_cos,
                &sc.gc_tw_sin,
                &sc.gc_tw_h_cos,
                &sc.gc_tw_h_sin,
            );
            for k in 0..=nf / 2 {
                let sr = sc.gc_dr[k] * sc.gc_xr[k] + sc.gc_di[k] * sc.gc_xi[k];
                let si = sc.gc_di[k] * sc.gc_xr[k] - sc.gc_dr[k] * sc.gc_xi[k];
                let mag = (sr * sr + si * si).sqrt() + 1e-9;
                self.gcc_sr[k] += sr / mag;
                self.gcc_si[k] += si / mag;
            }
        }
        irfft(
            &self.gcc_sr,
            &self.gcc_si,
            nf,
            &mut sc.gc_c,
            &mut sc.gc_fft_re,
            &mut sc.gc_fft_im,
            &sc.gc_tw_cos,
            &sc.gc_tw_sin,
            &sc.gc_tw_h_cos,
            &sc.gc_tw_h_sin,
        );
        let (mut best, mut peak, mut asum) = (0usize, sc.gc_c[0], 0.0f32);
        for l in 0..G_MAXLAG {
            if sc.gc_c[l] > peak {
                peak = sc.gc_c[l];
                best = l;
            }
            asum += sc.gc_c[l].abs();
        }
        let conf = peak / (asum / G_MAXLAG as f32 + 1e-12);
        // Extra evidence for the opt-in continuous tracker. This is an
        // engineering gate, not an AEC3-equivalent double-talk detector.
        if self.measure_reliability {
            let second = sc.gc_c[..G_MAXLAG]
                .iter()
                .enumerate()
                .filter(|(lag, _)| lag.abs_diff(best) > 32)
                .map(|(_, value)| *value)
                .fold(0.0f32, f32::max);
            self.estimate.reliable = reliable_window(mic, reference, best)
                && peak.is_finite()
                && peak > 1.5 * second.max(1e-12);
        }
        self.estimate.observed_at = seen;
        self.estimate.confidence = conf;
        self.estimate.shift = if conf > G_CONF_THR {
            ((best as i64 - M as i64).max(0) / M as i64) * M as i64
        } else {
            0
        };
        if conf > G_CONF_THR && seen >= LOCK_AFTER {
            self.estimate.locked = true;
        }
        self.sc = sc;
        self.estimate
    }
}

// Use the actual GCC peak, not the deliberately under-shifted DAF alignment,
// to evaluate correlation on coincident samples. All work here is in the worker
// for continuous mode; f64 sums avoid overflow from finite f32 input.
fn reliable_window(mic: &[f32], reference: &[f32], lag: usize) -> bool {
    if lag >= G_WIN / 2 {
        return false;
    } // too little independent overlap
    let mut xy = 0.0f64;
    let mut xx = 0.0f64;
    let mut yy = 0.0f64;
    for i in lag..G_WIN {
        let (x, y) = (f64::from(reference[i - lag]), f64::from(mic[i]));
        if !x.is_finite() || !y.is_finite() || x.abs() >= 0.99 || y.abs() >= 0.99 {
            return false; // don't trust clipped windows
        }
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    let count = (G_WIN - lag) as f64;
    xx / count > 1e-10 && yy / count > 1e-12 && xy * xy > 0.65f64.powi(2) * xx * yy
}

#[derive(Default)]
struct DelayTracker {
    candidate: i64,
    confirmations: u8,
    last_observation: i64,
}
impl DelayTracker {
    fn accept(
        &mut self,
        e: DelayEstimate,
        now: i64,
        current: i64,
        acquired: bool,
        residual_ratio: f32,
    ) -> Option<i64> {
        if e.observed_at <= self.last_observation {
            return None;
        }
        self.last_observation = e.observed_at;
        if now < e.observed_at
            || now - e.observed_at > 2 * G_WIN as i64
            || !e.reliable
            || !e.confidence.is_finite()
            || e.confidence <= G_CONF_THR
        {
            self.confirmations = 0;
            return None;
        }
        if e.shift != self.candidate {
            self.candidate = e.shift;
            self.confirmations = 0;
        }
        self.confirmations = self.confirmations.saturating_add(1);
        if self.confirmations < 3 {
            return None;
        }
        // Correlation alone must not reset a filter that is still cancelling.
        // Conversely, loss of ERLE alone during local speech is insufficient.
        if acquired && ((e.shift - current).abs() < 2 * M as i64 || residual_ratio < 0.5) {
            return None;
        }
        self.confirmations = 0;
        Some(e.shift)
    }
}

// SPSC rendezvous: producer owns EMPTY, worker owns READY, producer reads DONE.
// Release/Acquire on state transfers ownership of ALL payload fields. Samples
// are atomic too, avoiding an UnsafeCell-based shared buffer/unsafe Send impl.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering},
};
const DELAY_EMPTY: u8 = 0;
const DELAY_READY: u8 = 1;
const DELAY_DONE: u8 = 2;
struct DelayMailbox {
    state: AtomicU8,
    stop: AtomicBool,
    mic: Box<[AtomicU32]>,
    reference: Box<[AtomicU32]>,
    seen: AtomicUsize,
    shift: AtomicUsize,
    confidence: AtomicU32,
    locked: AtomicBool,
    reliable: AtomicBool,
}
struct DelayWorker {
    mailbox: Arc<DelayMailbox>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl DelayWorker {
    fn new(mut estimator: GccEstimator, continuous: bool) -> std::io::Result<Self> {
        estimator.measure_reliability = continuous;
        let samples = || {
            (0..G_WIN)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let mailbox = Arc::new(DelayMailbox {
            state: AtomicU8::new(DELAY_EMPTY),
            stop: AtomicBool::new(false),
            mic: samples(),
            reference: samples(),
            seen: AtomicUsize::new(0),
            shift: AtomicUsize::new(0),
            confidence: AtomicU32::new(0),
            locked: AtomicBool::new(false),
            reliable: AtomicBool::new(false),
        });
        let shared = Arc::clone(&mailbox);
        let mut mic = vec![0.0; G_WIN];
        let mut reference = vec![0.0; G_WIN];
        let thread = std::thread::Builder::new()
            .name("aec-coarse-delay".into())
            .spawn(move || {
                while !shared.stop.load(Ordering::Acquire) {
                    if shared.state.load(Ordering::Acquire) != DELAY_READY {
                        // Only the worker sleeps; RT never wakes/notifies a thread
                        // through a syscall. This polling cost is an A/B criterion.
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        continue;
                    }
                    for i in 0..G_WIN {
                        mic[i] = f32::from_bits(shared.mic[i].load(Ordering::Relaxed));
                        reference[i] = f32::from_bits(shared.reference[i].load(Ordering::Relaxed));
                    }
                    let seen = shared.seen.load(Ordering::Relaxed) as i64;
                    if continuous {
                        // A new hypothesis must describe this window, not a sum
                        // permanently dominated by the original echo path.
                        estimator.gcc_sr.fill(0.0);
                        estimator.gcc_si.fill(0.0);
                        estimator.estimate = DelayEstimate::default();
                    }
                    let e = estimator.update(&mic, &reference, seen);
                    shared.shift.store(e.shift as usize, Ordering::Relaxed);
                    shared
                        .confidence
                        .store(e.confidence.to_bits(), Ordering::Relaxed);
                    shared.locked.store(e.locked, Ordering::Relaxed);
                    shared.reliable.store(e.reliable, Ordering::Relaxed);
                    shared.state.store(DELAY_DONE, Ordering::Release);
                    if e.locked && !continuous {
                        break;
                    }
                }
            })?;
        Ok(Self {
            mailbox,
            thread: Some(thread),
        })
    }

    fn submit(&self, mic: &[f32], reference: &[f32], seen: i64) {
        let m = &self.mailbox;
        if m.state.load(Ordering::Acquire) != DELAY_EMPTY {
            return;
        }
        for i in 0..G_WIN {
            m.mic[i].store(mic[i].to_bits(), Ordering::Relaxed);
            m.reference[i].store(reference[i].to_bits(), Ordering::Relaxed);
        }
        m.seen.store(
            usize::try_from(seen).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
        m.state.store(DELAY_READY, Ordering::Release);
    }

    fn poll(&self) -> Option<DelayEstimate> {
        let m = &self.mailbox;
        if m.state.load(Ordering::Acquire) != DELAY_DONE {
            return None;
        }
        let e = DelayEstimate {
            shift: m.shift.load(Ordering::Relaxed) as i64,
            confidence: f32::from_bits(m.confidence.load(Ordering::Relaxed)),
            locked: m.locked.load(Ordering::Relaxed),
            reliable: m.reliable.load(Ordering::Relaxed),
            observed_at: m.seen.load(Ordering::Relaxed) as i64,
        };
        m.state.store(DELAY_EMPTY, Ordering::Release);
        Some(e)
    }
}
impl Drop for DelayWorker {
    fn drop(&mut self) {
        // Owner teardown/reset must run off the data loop, like model loading.
        self.mailbox.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// DAF front-end state + weights.
pub struct Daf {
    // weights
    g_ln_w: Vec<f32>,
    g_ln_b: Vec<f32>,
    g_wih: Vec<f32>,
    g_whh: Vec<f32>,
    g_bih: Vec<f32>,
    g_bhh: Vec<f32>,
    b_ln_w: Vec<f32>,
    b_ln_b: Vec<f32>,
    b_wih: Vec<f32>,
    b_whh: Vec<f32>,
    b_bih: Vec<f32>,
    b_bhh: Vec<f32>,
    p_ln_w: Vec<f32>,
    p_ln_b: Vec<f32>,
    p_wih: Vec<f32>,
    p_whh: Vec<f32>,
    p_bih: Vec<f32>,
    p_bhh: Vec<f32>,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    p_head_w: Vec<f32>,
    p_head_b: Vec<f32>,
    // adaptive filter state
    hr: Vec<f32>,
    hi: Vec<f32>,
    xr_r: Vec<f32>,
    xr_i: Vec<f32>,
    p: Vec<f32>,
    x_old: Vec<f32>,
    p_idx: usize,
    hg: Vec<f32>,
    hb: Vec<f32>,
    hp: Vec<f32>,
    // gcc
    estimator: Option<GccEstimator>,
    delay_worker: Option<DelayWorker>,
    track_delay: bool,
    tracker: DelayTracker,
    residual_ratio: f32,
    transition: usize,
    mic_ring: Vec<f32>,
    ref_ring: Vec<f32>,
    gcc_conf: f32,
    cur_shift: i64,
    gcc_locked: bool,
    n_seen: i64,
    ref_dline: Vec<f32>,
    ref_dpos: usize,
    enable_prealign: bool,
    sc: Scratch,
}

impl Daf {
    pub fn new(gg: &Gguf) -> Option<Self> {
        let g = |n: &str| gg.tensor(n).map(|(d, _)| d.to_vec());
        let mut d = Self {
            g_ln_w: g("daf.glob.norm.ln.weight")?,
            g_ln_b: g("daf.glob.norm.ln.bias")?,
            g_wih: g("daf.glob.gru.weight_ih")?,
            g_whh: g("daf.glob.gru.weight_hh")?,
            g_bih: g("daf.glob.gru.bias_ih")?,
            g_bhh: g("daf.glob.gru.bias_hh")?,
            b_ln_w: g("daf.bins.norm.ln.weight")?,
            b_ln_b: g("daf.bins.norm.ln.bias")?,
            b_wih: g("daf.bins.gru.weight_ih")?,
            b_whh: g("daf.bins.gru.weight_hh")?,
            b_bih: g("daf.bins.gru.bias_ih")?,
            b_bhh: g("daf.bins.gru.bias_hh")?,
            p_ln_w: g("daf.part.ln.weight")?,
            p_ln_b: g("daf.part.ln.bias")?,
            p_wih: g("daf.part.gru.weight_ih")?,
            p_whh: g("daf.part.gru.weight_hh")?,
            p_bih: g("daf.part.gru.bias_ih")?,
            p_bhh: g("daf.part.gru.bias_hh")?,
            head_w: g("daf.head.weight")?,
            head_b: g("daf.head.bias")?,
            p_head_w: g("daf.part.head.weight")?,
            p_head_b: g("daf.part.head.bias")?,
            hr: vec![],
            hi: vec![],
            xr_r: vec![],
            xr_i: vec![],
            p: vec![],
            x_old: vec![],
            p_idx: 0,
            hg: vec![],
            hb: vec![],
            hp: vec![],
            estimator: None,
            delay_worker: None,
            track_delay: false,
            tracker: DelayTracker::default(),
            residual_ratio: 1.0,
            transition: 0,
            mic_ring: vec![],
            ref_ring: vec![],
            gcc_conf: 0.0,
            cur_shift: 0,
            gcc_locked: false,
            n_seen: 0,
            ref_dline: vec![],
            ref_dpos: 0,
            enable_prealign: true,
            sc: Scratch::default(),
        };
        d.reset();
        Some(d)
    }

    pub fn reset(&mut self) {
        self.hr = vec![0.0; N * NB];
        self.hi = vec![0.0; N * NB];
        self.xr_r = vec![0.0; N * NB];
        self.xr_i = vec![0.0; N * NB];
        self.p = vec![P_INIT; N * NB];
        self.x_old = vec![0.0; M];
        self.p_idx = 0;
        self.hg = vec![0.0; 8];
        self.hb = vec![0.0; NBP * 16];
        self.hp = vec![0.0; N * 8];
        // Reset is an initialization/control operation, never a data-loop call.
        self.delay_worker = None; // stop/join any previous estimator off RT
        self.track_delay = false;
        self.tracker = DelayTracker::default();
        self.residual_ratio = 1.0;
        self.transition = 0;
        self.estimator = Some(GccEstimator::new());
        self.mic_ring = vec![0.0; G_WIN];
        self.ref_ring = vec![0.0; G_WIN];
        self.gcc_conf = 0.0;
        self.cur_shift = 0;
        self.gcc_locked = false;
        self.n_seen = 0;
        self.ref_dline = vec![0.0; G_MAXLAG + M];
        self.ref_dpos = 0;

        let s = &mut self.sc;
        s.fft_re = vec![0.0; NFFT];
        s.fft_im = vec![0.0; NFFT];
        s.xn = vec![0.0; NFFT];
        s.buf = vec![0.0; NFFT];
        s.xr_ = vec![0.0; NB];
        s.xi_ = vec![0.0; NB];
        s.yr = vec![0.0; NB];
        s.yi = vec![0.0; NB];
        s.er = vec![0.0; NB];
        s.ei = vec![0.0; NB];
        s.dr = vec![0.0; NB];
        s.di = vec![0.0; NB];
        s.x2 = vec![0.0; NB];
        s.meanp = vec![0.0; NB];
        s.gbin = vec![0.0; NB];
        s.e2r = vec![0.0; NB];
        s.e2i = vec![0.0; NB];
        s.hcr = vec![0.0; NB];
        s.hci = vec![0.0; NB];
        s.spart = vec![0.0; N];
        s.mu = vec![0.0; N * NB];
        s.fbins = vec![0.0; NB * 10];
        s.logbuf = vec![0.0; NB * 6];
        s.plog = vec![0.0; N * 2];
        s.bx8 = vec![0.0; 18 * 8];
        s.bh8 = vec![0.0; 16 * 8];
        s.px8 = vec![0.0; 10 * 8];
        s.ph8 = vec![0.0; 8 * 8];
        s.tw_cos = vec![0.0; NFFT / 2];
        s.tw_sin = vec![0.0; NFFT / 2];
        fill_twiddles(NFFT, &mut s.tw_cos, &mut s.tw_sin);
        // half-size tables for the inner n/2-point complex FFT of the real transform
        s.tw_h_cos = vec![0.0; NFFT / 4];
        s.tw_h_sin = vec![0.0; NFFT / 4];
        fill_twiddles(NFFT / 2, &mut s.tw_h_cos, &mut s.tw_h_sin);
    }

    /// Opt-in, initialization only. The worker never touches adaptive weights.
    /// A full mailbox drops an observation rather than waiting in the callback.
    pub fn enable_async_delay(&mut self) -> std::io::Result<()> {
        self.enable_async_delay_mode(false)
    }

    /// Experimental continuous acquisition; requires the worker and corpus A/B.
    pub fn enable_delay_tracking(&mut self) -> std::io::Result<()> {
        self.enable_async_delay_mode(true)
    }

    fn enable_async_delay_mode(&mut self, continuous: bool) -> std::io::Result<()> {
        if self.delay_worker.is_some() {
            return if self.track_delay == continuous {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "delay mode already selected",
                ))
            };
        }
        if self.n_seen != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "async delay must be selected before streaming",
            ));
        }
        let estimator = self.estimator.take().expect("initialized estimator");
        match DelayWorker::new(estimator, continuous) {
            Ok(worker) => {
                self.delay_worker = Some(worker);
                self.track_delay = continuous;
                Ok(())
            }
            Err(error) => {
                self.estimator = Some(GccEstimator::new());
                Err(error)
            }
        }
    }

    fn gcc_update(&mut self) {
        if self.gcc_locked && !self.track_delay {
            return;
        }
        if let Some(worker) = &self.delay_worker {
            worker.submit(&self.mic_ring, &self.ref_ring, self.n_seen);
        } else if let Some(estimator) = &mut self.estimator {
            // prime_delay intentionally reopens the lock between observations.
            estimator.estimate.locked = self.gcc_locked;
            let e = estimator.update(&self.mic_ring, &self.ref_ring, self.n_seen);
            self.cur_shift = e.shift;
            self.gcc_conf = e.confidence;
            self.gcc_locked = e.locked;
        }
    }

    fn block(&mut self, d_cur: &[f32], x_cur: &[f32], e_out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if dfn_ops::simd_tier() >= 2 {
            // SAFETY: AVX was detected at run time.
            unsafe { self.block_avx(d_cur, x_cur, e_out) };
            return;
        }
        self.block_body(d_cur, x_cur, e_out);
    }

    /// The same body compiled for AVX: its loops autovectorize 256 bits wide
    /// instead of the baseline's 128-bit SSE2 (the block was ~31% of the streaming
    /// profile on the AVX-only i3). No FMA is enabled, so every element sees the
    /// same multiplies and adds and the output is bit-identical.
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

        // predict echo — partition-outer / bin-inner so the bin loop is contiguous
        // and autovectorizes; the per-bin sum still runs in partition order, so the
        // result matches the reference (golden SDR>60).
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

        // X2 + meanP — partition-outer / bin-inner (contiguous), same sum order.
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

        // controller features. The six log-features are staged into a contiguous
        // buffer (+1e-10, matching log10e) and log10'd in one batched SIMD pass; the
        // four ratios go straight into fbins.
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
        dfn_ops::log10_slice(&mut sc.logbuf);
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
            10,
            &mut self.hg,
            8,
            &self.g_wih,
            &self.g_whh,
            &self.g_bih,
            &self.g_bhh,
        );
        // per-bin controller GRU — 8 bins per SIMD lane (NBP=136=17*8 pads NB=129;
        // pad lanes carry zero input/state, finite outputs, and are never read).
        // Quality-equivalent to the scalar path (aec-eval); the DAF's sample
        // trajectory forks on a 1-ulp Kalman knife-edge regardless (docs/gtcrn-aec.md),
        // so ERLE, not sample identity, is the gate.
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
            dfn_ops::gru8(
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
        // per-partition log-features (h2, meanP) staged contiguously and log10'd in
        // one SIMD pass (was 2·N scalar log10 calls per block).
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
        dfn_ops::log10_slice(&mut sc.plog);
        // per-partition controller GRU — 8 partitions per SIMD lane (N=128=16*8).
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
            dfn_ops::gru8(
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

        // Kalman update. Per-partition slices hoist the base pointer and drop the
        // per-element bounds checks; the `s > 2.0` branch keeps this scalar (it
        // cannot autovectorize), but the branch is the point (it is the knife-edge,
        // §9) so it stays a plain loop.
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
        // H += E * mu * conj(X), with K_ITER data-reuse repeats. Per-partition slices
        // → no branch, no bounds check → the k-loop autovectorizes (SSE).
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

    // Reset only alignment-dependent state, in place. Do NOT call reset():
    // that method allocates and may join a worker. This leaves neural state,
    // raw history, delay line and worker ownership intact.
    fn clear_adaptation(&mut self) {
        self.hr.fill(0.0);
        self.hi.fill(0.0);
        self.xr_r.fill(0.0);
        self.xr_i.fill(0.0);
        self.p.fill(P_INIT);
        self.x_old.fill(0.0);
        self.hg.fill(0.0);
        self.hb.fill(0.0);
        self.hp.fill(0.0);
        self.p_idx = 0;
        self.residual_ratio = 1.0;
    }

    /// Process `hop` samples (a multiple of M): mic+ref -> e (echo removed) + yhat.
    pub fn process(
        &mut self,
        mic: &[f32],
        r#ref: &[f32],
        hop: usize,
        e_out: &mut [f32],
        yhat_out: &mut [f32],
    ) {
        let dl = self.ref_dline.len();
        let mut o = 0;
        while o < hop {
            if let Some(e) = self.delay_worker.as_ref().and_then(DelayWorker::poll) {
                if self.track_delay {
                    if let Some(shift) = self.tracker.accept(
                        e,
                        self.n_seen,
                        self.cur_shift,
                        self.gcc_locked,
                        self.residual_ratio,
                    ) {
                        if shift != self.cur_shift {
                            self.clear_adaptation();
                            self.cur_shift = shift;
                            self.transition = 10 * M;
                        }
                        self.gcc_conf = e.confidence;
                        self.gcc_locked = true;
                    }
                } else {
                    self.cur_shift = e.shift;
                    self.gcc_conf = e.confidence;
                    self.gcc_locked = e.locked;
                }
            }
            // A locked/off estimator no longer reads these 16k histories.
            // The reference delay line below remains active for filtering.
            if self.enable_prealign && (!self.gcc_locked || self.track_delay) {
                self.mic_ring.copy_within(M.., 0);
                self.mic_ring[G_WIN - M..].copy_from_slice(&mic[o..o + M]);
                self.ref_ring.copy_within(M.., 0);
                self.ref_ring[G_WIN - M..].copy_from_slice(&r#ref[o..o + M]);
            }
            for i in 0..M {
                self.ref_dline[(self.ref_dpos + i) % dl] = r#ref[o + i];
            }
            self.n_seen += M as i64;
            if self.enable_prealign
                && self.n_seen >= G_WIN as i64
                && ((self.n_seen - G_WIN as i64) % G_HOP as i64) < M as i64
            {
                self.gcc_update();
            }
            let mut xs = [0.0f32; M];
            for i in 0..M {
                let idx = self.ref_dpos as i64 + i as i64 - self.cur_shift;
                xs[i] = if self.n_seen - M as i64 + i as i64 >= self.cur_shift {
                    self.ref_dline[(idx.rem_euclid(dl as i64)) as usize]
                } else {
                    0.0
                };
            }
            self.ref_dpos = (self.ref_dpos + M) % dl;
            self.block(&mic[o..o + M], &xs, &mut e_out[o..o + M]);
            if self.track_delay {
                let energy = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
                let ratio =
                    (energy(&e_out[o..o + M]) / (energy(&mic[o..o + M]) + 1e-12)).clamp(0.0, 10.0);
                self.residual_ratio = 0.98 * self.residual_ratio + 0.02 * ratio as f32;
                for i in o..o + M {
                    if self.transition == 0 {
                        break;
                    }
                    let wet = 1.0 - self.transition as f32 / (10 * M) as f32;
                    e_out[i] = mic[i] + wet * (e_out[i] - mic[i]);
                    self.transition -= 1;
                }
            }
            o += M;
        }
        for i in 0..hop {
            yhat_out[i] = mic[i] - e_out[i];
        }
    }

    /// Whole-clip GCC to freeze the bulk delay before filtering (removes onset).
    pub fn prime_delay(&mut self, mic: &[f32], r#ref: &[f32], n: usize) {
        self.reset();
        let (mut best_shift, mut best_conf) = (0i64, 0.0f32);
        let mut o = 0;
        while o + M <= n {
            self.mic_ring.copy_within(M.., 0);
            self.mic_ring[G_WIN - M..].copy_from_slice(&mic[o..o + M]);
            self.ref_ring.copy_within(M.., 0);
            self.ref_ring[G_WIN - M..].copy_from_slice(&r#ref[o..o + M]);
            self.n_seen += M as i64;
            if self.n_seen >= G_WIN as i64
                && ((self.n_seen - G_WIN as i64) % G_HOP as i64) < M as i64
            {
                self.gcc_update();
                self.gcc_locked = false;
                if self.gcc_conf > best_conf {
                    best_conf = self.gcc_conf;
                    best_shift = self.cur_shift;
                }
            }
            o += M;
        }
        self.reset();
        if best_conf > G_CONF_THR {
            self.cur_shift = best_shift;
            self.gcc_locked = true;
        }
    }
}

#[cfg(test)]
mod rfft_tests {
    use super::{fft_inplace, fill_twiddles, irfft, rfft};

    // Naive DFT ground truth for real input: X[k] = Σ_t x[t] e^{-2πi k t/n}.
    fn naive(x: &[f32], n: usize, k: usize) -> (f64, f64) {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for t in 0..n {
            let a = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += x[t] as f64 * a.cos();
            im += x[t] as f64 * a.sin();
        }
        (re, im)
    }

    fn tables(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut c, mut s) = (vec![0.0; n / 2], vec![0.0; n / 2]);
        let (mut hc, mut hs) = (vec![0.0; n / 4], vec![0.0; n / 4]);
        fill_twiddles(n, &mut c, &mut s);
        fill_twiddles(n / 2, &mut hc, &mut hs);
        (c, s, hc, hs)
    }

    #[test]
    fn rfft_matches_naive_and_round_trips() {
        for &n in &[256usize, 512, 32768] {
            let (c, s, hc, hs) = tables(n);
            let mut seed = 12345u32;
            let mut g = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / 16_777_216.0 - 0.5
            };
            let x: Vec<f32> = (0..n).map(|_| g()).collect();
            let (mut outr, mut outi) = (vec![0.0f32; n / 2 + 1], vec![0.0f32; n / 2 + 1]);
            let (mut sre, mut sim) = (vec![0.0f32; n], vec![0.0f32; n]);
            rfft(
                &x, n, &mut outr, &mut outi, &mut sre, &mut sim, &c, &s, &hc, &hs,
            );
            // spectrum vs naive DFT (scale tolerance with n: bigger n accumulates more f32 error)
            let tol = 2e-3 * n as f64;
            for k in 0..=n / 2 {
                let (re, im) = naive(&x, n, k);
                assert!(
                    (outr[k] as f64 - re).abs() < tol && (outi[k] as f64 - im).abs() < tol,
                    "n={n} k={k}: ({},{}) vs naive ({re:.2},{im:.2})",
                    outr[k],
                    outi[k]
                );
            }
            // round-trip irfft(rfft(x)) == x
            let mut back = vec![0.0f32; n];
            irfft(
                &outr, &outi, n, &mut back, &mut sre, &mut sim, &c, &s, &hc, &hs,
            );
            let maxd = x
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(maxd < 1e-4, "n={n} round-trip max diff {maxd:e}");
        }
    }

    // The half-spectrum irfft must invert a Hermitian spectrum the same way the old
    // full-size complex IFFT did (real output), so DAF behaviour is unchanged.
    #[test]
    fn irfft_matches_full_complex_ifft() {
        let n = 256;
        let (c, s, hc, hs) = tables(n);
        let mut seed = 7u32;
        let mut g = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16_777_216.0 - 0.5
        };
        let br: Vec<f32> = (0..=n / 2).map(|_| g()).collect();
        let mut bi: Vec<f32> = (0..=n / 2).map(|_| g()).collect();
        bi[0] = 0.0;
        bi[n / 2] = 0.0; // Hermitian: DC and Nyquist are real
        // reference: build the full n-point Hermitian spectrum and inverse-FFT it
        let (mut fr, mut fi) = (vec![0.0f32; n], vec![0.0f32; n]);
        fr[..=n / 2].copy_from_slice(&br);
        fi[..=n / 2].copy_from_slice(&bi);
        for k in 1..n / 2 {
            fr[n - k] = br[k];
            fi[n - k] = -bi[k];
        }
        fft_inplace(&mut fr, &mut fi, true, &c, &s);
        // our half-spectrum irfft
        let mut out = vec![0.0f32; n];
        let (mut sre, mut sim) = (vec![0.0f32; n], vec![0.0f32; n]);
        irfft(&br, &bi, n, &mut out, &mut sre, &mut sim, &c, &s, &hc, &hs);
        let maxd = out
            .iter()
            .zip(&fr)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxd < 1e-4, "irfft vs full IFFT max diff {maxd:e}");
    }
}

#[cfg(test)]
mod tracking_tests {
    use super::*;
    #[test]
    fn tracking_requires_fresh_reliable_persistent_evidence_and_loss() {
        let mut tracker = DelayTracker::default();
        let mut e = DelayEstimate {
            shift: 1024,
            confidence: 20.0,
            reliable: true,
            observed_at: G_WIN as i64,
            locked: true,
        };
        for _ in 0..2 {
            assert_eq!(tracker.accept(e, e.observed_at, 0, true, 1.0), None);
            e.observed_at += G_HOP as i64;
        }
        assert_eq!(tracker.accept(e, e.observed_at, 0, true, 1.0), Some(1024));
        assert_eq!(tracker.accept(e, e.observed_at, 0, true, 1.0), None); // replay
        for _ in 0..4 {
            e.observed_at += G_HOP as i64;
            assert_eq!(tracker.accept(e, e.observed_at, 0, true, 0.1), None);
        }
        e.observed_at += G_HOP as i64;
        e.reliable = false;
        assert_eq!(tracker.accept(e, e.observed_at, 0, true, 1.0), None);
        e.observed_at += G_HOP as i64;
        e.reliable = true;
        assert_eq!(
            tracker.accept(e, e.observed_at + 3 * G_WIN as i64, 0, true, 1.0),
            None
        );
    }
    #[test]
    fn silent_and_clipped_observations_are_not_delay_evidence() {
        assert!(!reliable_window(&[0.0; G_WIN], &[0.0; G_WIN], 0));
        assert!(!reliable_window(&[1.0; G_WIN], &[1.0; G_WIN], 0));
    }
}
