//! Native Rust port of GTCRN-AEC (the compact ~49K GTCRN acoustic echo canceller
//! from LocalVQE, Apache-2.0). Chosen over WebRTC/DTLN/SpeexDSP by the `aec-eval`
//! bench for best near-end voice preservation at ultralow compute.
//!
//! Ported from LocalVQE `ggml/gtcrn.cpp` (core) + `ggml/daf_frontend.cpp` (AEC
//! front-end): STFT-as-matmul, ERB sub-band, GT-conv encoder, dual-path grouped
//! RNN, decoder, and the DAF adaptive front-end. Offline (`run_aec`) matches the
//! reference to 8.5e-5; the streaming path (`Streamer`/`run_aec_stream`) matches
//! `localvqe --stream` to 7e-5 and the core is bit-exact vs the batch reference.
//! Resamplers bridge PipeWire's 48 kHz to the model's 16 kHz. See `docs/gtcrn-aec.md`.

pub mod daf;
pub mod erb;
pub mod gguf;
pub mod hbaec;
pub mod model;
pub mod resample;
pub mod stft;

use gguf::Gguf;

/// Full offline AEC: mic + far-end reference -> echo-cancelled mic.
/// Mirrors LocalVQE `localvqe_process_f32` file mode: prime the bulk delay,
/// run the DAF adaptive front-end, AGC-normalise, GTCRN mask, iSTFT, un-gain.
#[must_use]
pub fn run_aec(m: &Model, mic: &[f32], reference: &[f32]) -> Vec<f32> {
    const MBLK: usize = 128; // DAF block
    let n = (mic.len().min(reference.len()) / MBLK) * MBLK;
    let mut out = vec![0.0f32; mic.len()];
    if n == 0 {
        return out;
    }
    let mut daf = daf::Daf::new(&m.w).expect("daf weights");
    daf.reset();
    daf.prime_delay(&mic[..n], &reference[..n], n);
    let mut e = vec![0.0f32; n];
    let mut yh = vec![0.0f32; n];
    daf.process(&mic[..n], &reference[..n], n, &mut e, &mut yh);

    // AGC: scale the error to ~0.05 RMS so the net sees a consistent level.
    let ms: f64 = e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n as f64;
    let gain = (0.05 / (ms.sqrt() as f32 + 1e-6)).clamp(0.05, 50.0);
    for i in 0..n {
        e[i] *= gain;
        yh[i] *= gain;
    }

    let (wcos, _) = m.w.tensor("stft.wcos").unwrap();
    let (wsin, _) = m.w.tensor("stft.wsin").unwrap();
    let (icos, _) = m.w.tensor("stft.icos").unwrap();
    let (isin, _) = m.w.tensor("stft.isin").unwrap();
    let (win2, _) = m.w.tensor("stft.win2").unwrap();
    let t_n = stft::n_frames(n);
    let spec_e = stft_freq_major(&e, wcos, wsin, t_n);
    let spec_y = stft_freq_major(&yh, wcos, wsin, t_n);
    let enh = model::forward(&m.w, &spec_e, &spec_y, t_n);
    let (re, im) = freq_major_to_time(&enh, t_n);
    let y = stft::istft(&re, &im, t_n, n, icos, isin, win2);
    let inv = 1.0 / gain;
    for i in 0..n {
        out[i] = y[i] * inv;
    }
    out
}

/// Live per-hop AEC (16 kHz, 256-sample hop), mirroring LocalVQE
/// `process_gtcrn_frame`: DAF hop → running-RMS AGC → 512 window → STFT frame →
/// streaming GTCRN → iSTFT frame → un-gain → 50% overlap-add. Recurrent state is
/// carried across `process_hop` calls — this is the path a PipeWire AEC plugin drives.
pub struct Streamer {
    daf: daf::Daf,
    core: model::StreamCore,
    buf_e: [f32; 512],
    buf_y: [f32; 512],
    acc: [f32; 512],
    wenv: [f32; 512],
    pow: f32,
    pow_init: bool,
    faulted: bool, // latch: no allocations/rebuild in a failed audio callback
    fft: Option<stft::RealFft>,
}

impl Streamer {
    pub const HOP: usize = 256;

    #[must_use]
    pub fn new(m: &Model) -> Self {
        let _ = dfn_ops::simd_tier(); // resolve ISA dispatch off the callback
        Self {
            daf: daf::Daf::new(&m.w).expect("daf weights"),
            core: model::StreamCore::prepared(&m.w),
            buf_e: [0.0; 512],
            buf_y: [0.0; 512],
            acc: [0.0; 512],
            wenv: [0.0; 512],
            pow: 0.0,
            pow_init: false,
            faulted: false,
            fft: stft::RealFft::from_matrices(
                m.w.tensor("stft.wcos").map_or(&[], |t| t.0),
                m.w.tensor("stft.wsin").map_or(&[], |t| t.0),
                m.w.tensor("stft.icos").map_or(&[], |t| t.0),
                m.w.tensor("stft.isin").map_or(&[], |t| t.0),
            ),
        }
    }

    /// Select the coarse-delay worker at initialization, before any audio.
    /// Neural scratch allocation and high-band cost are independent RT gates.
    pub fn enable_async_delay(&mut self) -> std::io::Result<()> {
        self.daf.enable_async_delay()
    }

    pub fn enable_delay_tracking(&mut self) -> std::io::Result<()> {
        self.daf.enable_delay_tracking()
    }

    /// A fault is latched until the owner rebuilds the Engine off the data loop.
    pub fn is_faulted(&self) -> bool {
        self.faulted
    }

    /// Process one 256-sample hop of mic + far-end reference; returns 256 samples.
    pub fn process_hop(&mut self, m: &Model, mic: &[f32], reference: &[f32]) -> [f32; 256] {
        const H: usize = 256;
        if self.faulted {
            return [0.0; H];
        }
        if mic.len() != H
            || reference.len() != H
            || mic.iter().chain(reference).any(|v| !v.is_finite())
        {
            self.faulted = true;
            return [0.0; H];
        }
        let mut e = [0.0f32; H];
        let mut yh = [0.0f32; H];
        self.daf.process(mic, reference, H, &mut e, &mut yh);
        if e.iter().chain(&yh).any(|v| !v.is_finite()) {
            self.faulted = true;
            return [0.0; H];
        }
        // running RMS gain (EMA) toward the 0.05 training level
        let p = (e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / H as f64) as f32;
        self.pow = if self.pow_init {
            0.95 * self.pow + 0.05 * p
        } else {
            p
        };
        self.pow_init = true;
        if !self.pow.is_finite() {
            self.faulted = true;
            return [0.0; H];
        }
        let gain = (0.05 / (self.pow.sqrt() + 1e-6)).clamp(0.05, 50.0);
        // slide 512 windows, append raw hop
        self.buf_e.copy_within(H.., 0);
        self.buf_y.copy_within(H.., 0);
        self.buf_e[H..].copy_from_slice(&e);
        self.buf_y[H..].copy_from_slice(&yh);
        let mut win_e = [0.0f32; 512];
        let mut win_y = [0.0f32; 512];
        for n in 0..512 {
            win_e[n] = self.buf_e[n] * gain;
            win_y[n] = self.buf_y[n] * gain;
        }
        let (wcos, _) = m.w.tensor("stft.wcos").unwrap();
        let (wsin, _) = m.w.tensor("stft.wsin").unwrap();
        let (icos, _) = m.w.tensor("stft.icos").unwrap();
        let (isin, _) = m.w.tensor("stft.isin").unwrap();
        let (win2, _) = m.w.tensor("stft.win2").unwrap();
        let (mut re_e, mut im_e) = ([0.0f32; 257], [0.0f32; 257]);
        let (mut re_y, mut im_y) = ([0.0f32; 257], [0.0f32; 257]);
        if let Some(fft) = &mut self.fft {
            fft.analyze(&win_e, &mut re_e, &mut im_e);
            fft.analyze(&win_y, &mut re_y, &mut im_y);
        } else {
            stft::stft_frame(&win_e, wcos, wsin, &mut re_e, &mut im_e);
            stft::stft_frame(&win_y, wcos, wsin, &mut re_y, &mut im_y);
        }
        let mut spe = [0.0f32; 514];
        let mut spy = [0.0f32; 514];
        for f in 0..257 {
            spe[f * 2] = re_e[f];
            spe[f * 2 + 1] = im_e[f];
            spy[f * 2] = re_y[f];
            spy[f * 2 + 1] = im_y[f];
        }
        if spe.iter().chain(&spy).any(|v| !v.is_finite()) {
            self.faulted = true;
            return [0.0; H];
        }
        let osp = self.core.process_frame(&m.w, &spe, &spy);
        if osp.iter().any(|v| !v.is_finite()) {
            self.faulted = true;
            return [0.0; H];
        }
        let (mut ore, mut oim) = ([0.0f32; 257], [0.0f32; 257]);
        for f in 0..257 {
            ore[f] = osp[f * 2];
            oim[f] = osp[f * 2 + 1];
        }
        let mut ft = [0.0f32; 512];
        match &mut self.fft {
            Some(fft) => fft.synthesize(&ore, &oim, &mut ft),
            None => stft::istft_frame(&ore, &oim, icos, isin, &mut ft),
        }
        let inv = 1.0 / gain;
        for v in &mut ft {
            *v *= inv;
        }
        for n in 0..512 {
            self.acc[n] += ft[n];
            self.wenv[n] += win2[n];
        }
        let mut out = [0.0f32; H];
        for (i, o) in out.iter_mut().enumerate() {
            *o = if self.wenv[i] > 1e-11 {
                self.acc[i] / self.wenv[i]
            } else {
                0.0
            };
        }
        self.acc.copy_within(H.., 0);
        self.acc[H..].fill(0.0);
        self.wenv.copy_within(H.., 0);
        self.wenv[H..].fill(0.0);
        if out.iter().chain(&self.acc).any(|v| !v.is_finite()) {
            self.faulted = true;
            return [0.0; H];
        }
        out
    }
}

/// Whole-signal convenience over the streaming path (16 kHz), for offline use and
/// parity with `localvqe --stream`.
#[must_use]
pub fn run_aec_stream(m: &Model, mic: &[f32], reference: &[f32]) -> Vec<f32> {
    let n = (mic.len().min(reference.len()) / Streamer::HOP) * Streamer::HOP;
    let mut s = Streamer::new(m);
    let mut out = vec![0.0f32; mic.len()];
    let mut o = 0;
    while o < n {
        let h = s.process_hop(
            m,
            &mic[o..o + Streamer::HOP],
            &reference[o..o + Streamer::HOP],
        );
        out[o..o + Streamer::HOP].copy_from_slice(&h);
        o += Streamer::HOP;
    }
    out
}

/// STFT into the core's freq-major interleaved layout `[(f*T+t)*2 + {re,im}]`.
fn stft_freq_major(sig: &[f32], wcos: &[f32], wsin: &[f32], t_n: usize) -> Vec<f32> {
    let (re, im) = stft::stft(sig, wcos, wsin);
    let fb = stft::N_FREQ;
    let mut spec = vec![0.0f32; fb * t_n * 2];
    for t in 0..t_n {
        for f in 0..fb {
            spec[(f * t_n + t) * 2] = re[t * fb + f];
            spec[(f * t_n + t) * 2 + 1] = im[t * fb + f];
        }
    }
    spec
}

fn freq_major_to_time(spec: &[f32], t_n: usize) -> (Vec<f32>, Vec<f32>) {
    let fb = stft::N_FREQ;
    let mut re = vec![0.0f32; t_n * fb];
    let mut im = vec![0.0f32; t_n * fb];
    for t in 0..t_n {
        for f in 0..fb {
            re[t * fb + f] = spec[(f * t_n + t) * 2];
            im[t * fb + f] = spec[(f * t_n + t) * 2 + 1];
        }
    }
    (re, im)
}

/// Model geometry read from the GGUF metadata.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_fft: usize,
    pub hop: usize,
    pub n_freq: usize,
    pub is_aec: bool,
}

pub struct Model {
    pub cfg: Config,
    pub w: Gguf,
}

/// Expected tensor geometry (GGUF physical dims) for the shipped GTCRN-AEC
/// model, one `name d0,d1,...` per line. Regenerate if the model changes.
const SCHEMA: &str = include_str!("gtcrn_aec_schema.txt");

/// Validate every tensor the forward path needs against [`SCHEMA`], so an
/// incompatible model is rejected at load rather than panicking in the audio
/// callback. Generic over the lookup so it can be unit-tested without a Gguf.
fn validate_schema(dims_of: impl Fn(&str) -> Option<Vec<usize>>) -> Result<(), String> {
    for (n, line) in SCHEMA.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (name, dims) = line
            .split_once(' ')
            .ok_or_else(|| format!("malformed schema line {}", n + 1))?;
        let expected: Vec<usize> = dims
            .split(',')
            .map(|d| {
                d.parse::<usize>()
                    .map_err(|_| format!("bad schema dim on line {}", n + 1))
            })
            .collect::<Result<_, _>>()?;
        match dims_of(name) {
            None => return Err(format!("model missing tensor {name}")),
            Some(got) if got != expected => {
                return Err(format!(
                    "tensor {name}: dims {got:?} != expected {expected:?}"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_geometry(cfg: &Config) -> Result<(), String> {
    if (cfg.n_fft, cfg.hop, cfg.n_freq, cfg.is_aec) != (512, 256, 257, true) {
        return Err(
            "unsupported GTCRN-AEC geometry: requires FFT=512, hop=256, bins=257, is_aec=1".into(),
        );
    }
    Ok(())
}

impl Model {
    /// Load weights + geometry from a GTCRN-AEC `.gguf`.
    pub fn load(path: &str) -> Result<Self, String> {
        let w = Gguf::load(path)?;
        let size = |key: &str| -> Result<usize, String> {
            usize::try_from(w.meta_u64(key).ok_or_else(|| format!("missing {key}"))?)
                .map_err(|_| format!("{key} exceeds addressable size"))
        };
        let cfg = Config {
            n_fft: size("gtcrn.n_fft")?,
            hop: size("gtcrn.hop_length")?,
            n_freq: size("gtcrn.n_freq_bins")?,
            is_aec: w.meta_u64("gtcrn.is_aec") == Some(1),
        };
        validate_geometry(&cfg)?;
        // Bind every weight the forward path will read now, on the load thread,
        // instead of letting a missing/misshaped tensor panic mid-callback.
        validate_schema(|name| w.tensor(name).map(|(_, d)| d.to_vec()))?;
        Ok(Self { cfg, w })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Point at the model with AEC_GTCRN_GGUF; skip when unavailable so the gate
    // stays green on machines without the weights.
    fn model_path() -> Option<String> {
        std::env::var("AEC_GTCRN_GGUF")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
    }

    #[test]
    fn loads_config_and_known_tensors() {
        let Some(path) = model_path() else {
            eprintln!("skip: set AEC_GTCRN_GGUF to the 49K gguf");
            return;
        };
        let m = Model::load(&path).expect("load gguf");
        assert_eq!(m.cfg.n_fft, 512);
        assert_eq!(m.cfg.n_freq, 257);
        assert!(m.cfg.is_aec);

        // STFT analysis matrix and an encoder conv weight have the shapes the
        // forward port expects.
        let (wcos, d) = m.w.tensor("stft.wcos").expect("stft.wcos");
        assert_eq!(d, &[512, 257]);
        assert_eq!(wcos.len(), 512 * 257);
        assert!(wcos.iter().all(|v| v.is_finite()));

        let (_, d0) = m.w.tensor("encoder.en_convs.0.w").expect("en_convs.0.w");
        assert_eq!(d0, &[5, 1, 18, 16]);

        // DAF echo front-end tensors are present (this is the AEC build).
        assert!(m.w.tensor("daf.head.weight").is_some());
    }

    #[test]
    fn stft_istft_round_trips() {
        let Some(path) = model_path() else { return };
        let m = Model::load(&path).expect("load");
        let (wcos, _) = m.w.tensor("stft.wcos").unwrap();
        let (wsin, _) = m.w.tensor("stft.wsin").unwrap();
        let (icos, _) = m.w.tensor("stft.icos").unwrap();
        let (isin, _) = m.w.tensor("stft.isin").unwrap();
        let (win2, _) = m.w.tensor("stft.win2").unwrap();

        // Deterministic pseudo-random signal, length a multiple of HOP.
        let l = stft::HOP * 40;
        let mut s = 0x1234_5678u32;
        let sig: Vec<f32> = (0..l)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 9) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect();

        let (re, im) = stft::stft(&sig, wcos, wsin);
        let t_n = stft::n_frames(l);
        let y = stft::istft(&re, im.as_slice(), t_n, l, icos, isin, win2);

        // Interior reconstructs closely (edges carry padding artifacts).
        let (a, b) = (stft::N_FFT, l - stft::N_FFT);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in a..b {
            num += f64::from(y[i] - sig[i]).powi(2);
            den += f64::from(sig[i]).powi(2);
        }
        let rel = (num / den).sqrt();
        assert!(rel < 1e-3, "STFT round-trip rel error {rel:.2e}");
    }

    /// Minimal .npy reader for C-order little-endian <f4 fixtures.
    fn npy_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
        let b = std::fs::read(path).expect("read npy");
        assert_eq!(&b[..6], b"\x93NUMPY");
        let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
        let header = std::str::from_utf8(&b[10..10 + hlen]).unwrap();
        assert!(header.contains("'<f4'"), "npy not <f4: {header}");
        assert!(header.contains("False"), "npy must be C-order");
        let sh = &header[header.find("(").unwrap() + 1..header.find(")").unwrap()];
        let shape: Vec<usize> = sh
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let data_off = 10 + hlen;
        let raw = &b[data_off..];
        assert_eq!(raw.len() % 4, 0, "truncated f32 fixture");
        let f = raw
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        (shape, f)
    }

    // Parity oracle = the LocalVQE `gtcrn.cpp` scalar reference run on the SAME
    // GGUF (the committed upstream `.npy` fixtures are from a different model
    // version and match only `feat`). Regenerate the oracle dir with the
    // `dump_gtcrn` harness (see aec-eval), then point AEC_GTCRN_FIXTURES at it.
    // Every stage must be bit-exact: same scalar ops, same weights.
    #[test]
    fn core_forward_bit_exact_vs_reference() {
        let (Some(gg), Ok(fix)) = (model_path(), std::env::var("AEC_GTCRN_FIXTURES")) else {
            eprintln!("skip: set AEC_GTCRN_GGUF and AEC_GTCRN_FIXTURES (reference dump)");
            return;
        };
        if !std::path::Path::new(&format!("{fix}/enc0.npy")).exists() {
            return;
        }
        let m = Model::load(&gg).expect("load");
        let (she, e) = npy_f32(&format!("{fix}/in_spec_e.npy"));
        let (_, y) = npy_f32(&format!("{fix}/in_spec_y.npy"));
        let t_n = she[2];
        let mut cap = Vec::new();
        model::forward_capture(&m.w, &e, &y, t_n, &mut cap);
        for (name, val) in &cap {
            let (_, wref) = npy_f32(&format!("{fix}/{name}.npy"));
            assert_eq!(val.len(), wref.len(), "{name} length");
            let max = val
                .iter()
                .zip(&wref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max < 1e-4, "{name} max abs diff {max:.2e} vs reference");
        }
    }

    #[test]
    fn streaming_core_matches_batch() {
        // Feed the same spec frames one-by-one through the stateful streaming core;
        // because the core is causal over time, it must reproduce the batch forward.
        let (Some(gg), Ok(fix)) = (model_path(), std::env::var("AEC_GTCRN_FIXTURES")) else {
            return;
        };
        if !std::path::Path::new(&format!("{fix}/enc0.npy")).exists() {
            return;
        }
        let m = Model::load(&gg).expect("load");
        let (she, e) = npy_f32(&format!("{fix}/in_spec_e.npy")); // (1,257,T,2)
        let (_, y) = npy_f32(&format!("{fix}/in_spec_y.npy"));
        let t_n = she[2];
        let want = model::forward(&m.w, &e, &y, t_n);
        let mut sc = model::StreamCore::new();
        let mut got = vec![0.0f32; 257 * t_n * 2];
        let mut ef = vec![0.0f32; 257 * 2];
        let mut yf = vec![0.0f32; 257 * 2];
        for t in 0..t_n {
            for fb in 0..257 {
                ef[fb * 2] = e[(fb * t_n + t) * 2];
                ef[fb * 2 + 1] = e[(fb * t_n + t) * 2 + 1];
                yf[fb * 2] = y[(fb * t_n + t) * 2];
                yf[fb * 2 + 1] = y[(fb * t_n + t) * 2 + 1];
            }
            let o = sc.process_frame(&m.w, &ef, &yf);
            for fb in 0..257 {
                got[(fb * t_n + t) * 2] = o[fb * 2];
                got[(fb * t_n + t) * 2 + 1] = o[fb * 2 + 1];
            }
        }
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "streaming vs batch max abs diff {max:.2e}");
    }

    #[test]
    fn run_aec_cancels_echo() {
        let Some(path) = model_path() else { return };
        let m = Model::load(&path).expect("load");
        // ref = deterministic noise; mic = a delayed, attenuated copy (pure echo,
        // no near-end). A working AEC drives the residual well below the mic.
        let n = 16000 * 3;
        let mut s = 0xABCD_1234u32;
        let reference: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.3
            })
            .collect();
        let delay = 300;
        let mic: Vec<f32> = (0..n)
            .map(|i| {
                if i >= delay {
                    reference[i - delay] * 0.5
                } else {
                    0.0
                }
            })
            .collect();
        let out = run_aec(&m, &mic, &reference);
        // Compare residual to mic energy over the converged second half.
        let h = n / 2;
        let me: f64 = mic[h..].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let oe: f64 = out[h..].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let erle_db = 10.0 * (me / (oe + 1e-12)).log10();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(erle_db > 10.0, "echo not cancelled, ERLE {erle_db:.1} dB");
    }

    #[test]
    fn stft_frame_matches_batch() {
        // The streaming per-frame primitive must equal the (verified) batch STFT
        // on the same 512-sample analysis window, so the live path is consistent.
        let Some(path) = model_path() else { return };
        let m = Model::load(&path).expect("load");
        let (wcos, _) = m.w.tensor("stft.wcos").unwrap();
        let (wsin, _) = m.w.tensor("stft.wsin").unwrap();
        let l = stft::HOP * 20;
        let mut s = 0x2468u32;
        let sig: Vec<f32> = (0..l)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 9) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect();
        let (re, im) = stft::stft(&sig, wcos, wsin);
        // rebuild the reflect-padded buffer the batch STFT used
        let mut pad = vec![0.0f32; l + stft::N_FFT];
        pad[stft::HOP..stft::HOP + l].copy_from_slice(&sig);
        for j in 0..stft::HOP {
            pad[j] = sig[stft::HOP - j];
            pad[stft::HOP + l + j] = sig[l - 2 - j];
        }
        let t = 7; // an interior frame
        let mut fre = vec![0.0f32; stft::N_FREQ];
        let mut fim = vec![0.0f32; stft::N_FREQ];
        stft::stft_frame(
            &pad[t * stft::HOP..t * stft::HOP + stft::N_FFT],
            wcos,
            wsin,
            &mut fre,
            &mut fim,
        );
        for f in 0..stft::N_FREQ {
            assert!((fre[f] - re[t * stft::N_FREQ + f]).abs() < 1e-4);
            assert!((fim[f] - im[t * stft::N_FREQ + f]).abs() < 1e-4);
        }
    }

    #[test]
    fn run_aec_stream_cancels_echo() {
        let Some(path) = model_path() else { return };
        let m = Model::load(&path).expect("load");
        let n = 16000 * 3;
        let mut s = 0x5151u32;
        let reference: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 9) as f32 / (1u32 << 23) as f32 - 1.0) * 0.3
            })
            .collect();
        let delay = 300;
        let mic: Vec<f32> = (0..n)
            .map(|i| {
                if i >= delay {
                    reference[i - delay] * 0.5
                } else {
                    0.0
                }
            })
            .collect();
        let out = run_aec_stream(&m, &mic, &reference);
        assert!(out.iter().all(|v| v.is_finite()));
        let h = n / 2;
        let me: f64 = mic[h..].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let oe: f64 = out[h..].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let erle = 10.0 * (me / (oe + 1e-12)).log10();
        eprintln!("run_aec_stream_cancels_echo ERLE = {erle:.1} dB");
        // Quality gate for the DAF (docs/gtcrn-aec.md, DAF numerical note): the adaptive filter's
        // sample trajectory forks on a 1-ulp knife-edge (daf.rs Kalman clamp), so
        // sample-exact comparison is meaningless across FP variants. Echo-return-loss
        // is the trajectory-robust invariant; a working filter clears ~33 dB here,
        // a broken one sits below 10.
        assert!(
            erle > 25.0,
            "streaming echo not cancelled, ERLE {erle:.1} dB"
        );
    }

    #[test]
    fn erb_passes_low_bins() {
        let Some(path) = model_path() else { return };
        let m = Model::load(&path).expect("load");
        let (bmw, _) = m.w.tensor("erb.bm").unwrap();
        let frame: Vec<f32> = (0..erb::FULL).map(|i| i as f32 * 0.01).collect();
        let banded = erb::bm(&frame, bmw);
        assert_eq!(banded.len(), erb::BANDED);
        assert_eq!(&banded[..erb::LOW], &frame[..erb::LOW]);
        assert!(banded.iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    use std::collections::HashMap;

    fn full() -> HashMap<String, Vec<usize>> {
        SCHEMA
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let (name, dims) = l.split_once(' ').unwrap();
                let dims = dims.split(',').map(|d| d.parse().unwrap()).collect();
                (name.to_string(), dims)
            })
            .collect()
    }

    #[test]
    fn full_schema_matches_and_is_complete() {
        let map = full();
        assert_eq!(map.len(), 188, "schema tensor count");
        assert!(validate_schema(|n| map.get(n).cloned()).is_ok());
    }

    #[test]
    fn a_missing_tensor_is_rejected_at_load() {
        let mut map = full();
        map.remove("stft.wcos");
        let err = validate_schema(|n| map.get(n).cloned()).unwrap_err();
        assert!(err.contains("missing tensor stft.wcos"), "{err}");
    }

    #[test]
    fn a_misshaped_tensor_is_rejected_at_load() {
        let mut map = full();
        map.insert("stft.wcos".into(), vec![1, 2]);
        let err = validate_schema(|n| map.get(n).cloned()).unwrap_err();
        assert!(
            err.contains("stft.wcos") && err.contains("expected"),
            "{err}"
        );
    }
    #[test]
    fn reject_shape_compatible_but_semantically_wrong_model() {
        let mut cfg = Config {
            n_fft: 512,
            hop: 256,
            n_freq: 257,
            is_aec: true,
        };
        assert!(validate_geometry(&cfg).is_ok());
        cfg.hop = 128;
        assert!(validate_geometry(&cfg).is_err());
        cfg.hop = 256;
        cfg.is_aec = false;
        assert!(validate_geometry(&cfg).is_err());
    }
}
