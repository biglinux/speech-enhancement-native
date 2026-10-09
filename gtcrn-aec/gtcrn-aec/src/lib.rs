//! Native Rust port of GTCRN-AEC, the 49K-parameter echo canceller from
//! LocalVQE (Apache-2.0): the network from `ggml/gtcrn.cpp` and the DAF front
//! end from `ggml/daf_frontend.cpp`, at 16 kHz in 256-sample hops. `run_aec`
//! mirrors LocalVQE's whole-file mode; `Streamer` is the per-hop path the
//! PipeWire plugin drives. Accuracy against the reference: `docs/gtcrn-aec.md`.

pub mod daf;
mod erb;
mod fft;
pub mod gguf;
pub mod hbaec;
pub mod model;
pub mod stft;

use gguf::Gguf;

/// Full offline AEC: mic + far-end reference -> echo-cancelled mic.
/// Mirrors LocalVQE `localvqe_process_f32` file mode: prime the bulk delay,
/// run the DAF adaptive front-end, AGC-normalize, GTCRN mask, iSTFT, un-gain.
pub fn run_aec(m: &Model, mic: &[f32], reference: &[f32]) -> Result<Vec<f32>, String> {
    const MBLK: usize = 128; // DAF block
    let n = (mic.len().min(reference.len()) / MBLK) * MBLK;
    let mut out = vec![0.0f32; mic.len()];
    if n == 0 {
        return Ok(out);
    }
    let mut daf = daf::Daf::new(&m.w)?;
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

    let [wcos, wsin, icos, isin, win2] = m.stft_tensors();
    let t_n = stft::n_frames(n);
    let spec_e = stft_freq_major(&e, wcos, wsin, t_n);
    let spec_y = stft_freq_major(&yh, wcos, wsin, t_n);
    let enh = model::forward(&m.net, &spec_e, &spec_y, t_n);
    let (re, im) = freq_major_to_time(&enh, t_n);
    let y = stft::istft(&re, &im, t_n, n, icos, isin, win2);
    let inv = 1.0 / gain;
    for i in 0..n {
        out[i] = y[i] * inv;
    }
    Ok(out)
}

/// Live per-hop AEC (16 kHz, 256-sample hop), mirroring LocalVQE
/// `process_gtcrn_frame`: DAF hop, running-RMS AGC, 512-sample window, STFT
/// frame, streaming GTCRN, iSTFT frame, un-gain, 50% overlap-add. Recurrent
/// state carries across `process_hop` calls, which never allocate.
pub struct Streamer {
    daf: daf::Daf,
    core: model::StreamCore,
    buf_e: [f32; 512],
    buf_y: [f32; 512],
    acc: [f32; 512],
    wenv: [f32; 512],
    pow: f32,
    pow_init: bool,
    fft: stft::RealFft,
    win2: [f32; 512],
}

impl Streamer {
    pub const HOP: usize = 256;

    pub fn new(m: &Model) -> Result<Self, String> {
        let _ = ops::simd_tier(); // resolve ISA dispatch off the callback
        let [.., win2] = m.stft_tensors();
        Ok(Self {
            daf: daf::Daf::new(&m.w)?,
            core: model::StreamCore::prepared(&m.net),
            buf_e: [0.0; 512],
            buf_y: [0.0; 512],
            acc: [0.0; 512],
            wenv: [0.0; 512],
            pow: 0.0,
            pow_init: false,
            fft: stft::RealFft::new(&m.window),
            win2: win2
                .try_into()
                .map_err(|_| "stft.win2 is not 512 samples")?,
        })
    }

    /// Return to the freshly built state without allocating.
    pub fn reset(&mut self) {
        self.daf.reset();
        self.core.reset();
        self.buf_e.fill(0.0);
        self.buf_y.fill(0.0);
        self.acc.fill(0.0);
        self.wenv.fill(0.0);
        self.pow = 0.0;
        self.pow_init = false;
    }

    /// Process one hop of mic + far-end reference. A non-finite intermediate
    /// (an overflow inside the adaptive filter, for instance) resets the
    /// streamer and yields one hop of silence, so a fault never outlives a hop.
    pub fn process_hop(
        &mut self,
        m: &Model,
        mic: &[f32; Self::HOP],
        reference: &[f32; Self::HOP],
    ) -> [f32; Self::HOP] {
        const H: usize = Streamer::HOP;
        let mut e = [0.0f32; H];
        let mut yh = [0.0f32; H];
        self.daf.process(mic, reference, H, &mut e, &mut yh);
        // running RMS gain (EMA) toward the 0.05 training level
        let p = (e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / H as f64) as f32;
        self.pow = if self.pow_init {
            0.95 * self.pow + 0.05 * p
        } else {
            p
        };
        self.pow_init = true;
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
        let (mut re_e, mut im_e) = ([0.0f32; 257], [0.0f32; 257]);
        let (mut re_y, mut im_y) = ([0.0f32; 257], [0.0f32; 257]);
        self.fft.analyze(&win_e, &mut re_e, &mut im_e);
        self.fft.analyze(&win_y, &mut re_y, &mut im_y);
        let mut spe = [0.0f32; 514];
        let mut spy = [0.0f32; 514];
        for f in 0..257 {
            spe[f * 2] = re_e[f];
            spe[f * 2 + 1] = im_e[f];
            spy[f * 2] = re_y[f];
            spy[f * 2 + 1] = im_y[f];
        }
        let osp = self.core.process_frame(&m.net, &spe, &spy);
        let (mut ore, mut oim) = ([0.0f32; 257], [0.0f32; 257]);
        for f in 0..257 {
            ore[f] = osp[f * 2];
            oim[f] = osp[f * 2 + 1];
        }
        let mut ft = [0.0f32; 512];
        self.fft.synthesize(&ore, &oim, &mut ft);
        let inv = 1.0 / gain;
        for v in &mut ft {
            *v *= inv;
        }
        for ((acc, wenv), (&v, &w2)) in self
            .acc
            .iter_mut()
            .zip(&mut self.wenv)
            .zip(ft.iter().zip(&self.win2))
        {
            *acc += v;
            *wenv += w2;
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
        // Every stage feeds the overlap-add, so checking its output and
        // carried tail catches a non-finite value anywhere in this hop.
        if out.iter().chain(&self.acc).any(|v| !v.is_finite()) {
            self.reset();
            return [0.0; H];
        }
        out
    }
}

/// Whole-signal convenience over the streaming path (16 kHz), for offline use and
/// parity with `localvqe --stream`.
pub fn run_aec_stream(m: &Model, mic: &[f32], reference: &[f32]) -> Result<Vec<f32>, String> {
    let n = mic.len().min(reference.len());
    let mut s = Streamer::new(m)?;
    let mut out = vec![0.0f32; mic.len()];
    let hops = mic[..n]
        .as_chunks()
        .0
        .iter()
        .zip(reference[..n].as_chunks().0)
        .zip(out.as_chunks_mut().0);
    for ((mic, reference), out) in hops {
        *out = s.process_hop(m, mic, reference);
    }
    Ok(out)
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
#[derive(Debug, PartialEq)]
struct Geometry {
    n_fft: usize,
    hop: usize,
    n_freq: usize,
    is_aec: bool,
}

pub struct Model {
    /// The DAF and STFT tensors; the network's moved into `net`.
    pub w: Gguf,
    net: model::Net,
    /// Analysis window of the STFT matrices.
    window: Vec<f32>,
}

/// Tensor names and GGUF dims of the shipped model, one `name d0,d1,...` per
/// line. The test `schema_lists_the_shipped_model` regenerates it from the GGUF
/// and prints the new text when the file is out of date.
const SCHEMA: &str = include_str!("gtcrn_aec_schema.txt");

/// Check a model's tensors against [`SCHEMA`]. A model that passes has every
/// name and shape of the shipped one, so the forward path's weight lookups and
/// kernels behave as tested and cannot fail later, in the audio callback.
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

fn validate_geometry(geometry: &Geometry) -> Result<(), String> {
    let supported = Geometry {
        n_fft: 512,
        hop: 256,
        n_freq: 257,
        is_aec: true,
    };
    if *geometry != supported {
        return Err(
            "unsupported GTCRN-AEC geometry: requires FFT=512, hop=256, bins=257, is_aec=1".into(),
        );
    }
    Ok(())
}

impl Model {
    /// Load and check the weights of a GTCRN-AEC `.gguf`.
    pub fn load(path: &str) -> Result<Self, String> {
        let mut w = Gguf::load(path)?;
        let size = |key: &str| -> Result<usize, String> {
            usize::try_from(w.meta_u64(key).ok_or_else(|| format!("missing {key}"))?)
                .map_err(|_| format!("{key} exceeds addressable size"))
        };
        validate_geometry(&Geometry {
            n_fft: size("gtcrn.n_fft")?,
            hop: size("gtcrn.hop_length")?,
            n_freq: size("gtcrn.n_freq_bins")?,
            is_aec: w.meta_u64("gtcrn.is_aec") == Some(1),
        })?;
        validate_schema(|name| w.tensor(name).map(|(_, d)| d.to_vec()))?;
        let net = model::Net::take(&mut w)?;
        let mut m = Self {
            w,
            net,
            window: Vec::new(),
        };
        let [wcos, wsin, icos, isin, _] = m.stft_tensors();
        m.window = stft::dft_window(wcos, wsin, icos, isin)
            .ok_or("STFT matrices are not a sine-windowed real DFT")?;
        Ok(m)
    }

    /// `stft.wcos`, `stft.wsin`, `stft.icos`, `stft.isin` and `stft.win2`,
    /// present since `load` checked the schema.
    fn stft_tensors(&self) -> [&[f32]; 5] {
        [
            "stft.wcos",
            "stft.wsin",
            "stft.icos",
            "stft.isin",
            "stft.win2",
        ]
        .map(|name| self.w.tensor(name).map_or(&[][..], |(data, _)| data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED_MODEL: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../model/localvqe-pi-aec-v1-49k-f32.gguf"
    );

    fn model() -> Model {
        let path = std::env::var("AEC_GTCRN_GGUF").unwrap_or_else(|_| SHIPPED_MODEL.into());
        Model::load(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    fn noise(n: usize, mut seed: u32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Pure echo: the microphone hears a delayed, halved copy of the reference.
    fn echo_only(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let reference: Vec<f32> = noise(n, seed).iter().map(|v| 0.3 * v).collect();
        let delay = 300;
        let mic = (0..n)
            .map(|i| {
                if i >= delay {
                    0.5 * reference[i - delay]
                } else {
                    0.0
                }
            })
            .collect();
        (mic, reference)
    }

    /// Echo reduction over the converged second half, in dB.
    fn erle_db(mic: &[f32], out: &[f32]) -> f64 {
        let h = mic.len() / 2;
        let me: f64 = mic[h..].iter().map(|&v| f64::from(v).powi(2)).sum();
        let oe: f64 = out[h..].iter().map(|&v| f64::from(v).powi(2)).sum();
        10.0 * (me / (oe + 1e-12)).log10()
    }

    #[test]
    fn non_dft_stft_matrices_are_rejected() {
        let m = model();
        let [wcos, wsin, icos, isin, _] = m.stft_tensors();
        assert!(stft::dft_window(wcos, wsin, icos, isin).is_some());
        let mut bent = wcos.to_vec();
        bent[N_BENT] += 1e-3;
        assert!(stft::dft_window(&bent, wsin, icos, isin).is_none());
    }
    const N_BENT: usize = 3 * stft::N_FFT + 7;

    #[test]
    fn stft_istft_round_trips() {
        let m = model();
        let [wcos, wsin, icos, isin, win2] = m.stft_tensors();
        let l = stft::HOP * 40;
        let sig = noise(l, 0x1234_5678);
        let (re, im) = stft::stft(&sig, wcos, wsin);
        let t_n = stft::n_frames(l);
        let y = stft::istft(&re, im.as_slice(), t_n, l, icos, isin, win2);
        // Interior only: the edges carry padding artifacts.
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
        let sh = &header[header.find('(').unwrap() + 1..header.find(')').unwrap()];
        let shape: Vec<usize> = sh
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let raw = &b[10 + hlen..];
        assert_eq!(raw.len() % 4, 0, "truncated f32 fixture");
        let f = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        (shape, f)
    }

    // The oracle is the LocalVQE `gtcrn.cpp` reference run on the same GGUF by
    // eval/dump_gtcrn.cpp: AEC_GTCRN_FIXTURES holds its input spectra
    // (in_spec_e.npy, in_spec_y.npy) and the stage dumps it wrote for them.
    #[test]
    #[ignore = "needs AEC_GTCRN_FIXTURES, a reference dump from eval/dump_gtcrn.cpp"]
    fn core_stages_match_reference_dump() {
        let fix = std::env::var("AEC_GTCRN_FIXTURES").expect("AEC_GTCRN_FIXTURES");
        let m = model();
        let (she, e) = npy_f32(&format!("{fix}/in_spec_e.npy"));
        let (_, y) = npy_f32(&format!("{fix}/in_spec_y.npy"));
        let t_n = she[2];
        for (name, val) in model::forward_capture(&m.net, &e, &y, t_n) {
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
        // The core is causal over time, so feeding frames one by one through
        // the stateful core must reproduce the batch forward.
        let m = model();
        let [wcos, wsin, ..] = m.stft_tensors();
        let l = stft::HOP * 60;
        let t_n = stft::n_frames(l);
        let e = stft_freq_major(&noise(l, 11), wcos, wsin, t_n);
        let y = stft_freq_major(&noise(l, 12), wcos, wsin, t_n);
        let want = model::forward(&m.net, &e, &y, t_n);
        let mut sc = model::StreamCore::prepared(&m.net);
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
            let o = sc.process_frame(&m.net, &ef, &yf);
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
        // The batch and streaming kernels round differently, so the
        // bound is relative to the output peak; measured ~8e-7.
        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max <= 1e-5 * peak,
            "streaming vs batch max abs diff {max:.2e}, output peak {peak:.2e}"
        );
    }

    #[test]
    fn run_aec_cancels_echo() {
        let m = model();
        let (mic, reference) = echo_only(16000 * 3, 0xABCD_1234);
        let out = run_aec(&m, &mic, &reference).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let erle = erle_db(&mic, &out);
        eprintln!("run_aec_cancels_echo ERLE = {erle:.1} dB");
        assert!(erle > 10.0, "echo not cancelled, ERLE {erle:.1} dB");
    }

    #[test]
    fn stft_frame_matches_batch() {
        // The per-frame primitive must equal the batch STFT on the same
        // 512-sample window.
        let m = model();
        let [wcos, wsin, ..] = m.stft_tensors();
        let l = stft::HOP * 20;
        let sig = noise(l, 0x2468);
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
        let m = model();
        let (mic, reference) = echo_only(16000 * 3, 0x5151);
        let out = run_aec_stream(&m, &mic, &reference).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let erle = erle_db(&mic, &out);
        eprintln!("run_aec_stream_cancels_echo ERLE = {erle:.1} dB");
        // The DAF's sample stream depends on rounding (docs/gtcrn-aec.md, DAF
        // numerical note), so the gate is echo reduction, not sample equality.
        assert!(
            erle > 25.0,
            "streaming echo not cancelled, ERLE {erle:.1} dB"
        );
    }

    #[test]
    fn fault_resets_to_the_fresh_state() {
        let m = model();
        let (mic, reference) = echo_only(Streamer::HOP * 40, 0x77);
        let hops = |s: &mut Streamer| -> Vec<[f32; Streamer::HOP]> {
            mic.as_chunks()
                .0
                .iter()
                .zip(reference.as_chunks().0)
                .map(|(m_, r)| s.process_hop(&m, m_, r))
                .collect()
        };
        let mut faulted = Streamer::new(&m).unwrap();
        hops(&mut faulted);
        let poison = [f32::NAN; Streamer::HOP];
        assert_eq!(
            faulted.process_hop(&m, &poison, &poison),
            [0.0; Streamer::HOP]
        );
        let after = hops(&mut faulted);
        assert_eq!(after, hops(&mut Streamer::new(&m).unwrap()));
    }

    fn schema_of(w: &Gguf) -> String {
        let mut names = w.tensor_names();
        names.sort_unstable();
        names
            .into_iter()
            .map(|name| {
                let dims: Vec<String> = w
                    .tensor(name)
                    .unwrap()
                    .1
                    .iter()
                    .map(usize::to_string)
                    .collect();
                format!("{name} {}\n", dims.join(","))
            })
            .collect()
    }

    #[test]
    fn schema_lists_the_shipped_model() {
        let generated = schema_of(&Gguf::load(SHIPPED_MODEL).unwrap());
        assert!(
            generated == SCHEMA,
            "src/gtcrn_aec_schema.txt is out of date; replace it with:\n{generated}"
        );
    }

    fn schema_map() -> std::collections::HashMap<String, Vec<usize>> {
        SCHEMA
            .lines()
            .map(|l| {
                let (name, dims) = l.split_once(' ').unwrap();
                (
                    name.to_string(),
                    dims.split(',').map(|d| d.parse().unwrap()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn a_missing_tensor_is_rejected_at_load() {
        let mut map = schema_map();
        map.remove("stft.wcos");
        let err = validate_schema(|n| map.get(n).cloned()).unwrap_err();
        assert!(err.contains("missing tensor stft.wcos"), "{err}");
    }

    #[test]
    fn a_misshaped_tensor_is_rejected_at_load() {
        let mut map = schema_map();
        map.insert("stft.wcos".into(), vec![1, 2]);
        let err = validate_schema(|n| map.get(n).cloned()).unwrap_err();
        assert!(
            err.contains("stft.wcos") && err.contains("expected"),
            "{err}"
        );
    }

    #[test]
    fn reject_shape_compatible_but_semantically_wrong_model() {
        let mut geometry = Geometry {
            n_fft: 512,
            hop: 256,
            n_freq: 257,
            is_aec: true,
        };
        assert!(validate_geometry(&geometry).is_ok());
        geometry.hop = 128;
        assert!(validate_geometry(&geometry).is_err());
        geometry.hop = 256;
        geometry.is_aec = false;
        assert!(validate_geometry(&geometry).is_err());
    }
}
