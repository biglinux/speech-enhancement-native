//! The Silero VAD 16 kHz model (snakers4/silero-vad 6.2.3, MIT) run natively, and the
//! voice gate built on it.
//!
//! DeepFilterNet3 lets short bursts of noise through while nobody speaks. The gate mutes
//! the denoiser's output whenever the detector hears no speech in the *raw* input: deciding
//! on the denoised signal mistakes those same bursts for speech.
//!
//! The network is small (~0.68 M multiply-adds per 32 ms), so it runs in f32: a recurrent
//! detector is the wrong place to take quantization risk. `vdot_f32` differs across SIMD
//! tiers in the last bits, which can only move a decision that sits exactly on a threshold.

use std::sync::{Arc, OnceLock};

use dfn_ops::resample::{Down3, LPF_DELAY};
use dfn_ops::{relu_inplace, sigmoid, vdot_f32};

/// 16 kHz samples per decision (32 ms).
pub const CHUNK: usize = 512;
/// 48 kHz input samples per decision.
pub const CHUNK_48K: usize = CHUNK * 3;
const CONTEXT: usize = 64;
const REFLECT: usize = 64;
const PADDED: usize = CONTEXT + CHUNK + REFLECT;
const WINDOW: usize = 256;
const STRIDE: usize = 128;
const FRAMES: usize = (PADDED - WINDOW) / STRIDE + 1; // 4
const BINS: usize = 129;
const HIDDEN: usize = 128;

static BLOB: &[u8] = include_bytes!("../model/silero_vad_16k.bin");

/// The weights, in the order `tools/export_weights.py` writes them.
struct Weights {
    stft: Vec<f32>, // [258][256]
    conv: [(Vec<f32>, Vec<f32>); 4],
    w_ih: Vec<f32>, // [512][128]
    w_hh: Vec<f32>, // [512][128]
    bias: Vec<f32>, // b_ih + b_hh
    head: Vec<f32>, // [128]
    head_bias: f32,
}

fn weights() -> Arc<Weights> {
    static SHARED: OnceLock<Arc<Weights>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            let mut all = BLOB
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b));
            let mut take = |n: usize| -> Vec<f32> { all.by_ref().take(n).collect() };
            let stft = take(258 * WINDOW);
            let conv = [
                (take(128 * BINS * 3), take(128)),
                (take(64 * 128 * 3), take(64)),
                (take(64 * 64 * 3), take(64)),
                (take(128 * 64 * 3), take(128)),
            ];
            let w_ih = take(4 * HIDDEN * HIDDEN);
            let w_hh = take(4 * HIDDEN * HIDDEN);
            let b_ih = take(4 * HIDDEN);
            let bias = take(4 * HIDDEN)
                .iter()
                .zip(&b_ih)
                .map(|(a, b)| a + b)
                .collect();
            let head = take(HIDDEN);
            let head_bias = take(1)[0];
            assert!(
                all.next().is_none(),
                "Silero weight blob has the wrong size"
            );
            Arc::new(Weights {
                stft,
                conv,
                w_ih,
                w_hh,
                bias,
                head,
                head_bias,
            })
        })
        .clone()
}

/// The network over 16 kHz audio: one speech probability per `CHUNK` samples.
pub struct Silero {
    w: Arc<Weights>,
    x: [f32; PADDED],
    spec: [f32; BINS * FRAMES], // [bin][frame]
    a: [f32; 128 * FRAMES],
    b: [f32; 128 * FRAMES],
    col: [f32; BINS * 3],
    gates: [f32; 4 * HIDDEN],
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
}

impl Default for Silero {
    fn default() -> Self {
        Self::new()
    }
}

impl Silero {
    #[must_use]
    pub fn new() -> Self {
        Self {
            w: weights(),
            x: [0.0; PADDED],
            spec: [0.0; BINS * FRAMES],
            a: [0.0; 128 * FRAMES],
            b: [0.0; 128 * FRAMES],
            col: [0.0; BINS * 3],
            gates: [0.0; 4 * HIDDEN],
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
        }
    }

    /// Forget the audio heard so far, as at the start of a stream.
    pub fn reset(&mut self) {
        self.x[..CONTEXT].fill(0.0);
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    /// Speech probability of the next `CHUNK` samples.
    pub fn process(&mut self, chunk: &[f32; CHUNK]) -> f32 {
        // [context | the chunk | the chunk's end reflected].
        self.x[CONTEXT..CONTEXT + CHUNK].copy_from_slice(chunk);
        let end = CONTEXT + CHUNK;
        for j in 0..REFLECT {
            self.x[end + j] = self.x[end - 2 - j];
        }

        // The STFT as a convolution: rows 0..129 real, 129..258 imaginary.
        for f in 0..FRAMES {
            let frame = &self.x[f * STRIDE..f * STRIDE + WINDOW];
            for bin in 0..BINS {
                let re = vdot_f32(&self.w.stft[bin * WINDOW..(bin + 1) * WINDOW], frame);
                let im = vdot_f32(
                    &self.w.stft[(bin + BINS) * WINDOW..(bin + BINS + 1) * WINDOW],
                    frame,
                );
                self.spec[bin * FRAMES + f] = (re * re + im * im).sqrt();
            }
        }

        // conv1..conv4 (kernel 3, padding 1, strides 1, 2, 2, 1), each with ReLU.
        conv(
            &self.w.conv[0],
            &self.spec,
            BINS,
            4,
            1,
            &mut self.a,
            &mut self.col,
        );
        conv(
            &self.w.conv[1],
            &self.a,
            128,
            4,
            2,
            &mut self.b,
            &mut self.col,
        );
        conv(
            &self.w.conv[2],
            &self.b,
            64,
            2,
            2,
            &mut self.a,
            &mut self.col,
        );
        conv(
            &self.w.conv[3],
            &self.a,
            64,
            1,
            1,
            &mut self.b,
            &mut self.col,
        );

        // LSTMCell(128, 128), gates in PyTorch's order i, f, g, o.
        for (r, gate) in self.gates.iter_mut().enumerate() {
            let row = r * HIDDEN..(r + 1) * HIDDEN;
            *gate = vdot_f32(&self.w.w_ih[row.clone()], &self.b[..HIDDEN])
                + vdot_f32(&self.w.w_hh[row], &self.h)
                + self.w.bias[r];
        }
        for k in 0..HIDDEN {
            let i = sigmoid(self.gates[k]);
            let f = sigmoid(self.gates[HIDDEN + k]);
            let g = self.gates[2 * HIDDEN + k].tanh();
            let o = sigmoid(self.gates[3 * HIDDEN + k]);
            self.c[k] = f * self.c[k] + i * g;
            self.h[k] = o * self.c[k].tanh();
        }

        // The next chunk's context is this one's last 64 samples.
        self.x.copy_within(CHUNK..CHUNK + CONTEXT, 0);

        let mut out = self.h;
        relu_inplace(&mut out);
        sigmoid(vdot_f32(&self.w.head, &out) + self.w.head_bias)
    }
}

/// Conv1d with kernel 3 and padding 1 over `[channel][time]`, then ReLU, as one dot
/// product per output: `col` gathers the input window in the weights' [in][tap] order.
fn conv(
    (w, bias): &(Vec<f32>, Vec<f32>),
    input: &[f32],
    c_in: usize,
    t_in: usize,
    stride: usize,
    out: &mut [f32],
    col: &mut [f32],
) {
    let c_out = bias.len();
    let t_out = (t_in - 1) / stride + 1;
    let col = &mut col[..c_in * 3];
    for t in 0..t_out {
        for ci in 0..c_in {
            for k in 0..3 {
                let at = (t * stride + k).checked_sub(1).filter(|&i| i < t_in);
                col[ci * 3 + k] = at.map_or(0.0, |i| input[ci * t_in + i]);
            }
        }
        for co in 0..c_out {
            let v = vdot_f32(&w[co * c_in * 3..(co + 1) * c_in * 3], col) + bias[co];
            out[co * t_out + t] = v.max(0.0);
        }
    }
}

/// Probabilities at which the gate opens and, after `HOLD_CHUNKS`, closes.
const OPEN_AT: f32 = 0.2;
const CLOSE_BELOW: f32 = 0.1;
/// 320 ms in a row below `CLOSE_BELOW` closes the gate, bridging the pauses
/// between words.
const HOLD_CHUNKS: u32 = 10;
/// Decisions remembered for outputs that lag the input (256 ms at 32 ms each).
const HISTORY: usize = 8;
/// A chunk this quiet is digital silence: no inference, and after ~1 s a fresh state.
const SILENT_PEAK: f32 = 1e-6;
const SILENT_RESET_CHUNKS: u32 = 32;

/// Mutes a denoiser's output while the raw 48 kHz input holds no speech.
///
/// `feed` takes the raw input; `gain` is asked, sample by sample, for the output that
/// stands for input sample `t`. Decision `k` covers 48 kHz input
/// `k * CHUNK_48K .. (k + 1) * CHUNK_48K`; the output for `t` opens if the chunk holding
/// `t`, or any up to the first that ends 32 ms past it, heard speech: a word starting
/// late in a chunk leaves that chunk below `OPEN_AT`, and hearing ahead keeps its
/// onset. An output that lags the input by less than `LAG` falls back to the newest
/// decision, and so does one whose decisions have left the last `HISTORY`: feed the
/// input at most a few chunks ahead of the output.
pub struct VoiceGate {
    vad: Silero,
    down: Down3,
    down_out: Vec<f32>,
    chunk: [f32; CHUNK],
    filled: usize,
    decided: u64,
    open: [bool; HISTORY],
    is_open: bool,
    quiet: u32,
    silent: u32,
    gain: f32,
    floor: f32,
    attack: f32,
    release: f32,
}

impl VoiceGate {
    /// `depth_db` of 60 or more mutes completely.
    #[must_use]
    pub fn new(depth_db: f32, max_block: usize) -> Self {
        let mut gate = Self {
            vad: Silero::new(),
            down: Down3::new(max_block),
            down_out: Vec::with_capacity(max_block / 3 + 2),
            chunk: [0.0; CHUNK],
            filled: 0,
            decided: 0,
            open: [false; HISTORY],
            is_open: false,
            quiet: 0,
            silent: 0,
            gain: 0.0,
            floor: 0.0,
            attack: 1.0 - (-1.0f32 / (0.010 * 48_000.0)).exp(),
            release: 1.0 - (-1.0f32 / (0.150 * 48_000.0)).exp(),
        };
        gate.set_depth(depth_db);
        gate
    }

    pub fn set_depth(&mut self, depth_db: f32) {
        self.floor = if depth_db >= 60.0 {
            0.0
        } else {
            10f32.powf(-depth_db.max(0.0) / 20.0)
        };
    }

    /// How far input sample `t` must lie behind the input fed so far for `gain(t)` to
    /// have every decision it asks for: the last of them ends less than two chunks
    /// past `t`, plus the resampler's group delay.
    pub const LAG: usize = 2 * CHUNK_48K + LPF_DELAY;

    /// Feed raw 48 kHz input.
    pub fn feed(&mut self, raw: &[f32]) {
        self.down.process(raw, &mut self.down_out);
        let mut taken = 0;
        while taken < self.down_out.len() {
            let n = (CHUNK - self.filled).min(self.down_out.len() - taken);
            self.chunk[self.filled..self.filled + n]
                .copy_from_slice(&self.down_out[taken..taken + n]);
            self.filled += n;
            taken += n;
            if self.filled == CHUNK {
                self.filled = 0;
                self.decide();
            }
        }
    }

    fn decide(&mut self) {
        let peak = self.chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let probability = if peak < SILENT_PEAK {
            self.silent += 1;
            if self.silent == SILENT_RESET_CHUNKS {
                self.vad.reset();
            }
            0.0
        } else {
            self.silent = 0;
            self.vad.process(&self.chunk)
        };
        let open = self.step(probability);
        self.open[(self.decided % HISTORY as u64) as usize] = open;
        self.decided += 1;
    }

    /// Hysteresis: open at `OPEN_AT`, close after `HOLD_CHUNKS` consecutive chunks
    /// below `CLOSE_BELOW`.
    fn step(&mut self, probability: f32) -> bool {
        if probability >= OPEN_AT {
            self.is_open = true;
            self.quiet = 0;
        } else if probability >= CLOSE_BELOW {
            self.quiet = 0;
        } else if self.is_open {
            self.quiet += 1;
            if self.quiet >= HOLD_CHUNKS {
                self.is_open = false;
            }
        }
        self.is_open
    }

    /// The decisions `gain(t)` combines: the chunk holding `t` through the first one
    /// that ends 32 ms past it.
    fn chunks(t: i64) -> (i64, i64) {
        let u = t + LPF_DELAY as i64;
        let len = CHUNK_48K as i64;
        (u.div_euclid(len).max(0), (u + len - 1).div_euclid(len))
    }

    /// Gain for the output standing for input sample `t` (negative before the stream).
    pub fn gain(&mut self, t: i64) -> f32 {
        let (first, last) = Self::chunks(t);
        let open = if last < first {
            false
        } else if (last as u64) < self.decided && self.decided - (first as u64) <= HISTORY as u64 {
            (first..=last).any(|k| self.open[(k as u64 % HISTORY as u64) as usize])
        } else {
            self.is_open
        };
        let target = if open { 1.0 } else { self.floor };
        let rate = if target > self.gain {
            self.attack
        } else {
            self.release
        };
        self.gain += (target - self.gain) * rate;
        self.gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_keeps_the_gate_closed_without_running_the_network() {
        let mut gate = VoiceGate::new(40.0, 4096);
        gate.feed(&vec![0.0; CHUNK_48K * 40]);
        assert_eq!(gate.decided, 40);
        assert!(!gate.is_open);
        assert!(gate.gain(CHUNK_48K as i64 * 39) <= 0.011);
    }

    #[test]
    fn an_output_lag_of_lag_samples_always_finds_its_own_decisions() {
        let mut gate = VoiceGate::new(40.0, 1);
        for fed in 1..=CHUNK_48K * 6 {
            gate.feed(&[0.01 * (fed as f32 * 0.3).sin()]);
            let Some(t) = fed.checked_sub(VoiceGate::LAG) else {
                continue;
            };
            let (first, last) = VoiceGate::chunks(t as i64);
            assert!(
                (last as u64) < gate.decided,
                "input {t} undecided after {fed} fed"
            );
            assert!(gate.decided - first as u64 <= HISTORY as u64);
        }
    }

    #[test]
    fn the_chunks_begin_at_the_sample_and_reach_32_ms_past_it() {
        let len = CHUNK_48K as i64;
        for t in 0..len * 4 {
            let (first, last) = VoiceGate::chunks(t);
            let u = t + LPF_DELAY as i64;
            assert!(first * len <= u && u < (first + 1) * len, "t {t}");
            // The last chunk ends at least one chunk past `t`, by less than one chunk
            // more, which is what LAG pays for.
            let end = (last + 1) * len;
            assert!(end >= u + len && end < u + 2 * len, "t {t}");
        }
    }

    #[test]
    fn a_gate_made_before_its_depth_is_known_starts_closed() {
        // The plugins build the gate at instantiation and set the depth on first run.
        let mut gate = VoiceGate::new(0.0, 0);
        gate.set_depth(40.0);
        assert!(gate.gain(-1) <= 0.011);
    }

    #[test]
    fn depth_sets_the_floor_and_sixty_mutes() {
        let mut gate = VoiceGate::new(40.0, 0);
        assert!((gate.floor - 0.01).abs() < 1e-6);
        gate.set_depth(60.0);
        assert_eq!(gate.floor, 0.0);
        gate.set_depth(0.0);
        assert_eq!(gate.floor, 1.0);
    }

    #[test]
    fn the_gate_closes_after_the_hold_of_consecutive_quiet_chunks() {
        let mut gate = VoiceGate::new(40.0, 0);
        assert!(gate.step(0.9));
        assert!((0..HOLD_CHUNKS - 1).all(|_| gate.step(0.0)));
        // A chunk between the thresholds keeps the state and restarts the hold.
        assert!(gate.step(0.15));
        assert!((0..HOLD_CHUNKS - 1).all(|_| gate.step(0.0)));
        assert!(!gate.step(0.0));
        // Closed, it stays closed between the thresholds and opens at OPEN_AT.
        assert!(!gate.step(0.15));
        assert!(gate.step(0.2));
    }
}
