//! FireRedVAD (FireRedTeam, Apache-2.0) run natively: Kaldi fbank, CMVN and the DFSMN
//! network, in two models.
//!
//! - [`Aed`] tells speech, singing and music apart. Each of its eight memory layers
//!   looks 20 frames ahead, so its answer for a frame comes 1.6 s later: it is for
//!   whole recordings.
//! - [`StreamVad`] answers speech or not for each frame as soon as the frame is heard,
//!   for live audio.
//!
//! [`SpeechOnly`] uses the AED to keep a denoised recording only where a voice speaks or
//! sings, for finding the pauses between words.
//!
//! The network is feed-forward over a finite window of frames, so unlike a recurrent
//! detector it has no state that drifts over hours of music or noise.

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use ops::resample::{Down3, LPF_DELAY};
use ops::{matvec_t, relu_inplace, sigmoid};
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};

/// 16 kHz samples between frames (10 ms).
pub const FRAME_SHIFT: usize = 160;
/// 16 kHz samples in a frame (25 ms); frame `f` covers `f * FRAME_SHIFT ..` this many.
pub const FRAME_LENGTH: usize = 400;
const FFT: usize = 512;
const MELS: usize = 80;
const HIDDEN: usize = 256;
const PROJECTION: usize = 128;
const ORDER: usize = 20;
const LAYERS: usize = 8;

const AED_BLOB: &[u8] = include_bytes!("../model/firered_aed.bin");
const STREAM_BLOB: &[u8] = include_bytes!("../model/firered_stream_vad.bin");

/// f32 values in a blob with `outputs` outputs, lookahead filters or not.
const fn blob_floats(outputs: usize, lookahead: bool) -> usize {
    let filters = if lookahead { 2 } else { 1 } * ORDER * PROJECTION;
    2 * MELS
        + (MELS * HIDDEN + HIDDEN)
        + (HIDDEN * PROJECTION + PROJECTION)
        + filters
        + (LAYERS - 1) * (PROJECTION * HIDDEN + HIDDEN + HIDDEN * PROJECTION + filters)
        + (PROJECTION * HIDDEN + HIDDEN)
        + (HIDDEN * outputs + outputs)
}
const _: () = assert!(AED_BLOB.len() == blob_floats(3, true) * 4, "AED blob size");
const _: () = assert!(
    STREAM_BLOB.len() == blob_floats(1, false) * 4,
    "Stream-VAD blob size"
);

/// One memory layer: the affine transforms into it, its filters over time.
struct Layer {
    /// `[in][out]` weights and bias of the affine step with ReLU.
    hidden: (Vec<f32>, Vec<f32>),
    /// `[in][out]` projection, with a bias in the first layer only.
    projection: (Vec<f32>, Option<Vec<f32>>),
    /// `[tap][channel]`: tap `k` weighs frame `t - (ORDER - 1) + k`.
    lookback: Vec<f32>,
    /// `[tap][channel]`: tap `k` weighs frame `t + 1 + k`.
    lookahead: Option<Vec<f32>>,
}

/// The weights, in the order `tools/export_weights.py` writes them.
struct Weights {
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
    layers: Vec<Layer>,
    dnn: (Vec<f32>, Vec<f32>),
    out: (Vec<f32>, Vec<f32>),
    outputs: usize,
}

fn parse(blob: &[u8], outputs: usize, lookahead: bool) -> Weights {
    let mut all = blob
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b));
    let mut take = |n: usize| -> Vec<f32> { all.by_ref().take(n).collect() };
    let mean = take(MELS);
    let inverse_std = take(MELS);
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let input = if layer == 0 { MELS } else { PROJECTION };
        let hidden = (take(input * HIDDEN), take(HIDDEN));
        let weights = take(HIDDEN * PROJECTION);
        let bias = (layer == 0).then(|| take(PROJECTION));
        let lookback = take(ORDER * PROJECTION);
        let ahead = lookahead.then(|| take(ORDER * PROJECTION));
        layers.push(Layer {
            hidden,
            projection: (weights, bias),
            lookback,
            lookahead: ahead,
        });
    }
    let dnn = (take(PROJECTION * HIDDEN), take(HIDDEN));
    let out = (take(HIDDEN * outputs), take(outputs));
    Weights {
        mean,
        inverse_std,
        layers,
        dnn,
        out,
        outputs,
    }
}

fn aed_weights() -> &'static Weights {
    static SHARED: OnceLock<Weights> = OnceLock::new();
    SHARED.get_or_init(|| parse(AED_BLOB, 3, true))
}

fn stream_weights() -> &'static Weights {
    static SHARED: OnceLock<Weights> = OnceLock::new();
    SHARED.get_or_init(|| parse(STREAM_BLOB, 1, false))
}

fn mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi log-mel fbank at 16 kHz: 25 ms Povey window every 10 ms, DC removal,
/// pre-emphasis 0.97, 512-point power spectrum, 80 triangular mel bands from 20 Hz
/// to Nyquist, natural log floored at the f32 epsilon; frames start at the first
/// sample and end before a partial one (`snip_edges`).
struct Fbank {
    window: Vec<f32>,
    /// Per band: the first FFT bin and the weights from there.
    bands: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn RealToComplex<f32>>,
    pending: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    energies: [f32; MELS],
}

impl Fbank {
    fn new(max_block: usize) -> Self {
        let window = (0..FRAME_LENGTH)
            .map(|i| {
                let cos = (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LENGTH - 1) as f64).cos();
                (0.5 - 0.5 * cos).powf(0.85) as f32
            })
            .collect();
        let (low, high) = (mel(20.0), mel(8000.0));
        let delta = (high - low) / (MELS + 1) as f32;
        let bin_width = 16_000.0 / FFT as f32;
        let bands = (0..MELS)
            .map(|band| {
                let left = low + band as f32 * delta;
                let (center, right) = (left + delta, left + 2.0 * delta);
                let weights: Vec<(usize, f32)> = (0..FFT / 2)
                    .filter_map(|i| {
                        let m = mel(bin_width * i as f32);
                        (m > left && m < right).then(|| {
                            let w = if m <= center {
                                (m - left) / (center - left)
                            } else {
                                (right - m) / (right - center)
                            };
                            (i, w)
                        })
                    })
                    .collect();
                let first = weights.first().map_or(0, |&(i, _)| i);
                (first, weights.into_iter().map(|(_, w)| w).collect())
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT);
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            window,
            bands,
            fft,
            pending: Vec::with_capacity(max_block + FRAME_LENGTH),
            frame: vec![0.0; FFT],
            spectrum,
            scratch,
            energies: [0.0; MELS],
        }
    }

    /// Takes 16 kHz samples in [-1, 1] and gives every complete frame, normalized.
    fn push(&mut self, audio: &[f32], weights: &Weights, mut frame: impl FnMut(&[f32; MELS])) {
        // The model was trained on 16-bit sample values. A corrupt sample, finite
        // but huge, would overflow the power spectrum and silence seconds of speech.
        self.pending.extend(audio.iter().map(|&x| {
            let x = if x.is_finite() {
                x.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            x * 32768.0
        }));
        let mut start = 0;
        while self.pending.len() - start >= FRAME_LENGTH {
            self.compute(start, weights);
            frame(&self.energies);
            start += FRAME_SHIFT;
        }
        self.pending.drain(..start);
    }

    fn compute(&mut self, start: usize, weights: &Weights) {
        let x = &mut self.frame[..FRAME_LENGTH];
        x.copy_from_slice(&self.pending[start..start + FRAME_LENGTH]);
        let mean = x.iter().sum::<f32>() / FRAME_LENGTH as f32;
        x.iter_mut().for_each(|v| *v -= mean);
        for i in (1..FRAME_LENGTH).rev() {
            x[i] -= 0.97 * x[i - 1];
        }
        x[0] -= 0.97 * x[0];
        for (v, w) in x.iter_mut().zip(&self.window) {
            *v *= w;
        }
        self.frame[FRAME_LENGTH..].fill(0.0);
        // Both buffers have the planned lengths, so the transform cannot fail.
        let _ =
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch);
        for (band, (energy, (first, w))) in self.energies.iter_mut().zip(&self.bands).enumerate() {
            let power: f32 = w
                .iter()
                .zip(&self.spectrum[*first..])
                .map(|(w, c)| w * c.norm_sqr())
                .sum();
            *energy =
                (power.max(f32::EPSILON).ln() - weights.mean[band]) * weights.inverse_std[band];
        }
    }
}

/// The DFSMN over a stream of frames. Each layer keeps the projections it has seen,
/// enough for its filters, and the inputs it still owes a residual; with lookahead
/// a layer answers frame `t` once frame `t + ORDER` has arrived.
struct Network {
    weights: &'static Weights,
    ahead: usize,
    span: usize,
    /// Per layer: projections, `[span][PROJECTION]` ring by frame.
    projections: Vec<Vec<f32>>,
    /// Per layer after the first: inputs, `[ahead + 1][PROJECTION]` ring by frame.
    inputs: Vec<Vec<f32>>,
    received: [usize; LAYERS],
    hidden: Vec<f32>,
    out: [f32; 3],
}

impl Network {
    fn new(weights: &'static Weights) -> Self {
        let ahead = if weights.layers[0].lookahead.is_some() {
            ORDER
        } else {
            0
        };
        let span = ORDER + ahead;
        Self {
            weights,
            ahead,
            span,
            projections: vec![vec![0.0; span * PROJECTION]; LAYERS],
            inputs: vec![vec![0.0; (ahead + 1) * PROJECTION]; LAYERS],
            received: [0; LAYERS],
            hidden: vec![0.0; HIDDEN],
            out: [0.0; 3],
        }
    }

    fn reset(&mut self) {
        self.received = [0; LAYERS];
    }

    /// Feeds layer `l` its next input frame.
    fn enter(&mut self, l: usize, input: &[f32], answer: &mut impl FnMut(&[f32])) {
        let layer = &self.weights.layers[l];
        let n_in = if l == 0 { MELS } else { PROJECTION };
        let u = self.received[l];
        matvec_t(&mut self.hidden, &layer.hidden.0, input, n_in, HIDDEN);
        add_bias(&mut self.hidden, &layer.hidden.1);
        relu_inplace(&mut self.hidden);
        let slot = (u % self.span) * PROJECTION;
        let p = &mut self.projections[l][slot..slot + PROJECTION];
        matvec_t(p, &layer.projection.0, &self.hidden, HIDDEN, PROJECTION);
        if let Some(bias) = &layer.projection.1 {
            add_bias(p, bias);
            relu_inplace(p);
        }
        if l > 0 {
            let slot = (u % (self.ahead + 1)) * PROJECTION;
            self.inputs[l][slot..slot + PROJECTION].copy_from_slice(input);
        }
        self.received[l] = u + 1;
        if u >= self.ahead {
            self.leave(l, u - self.ahead, answer);
        }
    }

    /// Computes layer `l`'s output for frame `t` and passes it on.
    fn leave(&mut self, l: usize, t: usize, answer: &mut impl FnMut(&[f32])) {
        let layer = &self.weights.layers[l];
        let received = self.received[l];
        let ring = &self.projections[l];
        let frame = |j: usize| &ring[(j % self.span) * PROJECTION..][..PROJECTION];
        let mut memory = [0.0f32; PROJECTION];
        memory.copy_from_slice(frame(t));
        for (k, taps) in layer
            .lookback
            .as_chunks::<PROJECTION>()
            .0
            .iter()
            .enumerate()
        {
            if let Some(j) = (t + k).checked_sub(ORDER - 1) {
                accumulate(&mut memory, taps, frame(j));
            }
        }
        if let Some(ahead) = &layer.lookahead {
            for (k, taps) in ahead.as_chunks::<PROJECTION>().0.iter().enumerate() {
                let j = t + 1 + k;
                // Frames past the end of the recording are zero.
                if j < received {
                    accumulate(&mut memory, taps, frame(j));
                }
            }
        }
        if l > 0 {
            let slot = (t % (self.ahead + 1)) * PROJECTION;
            for (m, x) in memory
                .iter_mut()
                .zip(&self.inputs[l][slot..slot + PROJECTION])
            {
                *m += x;
            }
        }
        if l + 1 < LAYERS {
            self.enter(l + 1, &memory, answer);
        } else {
            self.head(&memory, answer);
        }
    }

    fn head(&mut self, memory: &[f32], answer: &mut impl FnMut(&[f32])) {
        let w = self.weights;
        matvec_t(&mut self.hidden, &w.dnn.0, memory, PROJECTION, HIDDEN);
        add_bias(&mut self.hidden, &w.dnn.1);
        relu_inplace(&mut self.hidden);
        let out = &mut self.out[..w.outputs];
        matvec_t(out, &w.out.0, &self.hidden, HIDDEN, w.outputs);
        for (o, b) in out.iter_mut().zip(&w.out.1) {
            *o = sigmoid(*o + b);
        }
        answer(out);
    }

    /// Answers the frames the lookahead still holds, as if silence of zero
    /// projections followed.
    fn finish(&mut self, answer: &mut impl FnMut(&[f32])) {
        for l in 0..LAYERS {
            let received = self.received[l];
            for t in received.saturating_sub(self.ahead)..received {
                self.leave(l, t, answer);
            }
        }
    }
}

fn add_bias(x: &mut [f32], bias: &[f32]) {
    for (v, b) in x.iter_mut().zip(bias) {
        *v += b;
    }
}

fn accumulate(memory: &mut [f32; PROJECTION], taps: &[f32; PROJECTION], frame: &[f32]) {
    for ((m, w), x) in memory.iter_mut().zip(taps).zip(frame) {
        *m += w * x;
    }
}

/// Speech, singing and music probabilities for each 10 ms frame of a recording,
/// answered `AED_DELAY` frames late.
pub struct Aed {
    fbank: Fbank,
    network: Network,
}

/// Frames between hearing a frame and the [`Aed`]'s answer for it.
pub const AED_DELAY: usize = LAYERS * ORDER;

impl Default for Aed {
    fn default() -> Self {
        Self::new()
    }
}

impl Aed {
    #[must_use]
    pub fn new() -> Self {
        Self {
            fbank: Fbank::new(8192),
            network: Network::new(aed_weights()),
        }
    }

    /// Takes 16 kHz audio in [-1, 1]; `heard` gets `[speech, singing, music]` for each
    /// frame answered, in order from frame 0.
    pub fn push(&mut self, audio: &[f32], mut heard: impl FnMut(&[f32; 3])) {
        let Self { fbank, network } = self;
        let mut answer = |p: &[f32]| heard(&[p[0], p[1], p[2]]);
        fbank.push(audio, network.weights, |frame| {
            network.enter(0, frame, &mut answer)
        });
    }

    /// Answers the frames still held back; the partial frame at the end is dropped.
    pub fn finish(&mut self, mut heard: impl FnMut(&[f32; 3])) {
        let mut answer = |p: &[f32]| heard(&[p[0], p[1], p[2]]);
        self.network.finish(&mut answer);
    }
}

/// Speech probability for each 10 ms frame of live audio, answered as soon as the
/// frame is complete.
pub struct StreamVad {
    fbank: Fbank,
    network: Network,
}

impl StreamVad {
    /// `max_block`: the most 16 kHz samples one `push` takes without allocating.
    #[must_use]
    pub fn new(max_block: usize) -> Self {
        Self {
            fbank: Fbank::new(max_block),
            network: Network::new(stream_weights()),
        }
    }

    /// Takes 16 kHz audio in [-1, 1]; `heard` gets the probability of each frame
    /// completed, in order.
    pub fn push(&mut self, audio: &[f32], mut heard: impl FnMut(f32)) {
        let Self { fbank, network } = self;
        let mut answer = |p: &[f32]| heard(p[0]);
        fbank.push(audio, network.weights, |frame| {
            network.enter(0, frame, &mut answer)
        });
    }

    /// Forgets the audio heard so far, as at the start of a stream.
    pub fn reset(&mut self) {
        self.fbank.pending.clear();
        self.network.reset();
    }
}

/// Per sensitivity level from 1, the most sensitive, to 5: the smoothed probability
/// from which speech opens a voice, and the one from which singing alone does.
/// Guitars and synths reach 0.7 as singing, sung phrases 0.94 and more. A voice stays
/// open until `HOLD_FRAMES` in a row stay below `VOICE_CLOSE_BELOW`.
const OPEN_AT: [(f32, f32); 5] = [
    (0.10, 0.75),
    (0.20, 0.80),
    (0.25, 0.85),
    (0.40, 0.90),
    (0.60, 0.95),
];
const VOICE_CLOSE_BELOW: f32 = 0.25;
const HOLD_FRAMES: u32 = 32;
/// The denoised audio's speech counts only while the raw input sounds less like music
/// than this: speech buried in noise is plain once denoised, but so are instruments.
const MUSIC_BELOW: f32 = 0.3;
/// Frames in the centered moving average of each probability.
const SMOOTHING: usize = 5;
/// What is kept before and after the frames that heard a voice, in 48 kHz samples:
/// a word's onset is heard late, and its fading tail not at all.
const KEEP_BEFORE: i64 = 48_000;
const KEEP_AFTER: i64 = 24_000;
/// Frames a late singing opening reaches back for its rise, and so the most output
/// held back while a rise has not opened yet.
const RISE_REACH: i64 = 200;
/// 48 kHz input samples per frame shift.
const SHIFT_48K: i64 = 3 * FRAME_SHIFT as i64;

/// 48 kHz audio into the AED.
struct Ear {
    down: Down3,
    down_out: Vec<f32>,
    aed: Aed,
}

impl Ear {
    fn new() -> Self {
        Self {
            down: Down3::new(8192),
            down_out: Vec::new(),
            aed: Aed::new(),
        }
    }

    fn hear(&mut self, audio: &[f32], heard: impl FnMut(&[f32; 3])) {
        self.down.process(audio, &mut self.down_out);
        self.aed.push(&self.down_out, heard);
    }

    /// Silence through the resampler's delay brings the last frames to the AED.
    fn finish(&mut self, mut heard: impl FnMut(&[f32; 3])) {
        self.hear(&[0.0; LPF_DELAY + 3 * FRAME_LENGTH], &mut heard);
        self.aed.finish(heard);
    }
}

/// Keeps a denoised recording only where a voice speaks or sings, and silences the
/// rest, so its level shows the pauses between words.
///
/// The denoiser lets melodic instruments and effects through as if they were a voice.
/// The AED hears the raw input, where it tells voices from instruments, and the
/// denoised audio, where it hears the speech that noise buries in the raw input. A
/// frame's probability is the larger of the raw input's speech or singing and, unless
/// the raw input sounds like music, the denoised audio's speech. The sensitivity level
/// sets the probability that opens a voice, higher for singing alone (`OPEN_AT`).
/// Output trails the input by the AED's delay, the smoothing and `KEEP_BEFORE`;
/// `finish` returns the rest, the same number of samples in all.
pub struct SpeechOnly {
    speech_open: f32,
    singing_open: f32,
    raw: Ear,
    clean: Ear,
    raw_heard: VecDeque<[f32; 3]>,
    clean_heard: VecDeque<f32>,
    /// Per frame from `first` on that a decision still needs: the raw input's speech,
    /// singing and music and the denoised audio's speech.
    frames: VecDeque<[f32; 4]>,
    first: i64,
    /// Frames decided; frame `f` starts at 48 kHz input `f * SHIFT_48K - LPF_DELAY`.
    decided: i64,
    is_open: bool,
    quiet: u32,
    /// The first frame of the current run at or over `speech_open`: singing that
    /// opens later keeps `KEEP_BEFORE` before this frame, as speech would.
    rising: Option<i64>,
    /// Merged, ordered sample spans to keep.
    kept: VecDeque<(i64, i64)>,
    held: VecDeque<f32>,
    emitted: i64,
}

impl Default for SpeechOnly {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechOnly {
    /// At sensitivity level 3, the normal one.
    #[must_use]
    pub fn new() -> Self {
        Self::with_sensitivity(3)
    }

    /// At sensitivity `level`, from 1, which opens on the faintest voice, to 5, which
    /// opens only on a clear one.
    ///
    /// # Panics
    ///
    /// If `level` is not 1 to 5.
    #[must_use]
    pub fn with_sensitivity(level: u8) -> Self {
        assert!((1..=5).contains(&level), "sensitivity must be 1 to 5");
        let (speech_open, singing_open) = OPEN_AT[usize::from(level - 1)];
        Self {
            speech_open,
            singing_open,
            raw: Ear::new(),
            clean: Ear::new(),
            raw_heard: VecDeque::new(),
            clean_heard: VecDeque::new(),
            frames: VecDeque::new(),
            first: 0,
            decided: 0,
            is_open: false,
            quiet: 0,
            rising: None,
            kept: VecDeque::new(),
            held: VecDeque::new(),
            emitted: 0,
        }
    }

    /// Takes the next raw 48 kHz samples and the denoised ones aligned with them, and
    /// appends to `out` the denoised samples whose fate is known, silenced or not.
    ///
    /// # Panics
    ///
    /// If `raw` and `clean` differ in length.
    pub fn push(&mut self, raw: &[f32], clean: &[f32], out: &mut Vec<f32>) {
        assert_eq!(
            raw.len(),
            clean.len(),
            "raw and denoised samples must align"
        );
        self.held.extend(clean);
        let (raw_heard, clean_heard) = (&mut self.raw_heard, &mut self.clean_heard);
        self.raw.hear(raw, |p| raw_heard.push_back(*p));
        self.clean.hear(clean, |p| clean_heard.push_back(p[0]));
        self.pair();
        self.decide(false);
        // An undecided frame, or a rise that may still open, cannot reach back past
        // this sample.
        let back = self
            .rising
            .map_or(self.decided, |r| r.max(self.decided - RISE_REACH));
        let known = back * SHIFT_48K - LPF_DELAY as i64 - KEEP_BEFORE;
        self.emit(known, out);
    }

    /// Appends everything still held to `out`.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        let (raw_heard, clean_heard) = (&mut self.raw_heard, &mut self.clean_heard);
        self.raw.finish(|p| raw_heard.push_back(*p));
        self.clean.finish(|p| clean_heard.push_back(p[0]));
        self.pair();
        self.decide(true);
        self.emit(i64::MAX, out);
    }

    /// Joins the frames both ears have answered.
    fn pair(&mut self) {
        while let (Some(raw), Some(&clean)) = (self.raw_heard.front(), self.clean_heard.front()) {
            self.frames.push_back([raw[0], raw[1], raw[2], clean]);
            self.raw_heard.pop_front();
            self.clean_heard.pop_front();
        }
    }

    /// Decides every frame whose averages are known: all of them when `ending`, since
    /// frames past the recording count as zero, as do frames before it.
    fn decide(&mut self, ending: bool) {
        let half = (SMOOTHING / 2) as i64;
        let answered = self.first + self.frames.len() as i64;
        while self.decided < answered && (ending || self.decided + half < answered) {
            let f = self.decided;
            let mut mean = [0.0f32; 4];
            for j in (f - half..=f + half).filter(|&j| j >= self.first && j < answered) {
                let frame = self.frames[(j - self.first) as usize];
                for (m, v) in mean.iter_mut().zip(frame) {
                    *m += v / SMOOTHING as f32;
                }
            }
            let [speech, singing, music, clean_speech] = mean;
            let clean_speech = if music < MUSIC_BELOW {
                clean_speech
            } else {
                0.0
            };
            let p = speech.max(singing).max(clean_speech);
            if p >= self.speech_open {
                self.rising.get_or_insert(f);
            } else {
                self.rising = None;
            }
            if speech.max(clean_speech) >= self.speech_open || singing >= self.singing_open {
                self.is_open = true;
                self.quiet = 0;
            } else if p >= VOICE_CLOSE_BELOW {
                self.quiet = 0;
            } else if self.is_open {
                self.quiet += 1;
                if self.quiet >= HOLD_FRAMES {
                    self.is_open = false;
                }
            }
            if self.is_open {
                let from = self.rising.map_or(f, |r| r.max(f - RISE_REACH));
                let start = from * SHIFT_48K - LPF_DELAY as i64 - KEEP_BEFORE;
                let end = f * SHIFT_48K - LPF_DELAY as i64 + SHIFT_48K + KEEP_AFTER;
                match self.kept.back_mut() {
                    Some(last) if start <= last.1 => last.1 = last.1.max(end),
                    _ => self.kept.push_back((start, end)),
                }
            }
            self.decided += 1;
            while self.first < self.decided - half {
                self.frames.pop_front();
                self.first += 1;
            }
        }
    }

    fn emit(&mut self, until: i64, out: &mut Vec<f32>) {
        let until = until.min(self.emitted + self.held.len() as i64);
        while self.emitted < until {
            while self
                .kept
                .front()
                .is_some_and(|&(_, end)| end <= self.emitted)
            {
                self.kept.pop_front();
            }
            let (keep, next) = match self.kept.front() {
                Some(&(start, end)) if start <= self.emitted => (true, end.min(until)),
                Some(&(start, _)) => (false, start.min(until)),
                None => (false, until),
            };
            let n = usize::try_from(next - self.emitted).unwrap_or_default();
            out.extend(self.held.drain(..n).map(|x| if keep { x } else { 0.0 }));
            self.emitted = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spans `SpeechOnly` at sensitivity `level` keeps for raw AED answers
    /// `[speech, singing, music]` per frame, the denoised audio hearing nothing.
    fn kept(level: u8, frames: &[[f32; 3]]) -> Vec<(i64, i64)> {
        let mut filter = SpeechOnly::with_sensitivity(level);
        filter.raw_heard.extend(frames);
        filter.clean_heard.extend(frames.iter().map(|_| 0.0));
        filter.pair();
        filter.decide(true);
        filter.kept.into_iter().collect()
    }

    #[test]
    fn instrument_heard_as_singing_stays_silent() {
        assert!(kept(3, &[[0.0, 0.7, 1.0]; 300]).is_empty());
    }

    #[test]
    fn sung_phrase_keeps_its_rise() {
        let mut frames = vec![[0.0, 0.0, 1.0]; 300];
        frames[150..200].fill([0.0, 0.6, 1.0]);
        frames[200..260].fill([0.0, 0.95, 1.0]);
        let spans = kept(3, &frames);
        // The five-frame average reaches 0.25 on the rise's first frame.
        let rise = 150 * SHIFT_48K - LPF_DELAY as i64 - KEEP_BEFORE;
        assert_eq!(spans.first().map(|span| span.0), Some(rise));
    }

    #[test]
    fn sensitivity_sets_speech_opening() {
        let faint = [[0.3, 0.0, 0.0]; 100];
        assert!(!kept(2, &faint).is_empty());
        assert!(kept(4, &faint).is_empty());
    }
}
