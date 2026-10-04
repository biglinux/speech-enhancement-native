//! Mono 48 kHz streaming adapter. Arbitrary host block sizes, including in-place.
//! Two-hop framing delay + four-hop model delay = 2880 samples (60 ms).
use crate::{Bundle, Model, Result, BINS, FFT, HOP, MODEL_DELAY};
use realfft::{num_complex::Complex32, ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
pub const LATENCY: usize = (2 + MODEL_DELAY) * HOP;

/// Flush-to-zero + denormals-are-zero on the calling (real-time) thread. The
/// recurrent FP32 state decays toward denormal magnitudes when the input goes
/// quiet; without FTZ/DAZ, denormal arithmetic on x86 is 10-100x slower and
/// produces multi-millisecond hop spikes (xruns) during silence. MXCSR is
/// per-thread, so every real-time entry point (LADSPA `run`, the C `process`)
/// sets it. Idempotent and cheap; left set for the thread by design.
#[inline]
pub(crate) fn flush_denormals() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::asm;
        // FTZ = bit 15 (0x8000), DAZ = bit 6 (0x0040). stmxcsr/ldmxcsr rather than
        // the deprecated _mm_{get,set}csr intrinsics.
        let mut csr: u32 = 0;
        unsafe {
            asm!("stmxcsr [{p}]", p = in(reg) &mut csr, options(nostack, preserves_flags));
            csr |= 0x8040;
            asm!("ldmxcsr [{p}]", p = in(reg) &csr, options(nostack, readonly, preserves_flags));
        }
    }
}
/// Windowed FFT of the newest hop over the one before it.
pub(crate) struct Analysis {
    fft: Arc<dyn RealToComplex<f32>>,
    scratch: Vec<Complex32>,
    window: Vec<f32>,
    history: Vec<f32>,
    real: Vec<f32>,
    complex: Vec<Complex32>,
    spectrum: Vec<f32>,
}
impl Analysis {
    fn new(window: &[f32]) -> Result<Self> {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT);
        let mut s = Self {
            scratch: fft.make_scratch_vec(),
            fft,
            window: window.to_vec(),
            history: vec![0.0; FFT],
            real: vec![0.0; FFT],
            complex: vec![Complex32::new(0.0, 0.0); BINS],
            spectrum: vec![0.0; BINS * 2],
        };
        // Prefault scratch + exercise FFT dispatch during initialization, outside the callback.
        s.fft
            .process_with_scratch(&mut s.real, &mut s.complex, &mut s.scratch)
            .map_err(|e| e.to_string())?;
        s.real.fill(0.0);
        s.complex.fill(Complex32::new(0.0, 0.0));
        Ok(s)
    }
    /// The interleaved spectrum, or `None` if the transform failed.
    pub(crate) fn run(&mut self, hop: &[f32]) -> Option<&[f32]> {
        self.history.copy_within(HOP..FFT, 0);
        self.history[HOP..].copy_from_slice(hop);
        for ((r, &x), &w) in self.real.iter_mut().zip(&self.history).zip(&self.window) {
            *r = x * w;
        }
        self.fft
            .process_with_scratch(&mut self.real, &mut self.complex, &mut self.scratch)
            .ok()?;
        for (o, c) in self
            .spectrum
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(&self.complex)
        {
            o[0] = c.re;
            o[1] = c.im;
        }
        Some(&self.spectrum)
    }
}

/// Inverse FFT and overlap-add of one enhanced spectrum.
pub(crate) struct Synthesis {
    ifft: Arc<dyn ComplexToReal<f32>>,
    scratch: Vec<Complex32>,
    window: Vec<f32>,
    real: Vec<f32>,
    complex: Vec<Complex32>,
    ola: Vec<f32>,
}
impl Synthesis {
    fn new(window: &[f32]) -> Result<Self> {
        let ifft = RealFftPlanner::<f32>::new().plan_fft_inverse(FFT);
        let mut s = Self {
            scratch: ifft.make_scratch_vec(),
            ifft,
            window: window.to_vec(),
            real: vec![0.0; FFT],
            complex: vec![Complex32::new(0.0, 0.0); BINS],
            ola: vec![0.0; FFT],
        };
        s.ifft
            .process_with_scratch(&mut s.complex, &mut s.real, &mut s.scratch)
            .map_err(|e| e.to_string())?;
        s.real.fill(0.0);
        s.complex.fill(Complex32::new(0.0, 0.0));
        Ok(s)
    }
    /// Writes the next output hop; false if the transform failed or the result is not finite.
    pub(crate) fn run(&mut self, spec: &[f32], hop: &mut [f32]) -> bool {
        for (o, x) in self.complex.iter_mut().zip(spec.as_chunks::<2>().0) {
            o.re = x[0];
            o.im = x[1];
        }
        // Real inverse transform requires real-valued DC and Nyquist bins.
        self.complex[0].im = 0.0;
        self.complex[BINS - 1].im = 0.0;
        if self
            .ifft
            .process_with_scratch(&mut self.complex, &mut self.real, &mut self.scratch)
            .is_err()
        {
            return false;
        }
        for ((o, &r), &w) in self.ola.iter_mut().zip(&self.real).zip(&self.window) {
            *o += r * w / FFT as f32;
            if !o.is_finite() {
                return false;
            }
        }
        hop.copy_from_slice(&self.ola[..HOP]);
        self.ola.copy_within(HOP..FFT, 0);
        self.ola[HOP..].fill(0.0);
        true
    }
}

/// Input as the plugin takes it: non-finite samples become 0, huge ones are bounded.
pub(crate) fn bounded_input(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

pub struct AudioProcessor {
    pub model: Model,
    analysis: Analysis,
    synthesis: Synthesis,
    pending_in: Vec<f32>,
    pending_out: Vec<f32>,
    position: usize,
    dry_mix: f32,
    target_mix: f32,
    last_control: f32,
    mix_step: f32,
    fault: bool,
    pub hops: u64,
    pub sanitized_samples: u64,
}
impl AudioProcessor {
    pub fn new(bundle: Arc<Bundle>) -> Result<Self> {
        let model = Model::new(bundle)?;
        let analysis = Analysis::new(&model.window)?;
        let synthesis = Synthesis::new(&model.window)?;
        // Resolve the dB conversion's libm path outside a future control-change callback.
        let _ = std::hint::black_box(dfn_ops::atten_lim_from_db(std::hint::black_box(12.0)));
        Ok(Self {
            model,
            analysis,
            synthesis,
            pending_in: vec![0.0; HOP],
            pending_out: vec![0.0; HOP],
            position: 0,
            dry_mix: 0.0,
            target_mix: 0.0,
            last_control: 100.0,
            mix_step: 1.0 / 5.0,
            fault: false,
            hops: 0,
            sanitized_samples: 0,
        })
    }
    /// The parts of a processor fresh from `reset`, for the offline pipeline.
    /// The dry mix stays constant while the control does not change.
    pub(crate) fn into_parts(self) -> (Analysis, Model, Synthesis, f32) {
        (self.analysis, self.model, self.synthesis, self.dry_mix)
    }
    /// dB range 0..100, non-finite => full enhancement. Ramped over at most five hops.
    pub fn set_attenuation_db(&mut self, db: f32) {
        let db = if db.is_finite() {
            db.clamp(0.0, 100.0)
        } else {
            100.0
        };
        if db != self.last_control {
            self.target_mix = dfn_ops::atten_lim_from_db(db);
            self.last_control = db;
        }
    }
    pub fn reset(&mut self) {
        self.model.reset();
        self.analysis.history.fill(0.0);
        self.synthesis.ola.fill(0.0);
        self.pending_in.fill(0.0);
        self.pending_out.fill(0.0);
        self.position = 0;
        self.dry_mix = self.target_mix;
        self.fault = false;
        self.hops = 0;
        self.sanitized_samples = 0;
    }
    pub fn faulted(&self) -> bool {
        self.fault
    }
    /// One sample in/out is intentional: block partition invariance without queues or growth paths.
    #[inline]
    pub fn sample(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.sanitized_samples = self.sanitized_samples.saturating_add(1);
        }
        let y = self.pending_out[self.position];
        self.pending_in[self.position] = bounded_input(x);
        self.position += 1;
        if self.position == HOP {
            self.position = 0;
            self.process_hop();
        }
        if self.fault {
            0.0
        } else if y.is_finite() {
            y
        } else {
            self.fault = true;
            0.0
        }
    }
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (y, &x) in output.iter_mut().zip(input) {
            *y = self.sample(x);
        }
    }
    fn process_hop(&mut self) {
        if !self.fault {
            self.fault = !self.enhance_hop();
        }
        if self.fault {
            self.pending_out.fill(0.0);
        } else {
            self.hops = self.hops.saturating_add(1);
        }
    }
    fn enhance_hop(&mut self) -> bool {
        let Some(spectrum) = self.analysis.run(&self.pending_in) else {
            return false;
        };
        self.dry_mix += (self.target_mix - self.dry_mix).clamp(-self.mix_step, self.mix_step);
        let spec = self.model.process_spectrum(spectrum, self.dry_mix);
        !spec.iter().any(|v| !v.is_finite()) && self.synthesis.run(spec, &mut self.pending_out)
    }
}
