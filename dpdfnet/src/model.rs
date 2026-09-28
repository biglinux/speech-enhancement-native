//! Fixed DPDFNet2/8-48k-HR topology, matched to Ceva commit 9bd9844a.
use crate::{layers::*, weights::*};
use std::sync::Arc;
pub const FFT: usize = 960;
pub const HOP: usize = 480;
pub const BINS: usize = 481;
pub const DF_BINS: usize = 96;
pub const MODEL_DELAY: usize = 4; // mask delay 2 + middle of DF history delay 2
const C: usize = 64;
const COEFS: usize = DF_BINS * 5 * 2;

pub struct DeepFilter {
    raw: History,
    masked: History,
    coefs: History,
    temp: Vec<f32>,
}
impl Default for DeepFilter {
    fn default() -> Self {
        Self::new()
    }
}
impl DeepFilter {
    pub fn new() -> Self {
        Self {
            raw: History::new(5, BINS * 2),
            masked: History::new(5, BINS * 2),
            coefs: History::new(3, COEFS),
            temp: vec![0.0; BINS * 2],
        }
    }
    pub fn apply(
        &mut self,
        spec: &[f32],
        mask: &[f32],
        coefs: &[f32],
        out: &mut [f32],
        dry_mix: f32,
    ) {
        self.raw.push(spec);
        // Upstream before_df: mask(t) multiplies spectrum(t-2), NOT current spectrum.
        for ((dst, src), &m) in self
            .temp
            .chunks_exact_mut(2)
            .zip(self.raw.frame(2).chunks_exact(2))
            .zip(mask)
        {
            dst[0] = src[0] * m;
            dst[1] = src[1] * m;
        }
        self.masked.push(&self.temp);
        self.coefs.push(coefs); // upstream coefs_buffer delay_frames=2
        out.copy_from_slice(self.masked.frame(2));
        let co = self.coefs.frame(0);
        for f in 0..DF_BINS {
            let (mut re, mut im) = (0.0, 0.0);
            for tap in 0..5 {
                let s = self.masked.frame(tap);
                let j = f * 10 + tap * 2;
                re += s[2 * f] * co[j] - s[2 * f + 1] * co[j + 1];
                im += s[2 * f] * co[j + 1] + s[2 * f + 1] * co[j];
            }
            out[2 * f] = re;
            out[2 * f + 1] = im;
        }
        let dry = self.raw.frame(0);
        for (y, &x) in out.iter_mut().zip(dry) {
            *y = (1.0 - dry_mix) * (*y) + dry_mix * x;
        }
    }
    pub fn reset(&mut self) {
        self.raw.reset();
        self.masked.reset();
        self.coefs.reset();
        self.temp.fill(0.0);
    }
}

pub struct Model {
    // Retain validated model metadata and the single shared backing allocation.
    pub bundle: Arc<Bundle>,
    pub window: F32s,
    pub wnorm: f32,
    alpha: f32,
    beta: f32,
    eps: f32,
    mu0: F32s,
    s0: F32s,
    mu: Vec<f32>,
    s: Vec<f32>,
    mag: Vec<f32>,
    complex: Vec<f32>,
    scaled: Vec<f32>,
    mag_hist: History,
    complex_hist: History,
    path_hist: History,
    erb0: Pipeline,
    erb1: Pipeline,
    erb2: Pipeline,
    erb3: Pipeline,
    df0: Pipeline,
    df1: Pipeline,
    erb_dual: Dprnn,
    df_dual: Dprnn,
    enc_erb_fc: Linear,
    enc_df_fc: Linear,
    enc_gru: Squeezed,
    joined: Vec<f32>,
    erb_gru: Squeezed,
    erb_fc: Linear,
    expanded: Vec<f32>,
    skip3: Pipeline,
    up3: Pipeline,
    skip2: Pipeline,
    up2: Pipeline,
    skip1: Pipeline,
    up1: Pipeline,
    skip0: Pipeline,
    mask_out: Pipeline,
    add: Vec<f32>,
    mask: Vec<f32>,
    df_gru: Squeezed,
    df_skip: Linear,
    df_out: Linear,
    df_path: Pipeline,
    df_embed: Vec<f32>,
    df_skip_buf: Vec<f32>,
    coefs: Vec<f32>,
    filter: DeepFilter,
    out: Vec<f32>,
}
impl Model {
    pub fn new(bundle: Arc<Bundle>) -> Result<Self> {
        let b = &*bundle;
        let parsed = b.manifest()?;
        let m = &parsed;
        require(
            num(m, "sample_rate")? == 48000
                && num(m, "fft")? == FFT
                && num(m, "hop")? == HOP
                && num(m, "df_bins")? == DF_BINS,
            "only 48k HR geometry is implemented",
        )?;
        let depth = num(m, "depth")?;
        require(
            depth == 2 || depth == 8,
            "only trained depths 2 and 8 supported",
        )?;
        require(
            string(m, "mask_method")? == "before_df" && num(m, "lookahead")? == 2,
            "mask/lookahead contract mismatch",
        )?;
        let mu0 = b.f32s(&m["norm"]["mu0"], BINS)?;
        let s0 = b.f32s(&m["norm"]["s0"], DF_BINS)?;
        require(
            s0.iter().all(|&x| x >= 0.0),
            "negative initial spectral normalization",
        )?;
        let alpha = float(&m["norm"], "alpha")?;
        let eps = float(&m["norm"], "eps")?;
        let beta = float(&m["norm"], "one_minus_alpha")?;
        require(
            alpha > 0.0
                && alpha < 1.0
                && beta > 0.0
                && (alpha + beta - 1.0).abs() < 1e-6
                && eps > 0.0,
            "normalization parameters out of bounds",
        )?;
        require(
            float(&m["norm"], "std")? == 40.0,
            "only fixed variance normalization supported",
        )?;
        let window = b.f32s(&m["window"], FFT)?;
        let wnorm = float(m, "wnorm")?;
        require(wnorm > 0.0, "invalid FFT normalization")?;
        // COLA/NOLA: this executor assumes 50% overlap with a power-complementary window.
        for i in 0..HOP {
            require(
                ((window[i] * window[i] + window[i + HOP] * window[i + HOP]) - 1.0).abs() < 2e-5,
                "window is not power-complementary at 50% overlap",
            )?;
        }
        let e = &m["encoder"];
        let d = &m["mask_decoder"];
        let df = &m["df_decoder"];
        fn pipe(
            b: &Bundle,
            v: &serde_json::Value,
            shape: (usize, usize, usize),
            out: (usize, usize),
        ) -> Result<Pipeline> {
            let p = Pipeline::load(b, v)?;
            require(
                (p.in_t, p.in_f, p.in_c) == shape && (p.out_f, p.out_c) == out,
                "network convolution shape mismatch",
            )?;
            Ok(p)
        }
        fn lin(b: &Bundle, v: &serde_json::Value, i: usize, o: usize) -> Result<Linear> {
            let l = Linear::load(b, v)?;
            require(l.input == i && l.output == o, "network projection mismatch")?;
            Ok(l)
        }
        fn sq(b: &Bundle, v: &serde_json::Value, i: usize, o: usize) -> Result<Squeezed> {
            let l = Squeezed::load(b, v)?;
            require(l.input == i && l.output == o, "network squeezed mismatch")?;
            Ok(l)
        }
        let mut model = Self {
            bundle: bundle.clone(),
            window,
            wnorm,
            alpha,
            beta,
            eps,
            mu: mu0.to_vec(),
            s: s0.to_vec(),
            mu0,
            s0,
            mag: vec![0.0; BINS],
            complex: vec![0.0; DF_BINS * 2],
            scaled: vec![0.0; BINS * 2],
            mag_hist: History::new(3, 480),
            complex_hist: History::new(3, DF_BINS * 2),
            path_hist: History::new(5, DF_BINS * C),
            erb0: pipe(b, &e["erb0"], (3, 480, 1), (480, C))?,
            erb1: pipe(b, &e["erb1"], (1, 480, C), (160, C))?,
            erb2: pipe(b, &e["erb2"], (1, 160, C), (80, C))?,
            erb3: pipe(b, &e["erb3"], (1, 80, C), (40, C))?,
            df0: pipe(b, &e["df0"], (3, DF_BINS, 2), (DF_BINS, C))?,
            df1: pipe(b, &e["df1"], (1, DF_BINS, C), (48, C))?,
            erb_dual: Dprnn::load(b, &e["erb_dual"], 40, depth)?,
            df_dual: Dprnn::load(b, &e["df_dual"], 48, depth)?,
            enc_erb_fc: lin(b, &e["erb_fc"], 40 * C, 512)?,
            enc_df_fc: lin(b, &e["df_fc"], 48 * C, 512)?,
            enc_gru: sq(b, &e["gru"], 1024, 512)?,
            joined: vec![0.0; 1024],
            erb_gru: sq(b, &d["gru"], 512, 512)?,
            erb_fc: lin(b, &d["fc"], 512, 40 * C)?,
            expanded: vec![0.0; 40 * C],
            skip3: pipe(b, &d["skip3"], (1, 40, C), (40, C))?,
            up3: pipe(b, &d["up3"], (1, 40, C), (80, C))?,
            skip2: pipe(b, &d["skip2"], (1, 80, C), (80, C))?,
            up2: pipe(b, &d["up2"], (1, 80, C), (160, C))?,
            skip1: pipe(b, &d["skip1"], (1, 160, C), (160, C))?,
            up1: pipe(b, &d["up1"], (1, 160, C), (480, C))?,
            skip0: pipe(b, &d["skip0"], (1, 480, C), (480, C))?,
            mask_out: pipe(b, &d["out"], (1, 480, C), (480, 1))?,
            add: vec![0.0; 480 * C],
            mask: vec![0.0; BINS],
            df_gru: sq(b, &df["gru"], 512, 256)?,
            df_skip: lin(b, &df["skip"], 512, 256)?,
            df_out: lin(b, &df["out"], 256, COEFS)?,
            df_path: pipe(b, &df["path"], (5, DF_BINS, C), (DF_BINS, 10))?,
            df_embed: vec![0.0; 256],
            df_skip_buf: vec![0.0; 256],
            coefs: vec![0.0; COEFS],
            filter: DeepFilter::new(),
            out: vec![0.0; BINS * 2],
        };
        // Resolve inherited SIMD dispatch outside run(). No CPUID initialization in callback.
        #[cfg(target_arch = "x86_64")]
        let _ = dfn_ops::simd_tier();
        // Touch every hot-path allocation and initialize math dispatch before exposing the instance.
        // Warm-up is not measured as throughput and never replaces a real input frame.
        let silence = [0.0f32; BINS * 2];
        require(
            model
                .process_spectrum(&silence, 0.0)
                .iter()
                .all(|v| v.is_finite()),
            "model produced non-finite output during initialization",
        )?;
        model.reset();
        Ok(model)
    }
    /// Input and output: unnormalized FFT, [F, real/imag]. Same convention as official ONNX wrapper.
    /// Every frame is processed, including silence and 0 dB attenuation settings.
    pub fn process_spectrum(&mut self, spec: &[f32], dry_mix: f32) -> &[f32] {
        assert_eq!(spec.len(), BINS * 2);
        let dry_mix = if dry_mix.is_finite() {
            dry_mix.clamp(0.0, 1.0)
        } else {
            0.0
        };
        for (y, &x) in self.scaled.iter_mut().zip(spec) {
            // Keep finite external values within a wide, non-audio-clipping numerical safety bound.
            *y = if x.is_finite() {
                x.clamp(-1e10, 1e10) * self.wnorm
            } else {
                0.0
            };
        }
        for f in 0..BINS {
            let re = self.scaled[2 * f];
            let im = self.scaled[2 * f + 1];
            let mag = (re * re + im * im).sqrt();
            // Upstream deliberately uses 10*log10(magnitude + 1e-10), not 20*log10.
            let db = 10.0 * (mag + 1e-10).log10();
            self.mu[f] = self.alpha * self.mu[f] + self.beta * db;
            self.mag[f] = (db - self.mu[f]) / (40.0 + self.eps);
            if f < DF_BINS {
                self.s[f] = self.alpha * self.s[f] + self.beta * mag;
                let denom = (self.s[f] + self.eps).sqrt();
                self.complex[2 * f] = re / denom;
                self.complex[2 * f + 1] = im / denom;
            }
        }
        self.mag_hist.push(&self.mag[..480]);
        self.complex_hist.push(&self.complex);
        self.erb0.forward(self.mag_hist.view(480, 1));
        self.erb1.forward(View::frame(self.erb0.output(), 480, C));
        self.erb2.forward(View::frame(self.erb1.output(), 160, C));
        self.erb3.forward(View::frame(self.erb2.output(), 80, C));
        self.df0.forward(self.complex_hist.view(DF_BINS, 2));
        self.df1.forward(View::frame(self.df0.output(), DF_BINS, C));
        self.enc_erb_fc.apply(
            self.erb_dual.forward(self.erb3.output()),
            &mut self.joined[..512],
        );
        self.enc_df_fc.apply(
            self.df_dual.forward(self.df1.output()),
            &mut self.joined[512..],
        );
        self.enc_gru.forward(&self.joined);
        self.erb_fc.apply(
            self.erb_gru.forward(self.enc_gru.output()),
            &mut self.expanded,
        );
        self.skip3.forward(View::frame(self.erb3.output(), 40, C));
        add(&mut self.add[..40 * C], self.skip3.output(), &self.expanded);
        self.up3.forward(View::frame(&self.add[..40 * C], 40, C));
        self.skip2.forward(View::frame(self.erb2.output(), 80, C));
        add(
            &mut self.add[..80 * C],
            self.skip2.output(),
            self.up3.output(),
        );
        self.up2.forward(View::frame(&self.add[..80 * C], 80, C));
        self.skip1.forward(View::frame(self.erb1.output(), 160, C));
        add(
            &mut self.add[..160 * C],
            self.skip1.output(),
            self.up2.output(),
        );
        self.up1.forward(View::frame(&self.add[..160 * C], 160, C));
        self.skip0.forward(View::frame(self.erb0.output(), 480, C));
        add(&mut self.add, self.skip0.output(), self.up1.output());
        self.mask[..480].copy_from_slice(self.mask_out.forward(View::frame(&self.add, 480, C)));
        self.mask[480] = self.mask[478]; // torch pad(..., (0,1), mode='reflect'), not replication
        self.df_embed
            .copy_from_slice(self.df_gru.forward(self.enc_gru.output()));
        self.df_skip
            .apply(self.enc_gru.output(), &mut self.df_skip_buf);
        for (y, &v) in self.df_embed.iter_mut().zip(&self.df_skip_buf) {
            *y += v;
        }
        self.df_out.apply(&self.df_embed, &mut self.coefs);
        self.path_hist.push(self.df0.output());
        let path = self.df_path.forward(self.path_hist.view(DF_BINS, C));
        // tanh belongs to df_out BEFORE the pathway sum. Never clamp the final coefficients.
        for (y, &v) in self.coefs.iter_mut().zip(path) {
            *y += v;
        }
        self.filter.apply(
            &self.scaled,
            &self.mask,
            &self.coefs,
            &mut self.out,
            dry_mix,
        );
        for v in &mut self.out {
            *v /= self.wnorm;
        }
        &self.out
    }
    pub fn reset(&mut self) {
        self.mu.copy_from_slice(&self.mu0);
        self.s.copy_from_slice(&self.s0);
        self.mag_hist.reset();
        self.complex_hist.reset();
        self.path_hist.reset();
        self.filter.reset();
        self.erb_dual.reset();
        self.df_dual.reset();
        self.enc_gru.reset();
        self.erb_gru.reset();
        self.df_gru.reset();
        self.out.fill(0.0);
    }
    /// Diagnostic snapshots, read after processing from the SAME thread; not a concurrent API.
    pub fn trace(&self, id: usize) -> Option<&[f32]> {
        match id {
            0 => Some(&self.mag),
            1 => Some(&self.complex),
            2 => Some(self.erb0.output()),
            3 => Some(self.erb1.output()),
            4 => Some(self.erb2.output()),
            5 => Some(self.erb3.output()),
            6 => Some(self.df0.output()),
            7 => Some(self.df1.output()),
            8 => Some(self.erb_dual.output()),
            9 => Some(self.df_dual.output()),
            10 => Some(self.enc_gru.output()),
            11 => Some(&self.mask),
            12 => Some(&self.coefs),
            13 => Some(&self.out),
            _ => None,
        }
    }
}
fn add(y: &mut [f32], a: &[f32], b: &[f32]) {
    for ((y, &a), &b) in y.iter_mut().zip(a).zip(b) {
        *y = a + b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn before_df_alignment_identity() {
        let mut d = DeepFilter::new();
        let m = vec![1.0; BINS];
        let mut c = vec![0.0; COEFS];
        for f in 0..DF_BINS {
            c[f * 10 + 4] = 1.0;
        } // center tap; coefficients start zero for 2 frames
        let mut out = vec![0.0; BINS * 2];
        for n in 1i32..20 {
            let x = vec![n as f32; BINS * 2];
            d.apply(&x, &m, &c, &mut out, 0.0);
            if n >= 5 {
                for y in &out {
                    assert_eq!(*y, (n - 4) as f32);
                }
            }
        }
    }
    #[test]
    fn dry_reference_delayed_four_hops() {
        let mut d = DeepFilter::new();
        let m = vec![0.0; BINS];
        let c = vec![0.0; COEFS];
        let mut y = vec![0.0; BINS * 2];
        for n in 1i32..20 {
            d.apply(&vec![n as f32; BINS * 2], &m, &c, &mut y, 1.0);
            assert_eq!(y[20], (n - 4).max(0) as f32);
        }
    }
}
