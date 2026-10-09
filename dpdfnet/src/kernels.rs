//! Matrix, GRU and LayerNorm weights. Batches of four vectors share every int8
//! weight load.
use crate::weights::{Bundle, F32s, I8s, I16s, Result, float, num, require, string};
use serde_json::Value;

#[derive(Clone)]
enum Storage {
    Float(F32s),
    Packed(I8s, F32s),
}
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    storage: Storage,
    plan: crate::packed::Plan,
    widened: Option<I16s>,
}
impl Matrix {
    pub fn load(b: &Bundle, v: &Value) -> Result<Self> {
        let rows = num(v, "rows")?;
        let cols = num(v, "cols")?;
        require(
            rows > 0 && rows <= 3072 && cols > 0 && cols <= 1024,
            "matrix dimensions out of bounds",
        )?;
        let storage = match string(v, "kind")? {
            "f32" => Storage::Float(b.f32s(&v["weight"], rows * cols)?),
            "w8a16" => {
                require(
                    string(v, "layout")? == "pair-output8-v1",
                    "unknown matrix layout",
                )?;
                require(rows % 8 == 0 && cols % 2 == 0, "invalid packed dimensions")?;
                let scales = b.f32s(&v["scales"], rows)?;
                require(
                    scales.iter().all(|&s| s > 0.0),
                    "invalid quantization scale",
                )?;
                Storage::Packed(b.i8s(&v["weight"], rows * cols)?, scales)
            }
            _ => return Err("unknown matrix format".into()),
        };
        Ok(Self {
            rows,
            cols,
            storage,
            plan: crate::packed::Plan::select(),
            widened: None,
        })
    }
    /// Switches a 192x64 int8 matrix to the AVX2 kernel over i16 weights.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn widen(&mut self, b: &Bundle) -> Result<()> {
        if ops::simd_tier() == 3
            && self.rows == 192
            && self.cols == 64
            && let Storage::Packed(w, _) = &self.storage
        {
            self.widened = Some(b.widened_recurrent(w)?);
        }
        Ok(())
    }
    pub fn apply(&self, x: &[f32], y: &mut [f32], q: &mut [i16]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        #[cfg(target_arch = "x86_64")]
        if let (Some(w), Storage::Packed(_, s)) = (&self.widened, &self.storage) {
            let scale = ops::quantize_i16(x, &mut q[..self.cols]);
            crate::packed::widened_one(y, w, s, &q[..self.cols], scale);
            return;
        }
        match &self.storage {
            Storage::Float(w) => {
                for (r, o) in y.iter_mut().enumerate() {
                    *o = ops::vdot_f32(&w[r * self.cols..(r + 1) * self.cols], x);
                }
            }
            Storage::Packed(w, s) => {
                let scale = ops::quantize_i16(x, &mut q[..self.cols]);
                self.plan
                    .apply(y, w, s, &q[..self.cols], &[scale], self.rows, self.cols, 1);
            }
        }
    }
    /// Contiguous `[frequency, channel]` vectors. Only 4*cols i16 scratch is needed.
    /// Each 16-byte weight load serves four independent vectors on AVX2.
    pub fn batch(&self, x: &[f32], y: &mut [f32], count: usize, q: &mut [i16]) {
        debug_assert_eq!(x.len(), count * self.cols);
        debug_assert_eq!(y.len(), count * self.rows);
        let n = self.cols;
        let m = self.rows;
        let mut pos = 0;
        if let Storage::Packed(w, s) = &self.storage {
            while pos + 4 <= count {
                let mut sx = [0.0f32; 4];
                for k in 0..4 {
                    sx[k] = ops::quantize_i16(
                        &x[(pos + k) * n..(pos + k + 1) * n],
                        &mut q[k * n..(k + 1) * n],
                    );
                }
                self.plan.apply(
                    &mut y[pos * m..(pos + 4) * m],
                    w,
                    s,
                    &q[..4 * n],
                    &sx,
                    m,
                    n,
                    4,
                );
                pos += 4;
            }
        }
        while pos < count {
            self.apply(
                &x[pos * n..(pos + 1) * n],
                &mut y[pos * m..(pos + 1) * m],
                q,
            );
            pos += 1;
        }
    }

    pub(crate) fn is_quantized(&self) -> bool {
        matches!(&self.storage, Storage::Packed(..))
    }
    /// Consumes already quantized per-frequency vectors. No rounding/scaling is
    /// repeated; each direction keeps its own weights and per-output scales.
    pub(crate) fn batch_prequantized(&self, q: &[i16], sx: &[f32], y: &mut [f32], count: usize) {
        let Storage::Packed(w, s) = &self.storage else {
            panic!("prequantized input needs an int8 matrix");
        };
        assert_eq!(q.len(), count * self.cols);
        assert_eq!(sx.len(), count);
        assert_eq!(y.len(), count * self.rows);
        let n = self.cols;
        let m = self.rows;
        let mut pos = 0;
        while pos + 4 <= count {
            self.plan.apply(
                &mut y[pos * m..(pos + 4) * m],
                w,
                s,
                &q[pos * n..(pos + 4) * n],
                &sx[pos..pos + 4],
                m,
                n,
                4,
            );
            pos += 4;
        }
        while pos < count {
            self.plan.apply(
                &mut y[pos * m..(pos + 1) * m],
                w,
                s,
                &q[pos * n..(pos + 1) * n],
                &sx[pos..pos + 1],
                m,
                n,
                1,
            );
            pos += 1;
        }
    }
}

#[derive(Clone)]
pub struct GruWeights {
    pub input: Matrix,
    pub recurrent: Matrix,
    pub bias: F32s,
    pub hidden: usize,
}
impl GruWeights {
    pub fn load(b: &Bundle, v: &Value) -> Result<Self> {
        let hidden = num(v, "hidden")?;
        require(
            hidden == 64 || hidden == 256,
            "unsupported GRU hidden width",
        )?;
        let input = Matrix::load(b, &v["input"])?;
        let recurrent = Matrix::load(b, &v["recurrent"])?;
        require(
            input.rows == 3 * hidden && recurrent.rows == 3 * hidden && recurrent.cols == hidden,
            "GRU matrix contract mismatch",
        )?;
        require(
            string(v, "gate_order")? == "zrn" && string(v, "reset")? == "after",
            "wrong GRU convention",
        )?;
        let bias = b.f32s(&v["bias"], 6 * hidden)?;
        Ok(Self {
            input,
            recurrent,
            bias,
            hidden,
        })
    }
    pub fn update(&self, wx: &[f32], rh: &[f32], h: &mut [f32]) {
        ops::gru_update(h, wx, rh, &self.bias);
    }
}

pub struct LayerNorm {
    gamma: F32s,
    beta: F32s,
    eps: f32,
}
impl LayerNorm {
    pub fn load(b: &Bundle, v: &Value, c: usize) -> Result<Self> {
        let eps = float(v, "eps")?;
        require(eps > 0.0, "invalid LayerNorm epsilon")?;
        Ok(Self {
            gamma: b.f32s(&v["gamma"], c)?,
            beta: b.f32s(&v["beta"], c)?,
            eps,
        })
    }
    /// PyTorch biased variance (divide by C, not C-1), over channels only.
    pub fn apply_add(&self, x: &mut [f32], residual: &[f32]) {
        let c = x.len();
        let mean = x.iter().sum::<f32>() / c as f32;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + self.eps).sqrt();
        for i in 0..c {
            x[i] = (x[i] - mean) * inv * self.gamma[i] + self.beta[i] + residual[i];
        }
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::test_support::{Writer, assert_bits};
    #[test]
    fn single_batch_and_prequantized_have_identical_bits() {
        for (m, n) in [(192, 64), (768, 256), (24, 6)] {
            let mut w = Writer::new();
            let v = w.matrix(m, n, true);
            let b = w.finish();
            let matrix = Matrix::load(&b, &v).unwrap();
            for count in [1, 2, 3, 4, 5, 8] {
                let x: Vec<f32> = (0..count * n)
                    .map(|i| ((i * 53 % 211) as f32 - 105.0) * 0.007)
                    .collect();
                let mut base = vec![0.0; count * m];
                let mut batch = base.clone();
                let mut shared = base.clone();
                let mut q = vec![0; 4 * n];
                let mut full = vec![0; count * n];
                let mut scales = vec![0.0; count];
                for k in 0..count {
                    matrix.apply(
                        &x[k * n..(k + 1) * n],
                        &mut base[k * m..(k + 1) * m],
                        &mut q,
                    );
                    scales[k] =
                        ops::quantize_i16(&x[k * n..(k + 1) * n], &mut full[k * n..(k + 1) * n]);
                }
                matrix.batch(&x, &mut batch, count, &mut q);
                matrix.batch_prequantized(&full, &scales, &mut shared, count);
                assert_bits(&batch, &base);
                assert_bits(&shared, &base);
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod widened_tests {
    use super::*;
    use crate::test_support::{Writer, assert_bits};
    #[test]
    fn widened_recurrent_is_exact_and_shared() {
        let mut writer = Writer::new();
        let desc = writer.matrix(192, 64, true);
        let bundle = writer.finish();
        let base = Matrix::load(&bundle, &desc).unwrap();
        let mut candidate = base.clone();
        candidate.widen(&bundle).unwrap();
        let mut second = base.clone();
        second.widen(&bundle).unwrap();
        if ops::simd_tier() == 3 {
            let (a, b) = (candidate.widened.as_ref(), second.widened.as_ref());
            assert_eq!(a.unwrap().as_ptr(), b.unwrap().as_ptr());
        }
        let mut q = [0i16; 64];
        let mut want = [0.0; 192];
        let mut got = [0.0; 192];
        for trial in 0..50 {
            let x: Vec<f32> = (0..64)
                .map(|i| {
                    if trial % 7 == 0 {
                        0.0
                    } else {
                        (((i * 131 + trial * 7) % 257) as f32 - 128.0) * 0.00013
                    }
                })
                .collect();
            base.apply(&x, &mut want, &mut q);
            candidate.apply(&x, &mut got, &mut q);
            assert_bits(&got, &want);
        }
    }
}
