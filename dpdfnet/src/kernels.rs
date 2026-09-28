//! Recurrent hot path: row-wise W8A16 and 4-frequency weight reuse.
use crate::weights::{float, num, require, string, Bundle, F32s, I8s, Result};
use serde_json::Value;

#[derive(Clone)]
enum Storage {
    Float(F32s),
    Quant(I8s, F32s),
    Packed(I8s, F32s),
}
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    storage: Storage,
    matvec4: Matvec4,
    packed_plan: crate::packed::Plan,
    pair_plan: crate::packed::PairPlan,
    #[cfg(feature = "r9-recurrent-cache")]
    recurrent_cache: Option<crate::weights::I16s>,
    #[cfg(feature = "r10-batch-cache")]
    batch_cache: Option<crate::weights::I16s>,
    #[cfg(feature = "r10-batch-cache")]
    batch_cache_plan: crate::packed::BatchCachePlan,
}
/// Dequantized batched GEMV over four contiguous vectors: writes
/// `out[k*m + r] = sum_r(vector k) * scales[r] * sx[k]` for r in 0..m, k in 0..4.
/// Loops the rows inside one selected kernel rather than calling `dot4` per row.
type Matvec4 = fn(&mut [f32], &[i8], &[f32], &[i16], &[f32; 4], usize, usize);
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
                let q = b.i8s(&v["weight"], rows * cols)?;
                let scales = b.f32s(&v["scales"], rows)?;
                require(
                    scales.iter().all(|&s| s > 0.0),
                    "invalid quantization scale",
                )?;
                let layout = v
                    .get("layout")
                    .and_then(Value::as_str)
                    .unwrap_or("row-major-v1");
                match layout {
                    "pair-output8-v1" => {
                        require(rows % 8 == 0 && cols % 2 == 0, "invalid packed dimensions")?;
                        require(b.schema() == 2, "packed layout requires schema 2")?;
                        Storage::Packed(q, scales)
                    }
                    "row-major-v1" => {
                        if cfg!(all(
                            feature = "packed-gru",
                            not(feature = "scalar-reference")
                        )) && rows % 8 == 0
                            && cols % 2 == 0
                        {
                            Storage::Packed(b.packed_i8s(&v["weight"], rows, cols)?, scales)
                        } else {
                            Storage::Quant(q, scales)
                        }
                    }
                    _ => return Err("unknown matrix layout".into()),
                }
            }
            _ => return Err("unknown matrix format".into()),
        };
        let matvec4: Matvec4 = select_matvec4();
        Ok(Self {
            rows,
            cols,
            storage,
            matvec4,
            packed_plan: crate::packed::Plan::select(),
            pair_plan: crate::packed::PairPlan::select(),
            #[cfg(feature = "r9-recurrent-cache")]
            recurrent_cache: None,
            #[cfg(feature = "r10-batch-cache")]
            batch_cache: None,
            #[cfg(feature = "r10-batch-cache")]
            batch_cache_plan: crate::packed::BatchCachePlan::select(),
        })
    }
    #[cfg(feature = "r9-recurrent-cache")]
    pub(crate) fn enable_recurrent_cache(&mut self, b: &Bundle) -> Result<()> {
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar-reference")))]
        if dfn_ops::simd_tier() == 3 && self.rows == 192 && self.cols == 64 {
            if let Storage::Packed(w, _) = &self.storage {
                self.recurrent_cache = Some(b.recurrent_cache(w)?);
            }
        }
        let _ = b;
        Ok(())
    }
    #[cfg(feature = "r10-batch-cache")]
    pub(crate) fn enable_batch_cache(&mut self, b: &Bundle) -> Result<()> {
        if self.rows == 192 && self.cols == 64 && self.batch_cache_plan.available() {
            if let Storage::Packed(w, _) = &self.storage {
                self.batch_cache = Some(b.batch_i16_cache(w)?);
            }
        }
        Ok(())
    }
    pub fn apply(&self, x: &[f32], y: &mut [f32], q: &mut [i16]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        #[cfg(all(target_arch = "x86_64", feature = "r9-recurrent-cache"))]
        if let (Some(w), Storage::Packed(_, s)) = (&self.recurrent_cache, &self.storage) {
            let scale = dfn_ops::quantize_i16(x, &mut q[..self.cols]);
            crate::packed::cached_recurrent_one(y, w, s, &q[..self.cols], scale);
            return;
        }
        match &self.storage {
            Storage::Float(w) => {
                for (r, o) in y.iter_mut().enumerate() {
                    *o = dot_f32(&w[r * self.cols..(r + 1) * self.cols], x);
                }
            }
            Storage::Packed(w, s) => {
                let scale = dfn_ops::quantize_i16(x, &mut q[..self.cols]);
                self.packed_plan
                    .apply(y, w, s, &q[..self.cols], &[scale], self.rows, self.cols, 1);
            }
            Storage::Quant(w, s) => {
                let scale = dfn_ops::quantize_i16(x, &mut q[..self.cols]);
                #[cfg(not(feature = "scalar-reference"))]
                dfn_ops::matvec_i8_i16(y, w, s, &q[..self.cols], scale, self.rows, self.cols);
                #[cfg(feature = "scalar-reference")]
                for r in 0..self.rows {
                    let a: i32 = w[r * self.cols..(r + 1) * self.cols]
                        .iter()
                        .zip(&q[..self.cols])
                        .map(|(&a, &b)| a as i32 * b as i32)
                        .sum();
                    y[r] = a as f32 * s[r] * scale;
                }
            }
        }
    }
    /// Contiguous [frequency,channel] vectors. Only 4*cols i16 scratch is needed.
    /// Each 16-byte weight load serves four independent vectors on AVX2.
    pub fn batch(&self, x: &[f32], y: &mut [f32], count: usize, q: &mut [i16]) {
        debug_assert_eq!(x.len(), count * self.cols);
        debug_assert_eq!(y.len(), count * self.rows);
        let n = self.cols;
        let m = self.rows;
        let mut pos = 0;
        if matches!(&self.storage, Storage::Quant(..) | Storage::Packed(..)) {
            while pos + 4 <= count {
                let mut sx = [0.0f32; 4];
                for k in 0..4 {
                    sx[k] = dfn_ops::quantize_i16(
                        &x[(pos + k) * n..(pos + k + 1) * n],
                        &mut q[k * n..(k + 1) * n],
                    );
                }
                self.apply_quantized_four(&q[..4 * n], &sx, &mut y[pos * m..(pos + 4) * m]);
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

    /// Two independent matrices sharing exactly the same already-quantized inputs.
    /// Used only by the optional F/B schedule. Biases and all recurrent updates
    /// remain outside this operation. Mixed/unpacked/scalar paths remain valid.
    pub(crate) fn batch_pair_prequantized(
        &self,
        other: &Self,
        q: &[i16],
        sx: &[f32],
        ya: &mut [f32],
        yb: &mut [f32],
        count: usize,
    ) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        assert!(self.is_quantized() && other.is_quantized());
        let (m, n) = (self.rows, self.cols);
        assert_eq!(q.len(), count * n);
        assert_eq!(sx.len(), count);
        assert_eq!(ya.len(), count * m);
        assert_eq!(yb.len(), count * m);
        if let (Storage::Packed(wa, sa), Storage::Packed(wb, sb)) = (&self.storage, &other.storage)
        {
            if self.pair_plan.available() {
                let mut p = 0;
                while p + 4 <= count {
                    self.pair_plan.apply(
                        &mut ya[p * m..(p + 4) * m],
                        &mut yb[p * m..(p + 4) * m],
                        wa,
                        wb,
                        sa,
                        sb,
                        &q[p * n..(p + 4) * n],
                        &sx[p..p + 4],
                        m,
                        n,
                    );
                    p += 4;
                }
                self.batch_prequantized(&q[p * n..], &sx[p..], &mut ya[p * m..], count - p);
                other.batch_prequantized(&q[p * n..], &sx[p..], &mut yb[p * m..], count - p);
                return;
            }
        }
        self.batch_prequantized(q, sx, ya, count);
        other.batch_prequantized(q, sx, yb, count);
    }

    pub(crate) fn is_quantized(&self) -> bool {
        matches!(&self.storage, Storage::Quant(..) | Storage::Packed(..))
    }
    fn apply_quantized_four(&self, q: &[i16], sx: &[f32; 4], out: &mut [f32]) {
        #[cfg(feature = "r10-batch-cache")]
        if let (Some(w), Storage::Packed(_, s)) = (&self.batch_cache, &self.storage) {
            self.batch_cache_plan.apply(out, w, s, q, sx);
            return;
        }
        match &self.storage {
            Storage::Quant(w, s) => (self.matvec4)(out, w, s, q, sx, self.rows, self.cols),
            Storage::Packed(w, s) => self
                .packed_plan
                .apply(out, w, s, q, sx, self.rows, self.cols, 4),
            Storage::Float(_) => unreachable!("prequantization only for integer matrices"),
        }
    }
    /// Consumes already quantized per-frequency vectors. No rounding/scaling is
    /// repeated; each direction keeps its own weights and per-output scales.
    pub(crate) fn batch_prequantized(&self, q: &[i16], sx: &[f32], y: &mut [f32], count: usize) {
        assert!(self.is_quantized());
        assert_eq!(q.len(), count * self.cols);
        assert_eq!(sx.len(), count);
        assert_eq!(y.len(), count * self.rows);
        let n = self.cols;
        let m = self.rows;
        let mut pos = 0;
        while pos + 4 <= count {
            let scales = [sx[pos], sx[pos + 1], sx[pos + 2], sx[pos + 3]];
            self.apply_quantized_four(
                &q[pos * n..(pos + 4) * n],
                &scales,
                &mut y[pos * m..(pos + 4) * m],
            );
            pos += 4;
        }
        while pos < count {
            let out = &mut y[pos * m..(pos + 1) * m];
            let x = &q[pos * n..(pos + 1) * n];
            match &self.storage {
                Storage::Packed(w, s) => self.packed_plan.apply(out, w, s, x, &[sx[pos]], m, n, 1),
                Storage::Quant(w, s) => {
                    // Diagnostic tail, also used by scalar-reference. No float reduction.
                    for r in 0..m {
                        let mut sum = 0i32;
                        for j in 0..n {
                            sum += w[r * n + j] as i32 * x[j] as i32;
                        }
                        out[r] = (sum as f32 * s[r]) * sx[pos];
                    }
                }
                Storage::Float(_) => unreachable!(),
            }
            pos += 1;
        }
    }
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "scalar-reference")]
    {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
    #[cfg(not(feature = "scalar-reference"))]
    {
        dfn_ops::vdot_f32(a, b)
    }
}
fn dot4_scalar(w: &[i8], x: &[i16], n: usize) -> [i32; 4] {
    let mut s = [0i32; 4];
    for j in 0..n {
        for k in 0..4 {
            s[k] += w[j] as i32 * x[k * n + j] as i32;
        }
    }
    s
}
// n<=1024, |w|<=127 and |x|<=16383 => every partial and total sum fits i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot4_avx2(w: &[i8], x: &[i16], n: usize) -> [i32; 4] {
    use std::arch::x86_64::*;
    let mut a = [_mm256_setzero_si256(); 4];
    let end = n & !15;
    let mut j = 0;
    while j < end {
        let ww = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(j).cast()));
        for k in 0..4 {
            let xx = _mm256_loadu_si256(x.as_ptr().add(k * n + j).cast());
            a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(ww, xx));
        }
        j += 16;
    }
    let mut s = [0i32; 4];
    for k in 0..4 {
        let mut h = _mm_add_epi32(
            _mm256_castsi256_si128(a[k]),
            _mm256_extracti128_si256(a[k], 1),
        );
        h = _mm_hadd_epi32(h, h);
        h = _mm_hadd_epi32(h, h);
        s[k] = _mm_cvtsi128_si32(h);
        for jj in end..n {
            s[k] += w[jj] as i32 * x[k * n + jj] as i32;
        }
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot4_sse(w: &[i8], x: &[i16], n: usize) -> [i32; 4] {
    use std::arch::x86_64::*;
    let mut a = [_mm_setzero_si128(); 4];
    let end = n & !7;
    let mut j = 0;
    while j < end {
        let ww = _mm_cvtepi8_epi16(_mm_loadl_epi64(w.as_ptr().add(j).cast()));
        for k in 0..4 {
            let xx = _mm_loadu_si128(x.as_ptr().add(k * n + j).cast());
            a[k] = _mm_add_epi32(a[k], _mm_madd_epi16(ww, xx));
        }
        j += 8;
    }
    let mut s = [0i32; 4];
    for k in 0..4 {
        let h = _mm_add_epi32(a[k], _mm_srli_si128(a[k], 8));
        let h = _mm_add_epi32(h, _mm_srli_si128(h, 4));
        s[k] = _mm_cvtsi128_si32(h);
        for jj in end..n {
            s[k] += w[jj] as i32 * x[k * n + jj] as i32;
        }
    }
    s
}

fn select_matvec4() -> Matvec4 {
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar-reference")))]
    if !cfg!(feature = "force-sse41")
        && !cfg!(feature = "force-avx1")
        && std::is_x86_feature_detected!("avx2")
    {
        return matvec4_avx2_checked;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar-reference")))]
    if std::is_x86_feature_detected!("sse4.1") {
        return matvec4_sse_checked;
    }
    matvec4_scalar
}
fn matvec4_scalar(
    out: &mut [f32],
    w: &[i8],
    scales: &[f32],
    q4: &[i16],
    sx: &[f32; 4],
    m: usize,
    n: usize,
) {
    for r in 0..m {
        let s = dot4_scalar(&w[r * n..(r + 1) * n], q4, n);
        for k in 0..4 {
            out[k * m + r] = s[k] as f32 * scales[r] * sx[k];
        }
    }
}
#[cfg(target_arch = "x86_64")]
fn matvec4_avx2_checked(
    out: &mut [f32],
    w: &[i8],
    scales: &[f32],
    q4: &[i16],
    sx: &[f32; 4],
    m: usize,
    n: usize,
) {
    // Selected only after CPUID; same validated integer bounds as `dot4_avx2`.
    unsafe { matvec4_avx2(out, w, scales, q4, sx, m, n) }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matvec4_avx2(
    out: &mut [f32],
    w: &[i8],
    scales: &[f32],
    q4: &[i16],
    sx: &[f32; 4],
    m: usize,
    n: usize,
) {
    // The row loop lives inside this one target-feature function, so `dot4_avx2`
    // inlines and there is no per-row indirect call or CPUID re-dispatch.
    for r in 0..m {
        let s = dot4_avx2(&w[r * n..(r + 1) * n], q4, n);
        for k in 0..4 {
            out[k * m + r] = s[k] as f32 * scales[r] * sx[k];
        }
    }
}
#[cfg(target_arch = "x86_64")]
fn matvec4_sse_checked(
    out: &mut [f32],
    w: &[i8],
    scales: &[f32],
    q4: &[i16],
    sx: &[f32; 4],
    m: usize,
    n: usize,
) {
    unsafe { matvec4_sse(out, w, scales, q4, sx, m, n) }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn matvec4_sse(
    out: &mut [f32],
    w: &[i8],
    scales: &[f32],
    q4: &[i16],
    sx: &[f32; 4],
    m: usize,
    n: usize,
) {
    for r in 0..m {
        let s = dot4_sse(&w[r * n..(r + 1) * n], q4, n);
        for k in 0..4 {
            out[k * m + r] = s[k] as f32 * scales[r] * sx[k];
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
        #[cfg(not(feature = "scalar-reference"))]
        dfn_ops::gru_update(h, wx, rh, &self.bias);
        #[cfg(feature = "scalar-reference")]
        {
            let n = self.hidden;
            for i in 0..n {
                let z =
                    1.0 / (1.0 + (-(wx[i] + rh[i] + self.bias[i] + self.bias[3 * n + i])).exp());
                let r = 1.0
                    / (1.0
                        + (-(wx[n + i] + rh[n + i] + self.bias[n + i] + self.bias[4 * n + i]))
                            .exp());
                let a = (wx[2 * n + i]
                    + self.bias[2 * n + i]
                    + r * (rh[2 * n + i] + self.bias[5 * n + i]))
                    .tanh();
                h[i] = (1.0 - z) * a + z * h[i];
            }
        }
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
    /// PyTorch biased variance (divide by C, NOT C-1), over channels only.
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
mod tests {
    use super::*;
    #[test]
    fn four_dots_match_scalar() {
        for n in [1, 15, 16, 31, 64, 256, 1024] {
            let w: Vec<i8> = (0..n)
                .map(|i| (i % 255) as i16 - 127)
                .map(|v| v as i8)
                .collect();
            let x: Vec<i16> = (0..4 * n)
                .map(|i| ((i * 719) % 32767) as i32 - 16383)
                .map(|v| v as i16)
                .collect();
            let want = dot4_scalar(&w, &x, n);
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(want, dot4_avx2(&w, &x, n));
                }
                if std::is_x86_feature_detected!("sse4.1") {
                    assert_eq!(want, dot4_sse(&w, &x, n));
                }
            }
        }
    }
    #[test]
    fn quantized_sum_does_not_overflow() {
        assert!(127i64 * 16383 * 1024 < i32::MAX as i64);
        let w = vec![127; 1024];
        let x = vec![16383; 4096];
        assert_eq!(dot4_scalar(&w, &x, 1024), [127 * 16383 * 1024; 4]);
    }
}

#[cfg(test)]
mod round3_tests {
    use super::*;
    use crate::test_support::{assert_bits, Writer};
    #[test]
    fn single_batch_and_prequantized_have_identical_bits() {
        for (m, n) in [(192, 64), (768, 256), (24, 6), (13, 7)] {
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
                    scales[k] = dfn_ops::quantize_i16(
                        &x[k * n..(k + 1) * n],
                        &mut full[k * n..(k + 1) * n],
                    );
                }
                matrix.batch(&x, &mut batch, count, &mut q);
                matrix.batch_prequantized(&full, &scales, &mut shared, count);
                assert_bits(&batch, &base);
                assert_bits(&shared, &base);
            }
        }
    }
    #[test]
    fn compatibility_pack_is_shared_once_per_bundle() {
        let mut w = Writer::new();
        let v = w.matrix(192, 64, true);
        let b = w.finish();
        let _a = Matrix::load(&b, &v).unwrap();
        let size = b.compatibility_packed_bytes();
        let _z = Matrix::load(&b, &v).unwrap();
        assert_eq!(size, b.compatibility_packed_bytes());
        if cfg!(all(
            feature = "packed-gru",
            not(feature = "scalar-reference")
        )) {
            assert_eq!(size, 192 * 64);
        } else {
            assert_eq!(size, 0);
        }
    }
}

#[cfg(test)]
mod round8_matrix_tests {
    use super::*;
    use crate::test_support::{assert_bits, Writer};
    #[test]
    fn paired_api_matches_two_calls_including_tail_and_empty_batch() {
        for count in [0usize, 1, 3, 4, 5, 8, 40, 48] {
            let mut w = Writer::new();
            let a = w.matrix(192, 64, true);
            let mut b = w.matrix(192, 64, true);
            b["scales"] = w.floats(&vec![0.00037; 192]);
            let blob = w.finish();
            let a = Matrix::load(&blob, &a).unwrap();
            let b = Matrix::load(&blob, &b).unwrap();
            let x: Vec<f32> = (0..count * 64)
                .map(|i| ((i * 151 % 331) as f32 - 165.0) * 0.01)
                .collect();
            let mut q = vec![0i16; count * 64];
            let mut scales = vec![0.0; count];
            for i in 0..count {
                scales[i] =
                    dfn_ops::quantize_i16(&x[i * 64..i * 64 + 64], &mut q[i * 64..i * 64 + 64]);
            }
            let mut ya = vec![0.0; count * 192];
            let mut yb = ya.clone();
            let mut ra = ya.clone();
            let mut rb = ya.clone();
            a.batch_pair_prequantized(&b, &q, &scales, &mut ya, &mut yb, count);
            a.batch_prequantized(&q, &scales, &mut ra, count);
            b.batch_prequantized(&q, &scales, &mut rb, count);
            assert_bits(&ya, &ra);
            assert_bits(&yb, &rb);
        }
    }
}

#[cfg(all(test, feature = "r9-recurrent-cache"))]
mod cache_tests {
    use super::*;
    use crate::test_support::{assert_bits, Writer};
    #[test]
    fn widened_recurrent_is_exact_and_shared() {
        let mut writer = Writer::new();
        let desc = writer.matrix(192, 64, true);
        let bundle = writer.finish();
        let base = Matrix::load(&bundle, &desc).unwrap();
        let mut candidate = base.clone();
        candidate.enable_recurrent_cache(&bundle).unwrap();
        let mut second = base.clone();
        second.enable_recurrent_cache(&bundle).unwrap();
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar-reference")))]
        if dfn_ops::simd_tier() == 3 {
            assert_eq!(bundle.derived_recurrent_bytes(), 192 * 64 * 2);
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
