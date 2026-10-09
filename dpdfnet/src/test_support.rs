//! Synthetic fixtures for kernel tests, not trained models.
use crate::weights::Bundle;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub(crate) struct Writer {
    bytes: Vec<u8>,
    seed: u32,
}
impl Writer {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            seed: 0x719337a5,
        }
    }
    fn next(&mut self) -> f32 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed as f32 / u32::MAX as f32 - 0.5) * 0.07
    }
    pub(crate) fn floats(&mut self, values: &[f32]) -> Value {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        let offset = self.bytes.len();
        for value in values {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }
        json!({"dtype":"f32", "offset":offset, "len":values.len()})
    }
    pub(crate) fn random_floats(&mut self, count: usize) -> Value {
        let v: Vec<f32> = (0..count).map(|_| self.next()).collect();
        self.floats(&v)
    }
    pub(crate) fn matrix(&mut self, rows: usize, cols: usize, quant: bool) -> Value {
        if quant {
            let offset = self.bytes.len();
            let w: Vec<i8> = (0..rows * cols)
                .map(|i| (((i * 71 + 17) % 255) as i16 - 127) as i8)
                .collect();
            self.bytes
                .extend(pack_rows(&w, rows, cols).iter().map(|&v| v as u8));
            let scales = self.floats(&vec![0.0002; rows]);
            json!({"kind":"w8a16", "rows":rows,"cols":cols, "layout":"pair-output8-v1",
                "weight":{"dtype":"i8","offset":offset,"len":rows*cols},"scales":scales})
        } else {
            let weight = self.random_floats(rows * cols);
            json!({"kind":"f32","rows":rows,"cols":cols,"weight":weight})
        }
    }
    pub(crate) fn gru(&mut self, hidden: usize, quant: bool) -> Value {
        let input = self.matrix(3 * hidden, hidden, quant);
        let recurrent = self.matrix(3 * hidden, hidden, quant);
        let bias = self.random_floats(6 * hidden);
        json!({"hidden":hidden,"input":input,"recurrent":recurrent,"bias":bias,
               "gate_order":"zrn","reset":"after"})
    }
    fn linear(&mut self, input: usize, output: usize) -> Value {
        let weight = self.random_floats(input * output);
        let bias = self.random_floats(output);
        json!({"input":input,"output":output,"groups":1,"weight":weight,"bias":bias,"activation":"none"})
    }
    fn ln(&mut self, c: usize) -> Value {
        let gamma = self.floats(&vec![1.0; c]);
        let beta = self.random_floats(c);
        json!({"eps":1e-5,"gamma":gamma,"beta":beta})
    }
    pub(crate) fn block(&mut self, quant: bool) -> Value {
        let forward = self.gru(64, quant);
        let backward = self.gru(64, quant);
        let temporal = self.gru(64, quant);
        let fc_intra = self.linear(128, 64);
        let fc_inter = self.linear(64, 64);
        let ln_intra = self.ln(64);
        let ln_inter = self.ln(64);
        json!({"forward":forward,"backward":backward,"temporal":temporal,
               "fc_intra":fc_intra,"fc_inter":fc_inter,"ln_intra":ln_intra,"ln_inter":ln_inter})
    }
    pub(crate) fn finish(self) -> Arc<Bundle> {
        let m = json!({"schema":2,"architecture":"dpdfnet-48hr-v1",
            "weight_bytes":self.bytes.len(), "weights_sha256":format!("{:x}",Sha256::digest(&self.bytes))});
        Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &self.bytes).unwrap()
    }
}
/// Row-major `[rows, cols]` to the pair-output8-v1 layout of tools/pack_matrices.py.
pub(crate) fn pack_rows(w: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    assert!(rows > 0 && rows.is_multiple_of(8) && cols > 0 && cols.is_multiple_of(2));
    assert_eq!(w.len(), rows * cols);
    let mut packed = vec![0i8; w.len()];
    for r in 0..rows {
        for j in 0..cols {
            let dst = (r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2;
            packed[dst] = w[r * cols + j];
        }
    }
    packed
}
pub(crate) fn assert_bits(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a:?} != {b:?}");
    }
}
