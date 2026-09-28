//! ERB sub-band merge/split, matching LocalVQE `gtcrn.cpp`.
//! Low 65 bins pass through; the upper 192 bins compress to 64 ERB bands
//! (`erb.bm`, 64×192) and expand back (`erb.bs`, 192×64).

pub const LOW: usize = 65;
pub const HIGH_FULL: usize = 192; // 257 - 65
pub const HIGH_ERB: usize = 64;
pub const BANDED: usize = LOW + HIGH_ERB; // 129
pub const FULL: usize = LOW + HIGH_FULL; // 257

/// Merge one `257`-bin frame into `129` banded bins. `bm` is `erb.bm` (64×192).
#[must_use]
pub fn bm(frame: &[f32], bm: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; BANDED];
    y[..LOW].copy_from_slice(&frame[..LOW]);
    for j in 0..HIGH_ERB {
        let row = &bm[j * HIGH_FULL..j * HIGH_FULL + HIGH_FULL];
        let mut s = 0.0f32;
        for i in 0..HIGH_FULL {
            s += frame[LOW + i] * row[i];
        }
        y[LOW + j] = s;
    }
    y
}

/// Split one `129`-banded frame back to `257` bins. `bs` is `erb.bs` (192×64).
#[must_use]
pub fn bs(frame: &[f32], bs: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; FULL];
    y[..LOW].copy_from_slice(&frame[..LOW]);
    for j in 0..HIGH_FULL {
        let row = &bs[j * HIGH_ERB..j * HIGH_ERB + HIGH_ERB];
        let mut s = 0.0f32;
        for i in 0..HIGH_ERB {
            s += frame[LOW + i] * row[i];
        }
        y[LOW + j] = s;
    }
    y
}
