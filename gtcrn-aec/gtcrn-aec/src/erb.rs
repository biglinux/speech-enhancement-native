//! ERB sub-band merge/split, matching LocalVQE `gtcrn.cpp`.
//! Low 65 bins pass through; the upper 192 bins compress to 64 ERB bands
//! (`erb.bm`, 64×192) and expand back (`erb.bs`, 192×64).

pub const LOW: usize = 65;
pub const HIGH_FULL: usize = 192; // 257 - 65
pub const HIGH_ERB: usize = 64;
pub const BANDED: usize = LOW + HIGH_ERB; // 129
pub const FULL: usize = LOW + HIGH_FULL; // 257
