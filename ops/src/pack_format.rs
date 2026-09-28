//! Pair-output8-v1 container for the two embedded DFN weight sets.
//! It permutes bytes only; floats, scales, gate rows and biases are untouched.
//! Build-time and Rust runtime share this parser. Python converter is independent.
use std::ops::Range;
pub const MAGIC: &[u8; 8] = b"DFNPAIR1";
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub hidden: usize,
    pub floats: usize,
    pub integers: usize,
    pub scales: usize,
}
impl Geometry {
    pub const DFN3: Self = Self {
        hidden: 256,
        floats: 168561,
        integers: 1966080,
        scales: 7680,
    };
    pub const DFN3LL: Self = Self {
        hidden: 512,
        floats: 302064,
        integers: 9437184,
        scales: 18432,
    };
    pub fn raw_len(self) -> usize {
        4 * self.floats + self.integers + 4 * self.scales
    }
    pub fn matrix_len(self) -> usize {
        3 * self.hidden * self.hidden
    }
    pub fn matrices(self) -> usize {
        self.integers / self.matrix_len()
    }
}
#[derive(Debug)]
pub struct Sections {
    pub f: Range<usize>,
    pub i: Range<usize>,
    pub s: Range<usize>,
    pub packed: bool,
}
fn a64(n: usize) -> usize {
    (n + 63) & !63
}
fn u32at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
pub fn sections(b: &[u8], g: Geometry) -> Result<Sections, String> {
    if !b.starts_with(MAGIC) {
        if b.len() != g.raw_len() {
            return Err(format!(
                "raw weight length {}, expected {}",
                b.len(),
                g.raw_len()
            ));
        }
        return Ok(Sections {
            f: 0..4 * g.floats,
            i: 4 * g.floats..4 * g.floats + g.integers,
            s: 4 * g.floats + g.integers..g.raw_len(),
            packed: false,
        });
    }
    if b.len() < 64 {
        return Err("truncated DFNPAIR1 header".into());
    }
    if u32at(b, 8) != 1
        || u32at(b, 12) != 64
        || u32at(b, 16) as usize != g.hidden
        || u32at(b, 20) as usize != g.floats
        || u32at(b, 24) as usize != g.integers
        || u32at(b, 28) as usize != g.scales
    {
        return Err("DFNPAIR1 geometry/version mismatch".into());
    }
    // Use canonical offsets, never trust unchecked offsets/lengths from a file.
    let f = 64..64 + 4 * g.floats;
    let i = a64(f.end)..a64(f.end) + g.integers;
    let s = a64(i.end)..a64(i.end) + 4 * g.scales;
    if u64at(b, 32) != f.start as u64
        || u64at(b, 40) != i.start as u64
        || u64at(b, 48) != s.start as u64
        || u64at(b, 56) != s.end as u64
        || b.len() != s.end
    {
        return Err("DFNPAIR1 noncanonical offsets/length".into());
    }
    if b[f.end..i.start]
        .iter()
        .chain(b[i.end..s.start].iter())
        .any(|&x| x != 0)
    {
        return Err("nonzero DFNPAIR1 padding".into());
    }
    Ok(Sections {
        f,
        i,
        s,
        packed: true,
    })
}
pub fn pack_matrix(src: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if rows == 0
        || cols == 0
        || rows % 8 != 0
        || cols % 2 != 0
        || rows.checked_mul(cols) != Some(src.len())
    {
        return Err("invalid pack geometry".into());
    }
    let mut dst = vec![0; src.len()];
    for r in 0..rows {
        for j in 0..cols {
            dst[(r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2] = src[r * cols + j];
        }
    }
    Ok(dst)
}
pub fn unpack_matrix(src: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if rows == 0
        || cols == 0
        || rows % 8 != 0
        || cols % 2 != 0
        || rows.checked_mul(cols) != Some(src.len())
    {
        return Err("invalid unpack geometry".into());
    }
    let mut dst = vec![0; src.len()];
    for r in 0..rows {
        for j in 0..cols {
            dst[r * cols + j] = src[(r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2];
        }
    }
    Ok(dst)
}
pub fn convert(raw: &[u8], g: Geometry) -> Result<Vec<u8>, String> {
    let old = sections(raw, g)?;
    if old.packed {
        return Ok(raw.to_vec());
    }
    let f = 64;
    let i = a64(f + 4 * g.floats);
    let s = a64(i + g.integers);
    let end = s + 4 * g.scales;
    let mut b = vec![0; end];
    b[..8].copy_from_slice(MAGIC);
    for (o, n) in [
        (8, 1),
        (12, 64),
        (16, g.hidden as u32),
        (20, g.floats as u32),
        (24, g.integers as u32),
        (28, g.scales as u32),
    ] {
        b[o..o + 4].copy_from_slice(&n.to_le_bytes());
    }
    for (o, n) in [(32, f), (40, i), (48, s), (56, end)] {
        b[o..o + 8].copy_from_slice(&(n as u64).to_le_bytes());
    }
    b[f..f + 4 * g.floats].copy_from_slice(&raw[old.f]);
    b[s..end].copy_from_slice(&raw[old.s]);
    let len = g.matrix_len();
    for m in 0..g.matrices() {
        let packed = pack_matrix(
            &raw[old.i.start + m * len..old.i.start + (m + 1) * len],
            3 * g.hidden,
            g.hidden,
        )?;
        b[i + m * len..i + (m + 1) * len].copy_from_slice(&packed);
    }
    Ok(b)
}
