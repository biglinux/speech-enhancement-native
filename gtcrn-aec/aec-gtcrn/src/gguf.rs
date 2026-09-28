//! Bounded F32 GGUF reader. Parsing and aligned allocation happen at model load,
//! never in the audio callback. Tensor values are owned f32 storage: alignment
//! of Vec<u8> is not a Rust guarantee, even when file offsets are aligned.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Read;

// FxHash-style hasher: the streaming forward looks up each weight by name every
// frame, and SipHash (the std default) showed up in the profile. ~5x faster on
// short ASCII keys, no dependency.
#[derive(Default)]
struct FxHasher(u64);
impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(K);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

const GGUF_MAGIC: u32 = 0x4655_4747;
// Deliberate limits for the compact GTCRN model, not a general GGUF implementation.
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_ENTRIES: usize = 4096;
const MAX_STRING: usize = 65536;
const MAX_ARRAY: usize = 1_000_000;

struct TensorInfo {
    dims: Vec<usize>,
    data: Box<[f32]>,
}

pub struct Gguf {
    tensors: FxMap<String, TensorInfo>,
    kv_u64: FxMap<String, u64>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("GGUF offset overflow")?;
        let value = self.b.get(self.p..end).ok_or("truncated GGUF")?;
        self.p = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| "GGUF u32")?,
        ))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| "GGUF u64")?,
        ))
    }
    fn usize(&mut self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|_| "GGUF integer too large".into())
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.usize()?;
        if n > MAX_STRING {
            return Err("GGUF string too long".into());
        }
        std::str::from_utf8(self.take(n)?)
            .map(str::to_owned)
            .map_err(|_| "invalid GGUF UTF-8".into())
    }
    // Negative signed values are not unsigned geometry. Values of unknown type
    // cannot be skipped: their length is unknown, so reject rather than desync.
    fn value(&mut self, ty: u32, in_array: bool) -> Result<Option<u64>, String> {
        Ok(match ty {
            0 => Some(self.take(1)?[0] as u64),
            1 => u64::try_from(self.take(1)?[0] as i8).ok(),
            2 => Some(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => u64::try_from(i16::from_le_bytes(self.take(2)?.try_into().unwrap())).ok(),
            4 => Some(self.u32()? as u64),
            5 => u64::try_from(self.u32()? as i32).ok(),
            6 => {
                self.take(4)?;
                None
            }
            7 => {
                let b = self.take(1)?[0];
                if b > 1 {
                    return Err("invalid GGUF bool".into());
                }
                Some(b as u64)
            }
            8 => {
                self.string()?;
                None
            }
            9 => {
                if in_array {
                    return Err("nested GGUF arrays unsupported".into());
                }
                let et = self.u32()?;
                let n = self.usize()?;
                if n > MAX_ARRAY || et > 12 || et == 9 {
                    return Err("unsupported GGUF array".into());
                }
                for _ in 0..n {
                    self.value(et, true)?;
                }
                None
            }
            10 => Some(self.u64()?),
            11 => u64::try_from(self.u64()? as i64).ok(),
            12 => {
                self.take(8)?;
                None
            }
            _ => return Err(format!("unknown GGUF metadata type {ty}")),
        })
    }
}

impl Gguf {
    pub fn load(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mut bytes = Vec::new();
        // Bounded even if the file grows after open; metadata alone is not a bound.
        file.take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("read {path}: {e}"))?;
        Self::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_BYTES {
            return Err("GGUF exceeds 64 MiB budget".into());
        }
        let mut c = Cursor { b: bytes, p: 0 };
        if c.u32()? != GGUF_MAGIC {
            return Err("not a GGUF file".into());
        }
        if !matches!(c.u32()?, 2 | 3) {
            return Err("unsupported GGUF version".into());
        }
        let n_tensors = c.usize()?;
        let n_kv = c.usize()?;
        if n_tensors > MAX_ENTRIES || n_kv > MAX_ENTRIES {
            return Err("too many GGUF entries".into());
        }
        let mut kv_u64 = FxMap::default();
        let mut keys = HashSet::new();
        let mut alignment = 32usize;
        for _ in 0..n_kv {
            let key = c.string()?;
            if !keys.insert(key.clone()) {
                return Err("duplicate GGUF metadata".into());
            }
            let ty = c.u32()?;
            let value = c.value(ty, false)?;
            if key == "general.alignment" {
                alignment = usize::try_from(value.ok_or("invalid GGUF alignment type")?)
                    .map_err(|_| "GGUF alignment too large")?;
                // GGUF requires a multiple of eight. This loader intentionally
                // supports only the power-of-two subset, used by its rounding.
                if !(8..=4096).contains(&alignment) || !alignment.is_power_of_two() {
                    return Err("unsupported GGUF alignment".into());
                }
            }
            if let Some(v) = value {
                kv_u64.insert(key, v);
            }
        }
        let mut entries = Vec::new();
        let mut names = HashSet::new();
        let mut total = 0usize;
        for _ in 0..n_tensors {
            let name = c.string()?;
            if name.len() > 64 {
                return Err("GGUF tensor name exceeds 64 bytes".into());
            }
            if !names.insert(name.clone()) {
                return Err("duplicate GGUF tensor".into());
            }
            let nd = c.u32()? as usize;
            if !(1..=4).contains(&nd) {
                return Err("unsupported GGUF tensor rank".into());
            }
            let mut dims = Vec::with_capacity(nd);
            let mut n = 1usize;
            for _ in 0..nd {
                let d = c.usize()?;
                if d == 0 {
                    return Err("zero GGUF tensor dimension".into());
                }
                n = n.checked_mul(d).ok_or("GGUF tensor size overflow")?;
                dims.push(d);
            }
            if c.u32()? != 0 {
                return Err(format!("tensor {name} is not F32"));
            }
            let offset = c.usize()?;
            let size = n.checked_mul(4).ok_or("GGUF byte size overflow")?;
            total = total
                .checked_add(size)
                .ok_or("GGUF decoded size overflow")?;
            if total > MAX_BYTES || offset % alignment != 0 {
                return Err("GGUF tensor budget or alignment violation".into());
            }
            entries.push((name, dims, offset, n, size));
        }
        let data_start =
            c.p.checked_add(alignment - 1)
                .ok_or("GGUF data offset overflow")?
                & !(alignment - 1);
        if data_start > bytes.len() {
            return Err("truncated GGUF data section".into());
        }
        // Reject overlapping tensor payloads; this loader has no aliasing tensors.
        entries.sort_unstable_by_key(|e| e.2);
        let mut last_end = 0usize;
        let mut tensors = FxMap::default();
        for (name, dims, offset, n, size) in entries {
            if offset < last_end {
                return Err("overlapping GGUF tensors".into());
            }
            last_end = offset.checked_add(size).ok_or("GGUF payload overflow")?;
            let start = data_start
                .checked_add(offset)
                .ok_or("GGUF start overflow")?;
            let end = start.checked_add(size).ok_or("GGUF end overflow")?;
            let raw = bytes.get(start..end).ok_or("truncated GGUF tensor")?;
            let mut data = Vec::new();
            data.try_reserve_exact(n)
                .map_err(|_| "cannot allocate GGUF tensor")?;
            let (chunks, _) = raw.as_chunks::<4>();
            for chunk in chunks {
                let v = f32::from_le_bytes(*chunk);
                if !v.is_finite() {
                    return Err(format!("non-finite tensor {name}"));
                }
                data.push(v);
            }
            tensors.insert(
                name,
                TensorInfo {
                    dims,
                    data: data.into_boxed_slice(),
                },
            );
        }
        Ok(Self { tensors, kv_u64 })
    }

    #[must_use]
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.kv_u64.get(key).copied()
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<(&[f32], &[usize])> {
        let t = self.tensors.get(name)?;
        Some((&t.data, &t.dims))
    }

    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn minimal(value: f32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(GGUF_MAGIC.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(b"x");
        b.extend(1u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.extend(value.to_le_bytes());
        b
    }
    #[test]
    fn valid_tensor_is_owned_and_aligned() {
        let m = Gguf::parse(&minimal(0.25)).unwrap();
        let (v, dims) = m.tensor("x").unwrap();
        assert_eq!(v, &[0.25]);
        assert_eq!(dims, &[1]);
        assert_eq!((v.as_ptr() as usize) % std::mem::align_of::<f32>(), 0);
    }
    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let b = minimal(0.25);
        for end in 0..b.len() {
            assert!(Gguf::parse(&b[..end]).is_err(), "{end}");
        }
    }
    #[test]
    fn rejects_non_finite_unknown_versions_and_excess_counts() {
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(Gguf::parse(&minimal(v)).is_err());
        }
        let mut b = minimal(1.0);
        b[4..8].copy_from_slice(&4u32.to_le_bytes());
        assert!(Gguf::parse(&b).is_err());
        let mut b = minimal(1.0);
        b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Gguf::parse(&b).is_err());
    }
    #[test]
    fn cursor_rejects_unknown_types_and_nested_arrays() {
        assert!(Cursor { b: &[], p: 0 }.value(99, false).is_err());
        assert!(Cursor { b: &[], p: 0 }.value(9, true).is_err());
        assert!(Cursor { b: &[2], p: 0 }.value(7, false).is_err());
    }
}
