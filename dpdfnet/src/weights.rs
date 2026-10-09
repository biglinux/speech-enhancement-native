//! Validated, aligned, immutable weight storage shared by all instances.
//! Parsing, hashing, allocation and Arc cloning occur only during construction.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::Read,
    marker::PhantomData,
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;

const MANIFEST_LIMIT: usize = 4 * 1024 * 1024;
const WEIGHTS_LIMIT: usize = 64 * 1024 * 1024;

pub fn require(ok: bool, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(why.to_owned())
    }
}
pub fn num(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("missing/invalid integer: {key}"))
}
pub fn float(v: &Value, key: &str) -> Result<f32> {
    let x = v
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing float: {key}"))?;
    require(
        x.is_finite() && (x as f32).is_finite(),
        "non-finite parameter",
    )?;
    Ok(x as f32)
}
pub fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string: {key}"))
}
pub fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array: {key}"))
}

/// The exporter aligns every tensor to 64 bytes within the blob, so a 64-byte
/// aligned base keeps every SIMD weight load inside one cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Align64([u8; 64]);

struct AlignedBlob {
    words: Box<[Align64]>,
    bytes: usize,
}
impl AlignedBlob {
    fn new(data: &[u8]) -> Self {
        let mut words = vec![Align64([0; 64]); data.len().div_ceil(64)];
        for (word, chunk) in words.iter_mut().zip(data.chunks(64)) {
            word.0[..chunk.len()].copy_from_slice(chunk);
        }
        Self {
            words: words.into_boxed_slice(),
            bytes: data.len(),
        }
    }
}

/// `len` values of `T` at byte `offset` of a shared, immutable blob. Only this
/// module builds views, and only of f32, i8 and i16, for which every bit
/// pattern is a valid value.
#[derive(Clone)]
pub struct Tensor<T> {
    blob: Arc<AlignedBlob>,
    offset: usize,
    len: usize,
    element: PhantomData<T>,
}
pub type F32s = Tensor<f32>;
pub type I8s = Tensor<i8>;
pub type I16s = Tensor<i16>;
impl<T> Tensor<T> {
    fn view(blob: Arc<AlignedBlob>, offset: usize, len: usize) -> Self {
        assert_eq!(offset % std::mem::align_of::<T>(), 0);
        assert!(len
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|n| offset.checked_add(n))
            .is_some_and(|end| end <= blob.bytes));
        Self {
            blob,
            offset,
            len,
            element: PhantomData,
        }
    }
}
impl F32s {
    /// An aligned copy of weights derived during construction.
    pub(crate) fn aligned_copy(values: &[f32]) -> Self {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Self::view(Arc::new(AlignedBlob::new(&bytes)), 0, values.len())
    }
}
impl<T> Deref for Tensor<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: view() checked that the range is aligned for T and inside the
        // blob, which self keeps alive and nothing mutates; T accepts any bits.
        unsafe {
            let base = self.blob.words.as_ptr().cast::<u8>().add(self.offset);
            std::slice::from_raw_parts(base.cast::<T>(), self.len)
        }
    }
}

// Keeps the compact manifest bytes, not the much larger serde_json::Value tree.
pub struct Bundle {
    manifest_json: Box<[u8]>,
    blob: Arc<AlignedBlob>,
    widened: Mutex<HashMap<usize, I16s>>,
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let limit = limit as u64;
    let failed = |e: std::io::Error| format!("{}: {e}", path.display());
    let file = std::fs::File::open(path).map_err(failed)?;
    let size = file.metadata().map_err(failed)?.len();
    require(size <= limit, "model file exceeds size limit")?;
    let mut data = Vec::with_capacity(size as usize);
    // The file can grow between metadata() and the read.
    file.take(limit + 1)
        .read_to_end(&mut data)
        .map_err(failed)?;
    require(
        data.len() as u64 <= limit,
        "model file grew beyond size limit",
    )?;
    Ok(data)
}
impl Bundle {
    /// `DPDFNET_NATIVE_MODEL`, else the packaged bundle.
    pub fn default_dir() -> std::path::PathBuf {
        std::env::var_os("DPDFNET_NATIVE_MODEL").map_or_else(
            || "/usr/share/dpdfnet-native/dpdfnet2_48khz_hr-w8a16".into(),
            Into::into,
        )
    }
    pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Self>> {
        let dir = dir.as_ref();
        let m = read_bounded(&dir.join("manifest.json"), MANIFEST_LIMIT)?;
        let b = read_bounded(&dir.join("weights.bin"), WEIGHTS_LIMIT)?;
        Self::from_bytes(&m, &b)
    }
    pub fn from_bytes(manifest: &[u8], bytes: &[u8]) -> Result<Arc<Self>> {
        require(
            cfg!(target_endian = "little"),
            "big-endian targets are not supported",
        )?;
        require(
            manifest.len() <= MANIFEST_LIMIT && bytes.len() <= WEIGHTS_LIMIT,
            "model too large",
        )?;
        let manifest: Value = serde_json::from_slice(manifest).map_err(|e| e.to_string())?;
        match num(&manifest, "schema")? {
            2 => {}
            1 => {
                return Err(
                    "schema-1 bundle with row-major matrices: pack it with tools/pack_matrices.py"
                        .into(),
                )
            }
            _ => return Err("unsupported manifest schema".into()),
        }
        require(
            string(&manifest, "architecture")? == "dpdfnet-48hr-v1",
            "wrong model architecture",
        )?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        require(
            string(&manifest, "weights_sha256")? == actual,
            "weights checksum mismatch",
        )?;
        require(
            num(&manifest, "weight_bytes")? == bytes.len(),
            "declared weight size mismatch",
        )?;
        let compact = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            manifest_json: compact.into_boxed_slice(),
            blob: Arc::new(AlignedBlob::new(bytes)),
            widened: Mutex::new(HashMap::new()),
        }))
    }
    /// A 192x64 recurrent matrix widened to i16, shared by every instance.
    pub(crate) fn widened_recurrent(&self, w: &I8s) -> Result<I16s> {
        require(
            w.len() == 192 * 64 && Arc::ptr_eq(&w.blob, &self.blob),
            "only this bundle's 192x64 matrices are widened",
        )?;
        let mut cache = self.widened.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(widened) = cache.get(&w.offset) {
            return Ok(widened.clone());
        }
        // Two directions of up to 8 blocks in each of the two DPRNN branches.
        require(cache.len() < 32, "too many widened recurrent matrices")?;
        let bytes: Vec<u8> = w.iter().flat_map(|&v| i16::from(v).to_ne_bytes()).collect();
        let widened = I16s::view(Arc::new(AlignedBlob::new(&bytes)), 0, w.len());
        cache.insert(w.offset, widened.clone());
        Ok(widened)
    }
    /// Parses the manifest again; for construction, never for processing.
    pub fn manifest(&self) -> Result<Value> {
        serde_json::from_slice(&self.manifest_json).map_err(|e| e.to_string())
    }
    fn range(&self, v: &Value, kind: &str, size: usize) -> Result<(usize, usize)> {
        require(string(v, "dtype")? == kind, "wrong tensor dtype")?;
        let offset = num(v, "offset")?;
        let len = num(v, "len")?;
        let end = len
            .checked_mul(size)
            .and_then(|n| offset.checked_add(n))
            .ok_or("tensor range overflow")?;
        require(
            offset % size == 0 && end <= self.blob.bytes,
            "invalid tensor range/alignment",
        )?;
        Ok((offset, len))
    }
    pub fn f32s(&self, v: &Value, expected: usize) -> Result<F32s> {
        let (offset, len) = self.range(v, "f32", 4)?;
        require(len == expected, "f32 tensor shape mismatch")?;
        let a = F32s::view(self.blob.clone(), offset, len);
        require(a.iter().all(|v| v.is_finite()), "non-finite model weights")?;
        Ok(a)
    }
    pub fn i8s(&self, v: &Value, expected: usize) -> Result<I8s> {
        let (offset, len) = self.range(v, "i8", 1)?;
        require(len == expected, "i8 tensor shape mismatch")?;
        let a = I8s::view(self.blob.clone(), offset, len);
        require(
            a.iter().all(|&q| q != -128),
            "weights must be symmetric [-127,127]",
        )?;
        Ok(a)
    }
    pub fn weight_bytes(&self) -> usize {
        self.blob.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bundle(bytes: &[u8]) -> Arc<Bundle> {
        let m = serde_json::json!({"schema":2,"architecture":"dpdfnet-48hr-v1",
            "weights_sha256":format!("{:x}", Sha256::digest(bytes)), "weight_bytes":bytes.len()});
        Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), bytes).unwrap()
    }
    #[test]
    fn tensor_views_keep_backing_storage_alive() {
        let b = bundle(&1.25f32.to_le_bytes());
        let a = b
            .f32s(&serde_json::json!({"dtype":"f32","offset":0,"len":1}), 1)
            .unwrap();
        drop(b);
        assert_eq!(a[0], 1.25);
    }
    #[test]
    fn rejects_non_finite_tensor() {
        let b = bundle(&f32::NAN.to_le_bytes());
        assert!(b
            .f32s(&serde_json::json!({"dtype":"f32","offset":0,"len":1}), 1)
            .is_err());
    }
    #[test]
    fn rejects_misalignment_and_out_of_range() {
        let b = bundle(&[0u8; 8]);
        for offset in [1, 8, usize::MAX - 1] {
            assert!(b
                .f32s(
                    &serde_json::json!({"dtype":"f32","offset":offset,"len":1}),
                    1
                )
                .is_err());
        }
    }
    #[test]
    fn rejects_asymmetric_int8_minimum() {
        let b = bundle(&[128]);
        assert!(b
            .i8s(&serde_json::json!({"dtype":"i8","offset":0,"len":1}), 1)
            .is_err());
    }
    #[test]
    fn rejects_schema_1_with_a_repacking_hint() {
        let mut m = bundle(&[0; 4]).manifest().unwrap();
        m["schema"] = serde_json::json!(1);
        let e = Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &[0; 4]).err();
        assert!(e.unwrap().contains("pack_matrices.py"));
    }
    #[test]
    fn rejects_checksum_and_length_mismatch() {
        let b = bundle(&[0; 4]);
        let mut m = b.manifest().unwrap();
        assert!(Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &[1; 4]).is_err());
        m["weight_bytes"] = serde_json::json!(3);
        assert!(Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &[0; 4]).is_err());
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;
    #[test]
    fn owned_views_survive_move_clone_and_cross_thread_drop() {
        let words: Vec<f32> = (0..32).map(|i| i as f32 - 7.25).collect();
        let a = F32s::aligned_copy(&words);
        let a2 = a.clone();
        let p = a.as_ptr();
        drop(a);
        assert_eq!(a2.as_ptr(), p);
        std::thread::spawn(move || assert_eq!(&*a2, &words))
            .join()
            .unwrap();
        let raw = Arc::new(AlignedBlob::new(&[0, 1, 2, 127, 255]));
        let q = I8s::view(raw.clone(), 1, 4);
        let q2 = q.clone();
        drop(q);
        drop(raw);
        std::thread::spawn(move || assert_eq!(&*q2, &[1, 2, 127, -1]))
            .join()
            .unwrap();
    }
    #[test]
    fn empty_and_offset_views_are_valid() {
        let z = F32s::aligned_copy(&[]);
        assert!(z.is_empty());
        let blob = Arc::new(AlignedBlob::new(&[0u8; 128]));
        let z = F32s::view(blob.clone(), 128, 0);
        assert!(z.is_empty());
        let q = I8s::view(blob, 65, 17);
        assert_eq!(q.len(), 17);
        assert_eq!(q[16], 0);
    }
}
