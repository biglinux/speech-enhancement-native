//! Validated, aligned, immutable weight storage shared by all instances.
//! Parsing, hashing, allocation and Arc cloning occur only during construction.
use serde_json::Value;

// Diagnostic only: counters are compiled out of every performance candidate.
#[cfg(feature = "r10-weight-audit")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "r10-weight-audit")]
static I8_RESOLUTIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "r10-weight-audit")]
static F32_RESOLUTIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "r10-weight-audit")]
static I16_RESOLUTIONS: AtomicU64 = AtomicU64::new(0);
/// Construction-call counters. Read outside the callback, never reset globally.
#[cfg(feature = "r10-weight-audit")]
pub fn resolution_counts() -> [u64; 3] {
    [
        I8_RESOLUTIONS.load(Ordering::Relaxed),
        F32_RESOLUTIONS.load(Ordering::Relaxed),
        I16_RESOLUTIONS.load(Ordering::Relaxed),
    ]
}

use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::Read,
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
};

pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;

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

/// 64-byte-aligned backing cell. The exporter aligns each tensor to 64 bytes
/// inside the blob, but that only lands on a cache-line boundary if the base
/// allocation is itself 64-aligned; `Box<[u64]>` guaranteed only 8. Aligning the
/// base keeps every packed-weight and FP32 SIMD load off a line-crossing address,
/// which matters most on SSE4.1 and small-cache CPUs.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Align64([u8; 64]);

struct AlignedBlob {
    words: Box<[Align64]>,
    bytes: usize,
}
impl AlignedBlob {
    fn new(data: &[u8]) -> Self {
        let mut words = vec![Align64([0u8; 64]); data.len().div_ceil(64)];
        // SAFETY: destination covers ceil(len/64)*64 bytes and does not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), words.as_mut_ptr().cast(), data.len())
        };
        let blob = Self {
            words: words.into_boxed_slice(),
            bytes: data.len(),
        };
        debug_assert_eq!(
            blob.words.as_ptr() as usize % 64,
            0,
            "blob base not 64-aligned"
        );
        blob
    }
}
#[derive(Clone)]
pub struct F32s {
    #[cfg_attr(feature = "r10-resolved-weights", allow(dead_code))]
    blob: Arc<AlignedBlob>,
    #[cfg_attr(feature = "r10-resolved-weights", allow(dead_code))]
    offset: usize,
    len: usize,
    #[cfg(feature = "r10-resolved-weights")]
    ptr: *const f32,
}
impl F32s {
    /// Initialization-only aligned copy for private prepacked convolution data.
    pub(crate) fn aligned_copy(values: &[f32]) -> Self {
        // SAFETY: immutable bytes of a live f32 slice, copied before it is dropped.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        Self::view(Arc::new(AlignedBlob::new(bytes)), 0, values.len())
    }
}
// SAFETY: the private pointer always refers into this view's immutable Arc-owned
// AlignedBlob. Moving/cloning a view does not relocate the allocation. No mutable
// access is provided; Deref ties its borrow to &self. Constructors
// validate alignment/ranges, including empty slices. Only these owning types get
// Send/Sync, not a general raw pointer wrapper.
#[cfg(feature = "r10-resolved-weights")]
unsafe impl Send for F32s {}
#[cfg(feature = "r10-resolved-weights")]
unsafe impl Sync for F32s {}
impl F32s {
    fn view(blob: Arc<AlignedBlob>, offset: usize, len: usize) -> Self {
        assert_eq!(offset % std::mem::align_of::<f32>(), 0);
        assert!(len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|n| offset.checked_add(n))
            .is_some_and(|end| end <= blob.bytes));
        #[cfg(feature = "r10-resolved-weights")]
        let ptr = unsafe { blob.words.as_ptr().cast::<u8>().add(offset).cast::<f32>() };
        Self {
            blob,
            offset,
            len,
            #[cfg(feature = "r10-resolved-weights")]
            ptr,
        }
    }
}
impl Deref for F32s {
    type Target = [f32];
    #[cfg_attr(feature = "r10-resolved-weights", inline(always))]
    fn deref(&self) -> &[f32] {
        #[cfg(feature = "r10-resolved-weights")]
        {
            // SAFETY: view() and owning-type invariants above; not a 'static borrow.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(not(feature = "r10-resolved-weights"))]
        // SAFETY: checked length, alignment, little-endian platform and finite values at load.
        // Shared allocation is immutable and kept alive by this Arc.
        unsafe {
            std::slice::from_raw_parts(
                self.blob
                    .words
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.offset)
                    .cast(),
                self.len,
            )
        }
    }
}
#[derive(Clone)]
pub struct I8s {
    blob: Arc<AlignedBlob>,
    offset: usize,
    len: usize,
    #[cfg(feature = "r10-resolved-weights")]
    ptr: *const i8,
}
// SAFETY: the private pointer always refers into this view's immutable Arc-owned
// AlignedBlob. Moving/cloning a view does not relocate the allocation. No mutable
// access is provided; Deref ties its borrow to &self. Constructors
// validate alignment/ranges, including empty slices. Only these owning types get
// Send/Sync, not a general raw pointer wrapper.
#[cfg(feature = "r10-resolved-weights")]
unsafe impl Send for I8s {}
#[cfg(feature = "r10-resolved-weights")]
unsafe impl Sync for I8s {}
impl I8s {
    fn view(blob: Arc<AlignedBlob>, offset: usize, len: usize) -> Self {
        assert_eq!(offset % std::mem::align_of::<i8>(), 0);
        assert!(len
            .checked_mul(std::mem::size_of::<i8>())
            .and_then(|n| offset.checked_add(n))
            .is_some_and(|end| end <= blob.bytes));
        #[cfg(feature = "r10-resolved-weights")]
        let ptr = unsafe { blob.words.as_ptr().cast::<u8>().add(offset).cast::<i8>() };
        Self {
            blob,
            offset,
            len,
            #[cfg(feature = "r10-resolved-weights")]
            ptr,
        }
    }
}
impl Deref for I8s {
    type Target = [i8];
    #[cfg_attr(feature = "r10-resolved-weights", inline(always))]
    fn deref(&self) -> &[i8] {
        #[cfg(feature = "r10-resolved-weights")]
        {
            // SAFETY: view() and owning-type invariants above; not a 'static borrow.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(not(feature = "r10-resolved-weights"))]
        // SAFETY: range was validated; every byte is a valid i8.
        unsafe {
            std::slice::from_raw_parts(
                self.blob
                    .words
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.offset)
                    .cast(),
                self.len,
            )
        }
    }
}
/// Exact sign-extended cache view; no dequantization or changed scales.
#[cfg(any(feature = "r9-recurrent-cache", feature = "r10-batch-cache"))]
#[derive(Clone)]
pub(crate) struct I16s {
    // Retained even with a resolved pointer: owns the readable allocation.
    #[allow(dead_code)]
    blob: Arc<AlignedBlob>,
    len: usize,
    #[cfg(feature = "r10-resolved-weights")]
    ptr: *const i16,
}
#[cfg(any(feature = "r9-recurrent-cache", feature = "r10-batch-cache"))]
impl I16s {
    fn view(blob: Arc<AlignedBlob>, len: usize) -> Self {
        assert!(len.checked_mul(2).is_some_and(|n| n <= blob.bytes));
        #[cfg(feature = "r10-resolved-weights")]
        let ptr = blob.words.as_ptr().cast::<i16>();
        Self {
            blob,
            len,
            #[cfg(feature = "r10-resolved-weights")]
            ptr,
        }
    }
}
// SAFETY: same immutable Arc-owned pointer invariants as F32s/I8s above.
#[cfg(all(
    feature = "r10-resolved-weights",
    any(feature = "r9-recurrent-cache", feature = "r10-batch-cache")
))]
unsafe impl Send for I16s {}
#[cfg(all(
    feature = "r10-resolved-weights",
    any(feature = "r9-recurrent-cache", feature = "r10-batch-cache")
))]
unsafe impl Sync for I16s {}
#[cfg(any(feature = "r9-recurrent-cache", feature = "r10-batch-cache"))]
impl Deref for I16s {
    type Target = [i16];
    #[cfg_attr(feature = "r10-resolved-weights", inline(always))]
    fn deref(&self) -> &[i16] {
        #[cfg(feature = "r10-resolved-weights")]
        {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(not(feature = "r10-resolved-weights"))]
        unsafe {
            std::slice::from_raw_parts(self.blob.words.as_ptr().cast(), self.len)
        }
    }
}

// Retain compact metadata, not the much larger serde_json::Value tree.
// Instances parse it only during construction, then discard that temporary tree.
pub struct Bundle {
    manifest_json: Box<[u8]>,
    schema: usize,
    blob: Arc<AlignedBlob>,
    // Initialization-only cache for schema-1 compatibility. Prepacked schema-2
    // bundles use the main blob directly and leave this cache empty.
    packed: Mutex<HashMap<(usize, usize, usize), I8s>>,
    #[cfg(feature = "r9-recurrent-cache")]
    recurrent: Mutex<HashMap<(usize, usize, usize), I16s>>,
    #[cfg(feature = "r10-batch-cache")]
    batch_cache: Mutex<HashMap<(usize, usize, usize), I16s>>,
}
fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    require(size <= limit, "model file exceeds size limit")?;
    let mut data = Vec::with_capacity(size as usize);
    // Limit the read itself as well: a file can grow between metadata() and read().
    file.take(limit + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    require(
        data.len() as u64 <= limit,
        "model file grew beyond size limit",
    )?;
    Ok(data)
}
impl Bundle {
    pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Self>> {
        let dir = dir.as_ref();
        // Bound external data before allocating. These are model-only, not generic tensor files.
        let mp = dir.join("manifest.json");
        let bp = dir.join("weights.bin");
        let m = read_bounded(&mp, 4 * 1024 * 1024)?;
        let b = read_bounded(&bp, 64 * 1024 * 1024)?;
        Self::from_bytes(&m, &b)
    }
    pub fn from_bytes(manifest: &[u8], bytes: &[u8]) -> Result<Arc<Self>> {
        require(
            cfg!(target_endian = "little"),
            "big-endian targets are not supported",
        )?;
        require(
            manifest.len() <= 4 * 1024 * 1024 && bytes.len() <= 64 * 1024 * 1024,
            "model too large",
        )?;
        let manifest: Value = serde_json::from_slice(manifest).map_err(|e| e.to_string())?;
        require(
            matches!(num(&manifest, "schema")?, 1 | 2),
            "unsupported manifest schema",
        )?;
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
            schema: num(&manifest, "schema")?,
            manifest_json: compact.into_boxed_slice(),
            blob: Arc::new(AlignedBlob::new(bytes)),
            packed: Mutex::new(HashMap::new()),
            #[cfg(feature = "r9-recurrent-cache")]
            recurrent: Mutex::new(HashMap::new()),
            #[cfg(feature = "r10-batch-cache")]
            batch_cache: Mutex::new(HashMap::new()),
        }))
    }
    #[cfg(feature = "r9-recurrent-cache")]
    pub(crate) fn recurrent_cache(&self, w: &I8s) -> Result<I16s> {
        #[cfg(feature = "r10-weight-audit")]
        I16_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
        require(
            w.len() == 192 * 64,
            "recurrent cache is restricted to 192x64",
        )?;
        let key = (Arc::as_ptr(&w.blob) as usize, w.offset, w.len);
        let mut cache = self
            .recurrent
            .lock()
            .map_err(|_| "recurrent cache poisoned")?;
        if let Some(value) = cache.get(&key) {
            return Ok(value.clone());
        }
        // At most 2 intra cells x 2 branches x 8 DPRNN blocks. This is not a
        // generic expansion of all recurrent/embedding/pointwise tensors.
        require(cache.len() < 32, "too many small recurrent caches")?;
        let data: Vec<i16> = w.iter().map(|&v| i16::from(v)).collect();
        // SAFETY: readable bytes of a live initialized i16 slice; immediately copied.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
        let out = I16s::view(Arc::new(AlignedBlob::new(bytes)), data.len());
        cache.insert(key, out.clone());
        Ok(out)
    }
    /// Initialization only. 3 Wx + temporal Rh per DPRNN block, 192x64 only.
    /// At most 2 branches * 8 blocks * 4 matrices = 64 entries. R9 spectral Rh
    /// caches are separate and are not duplicated here. This is NOT a new format.
    #[cfg(feature = "r10-batch-cache")]
    pub(crate) fn batch_i16_cache(&self, w: &I8s) -> Result<I16s> {
        #[cfg(feature = "r10-weight-audit")]
        I16_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
        require(w.len() == 192 * 64, "batch cache restricted to 192x64")?;
        let key = (Arc::as_ptr(&w.blob) as usize, w.offset, w.len);
        let mut cache = self
            .batch_cache
            .lock()
            .map_err(|_| "batch cache poisoned")?;
        if let Some(value) = cache.get(&key) {
            return Ok(value.clone());
        }
        require(cache.len() < 64, "too many small batch caches")?;
        let values: Vec<i16> = w.iter().map(|&v| i16::from(v)).collect();
        // SAFETY: initialized i16 bytes are copied into a new aligned allocation.
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) };
        let value = I16s::view(Arc::new(AlignedBlob::new(bytes)), values.len());
        cache.insert(key, value.clone());
        Ok(value)
    }
    /// Diagnostics/init only: takes a mutex. NEVER call in process/forward.
    pub fn derived_batch_bytes(&self) -> usize {
        #[cfg(feature = "r10-batch-cache")]
        {
            return self
                .batch_cache
                .lock()
                .map(|m| m.values().map(|v| v.len * 2).sum())
                .unwrap_or(0);
        }
        #[cfg(not(feature = "r10-batch-cache"))]
        {
            0
        }
    }

    /// Diagnostics only: locks initialization cache; NEVER call from process().
    pub fn derived_recurrent_bytes(&self) -> usize {
        #[cfg(feature = "r9-recurrent-cache")]
        {
            self.recurrent
                .lock()
                .map(|m| m.values().map(|v| v.len * 2).sum())
                .unwrap_or(0)
        }
        #[cfg(not(feature = "r9-recurrent-cache"))]
        {
            0
        }
    }
    /// Initialization/diagnostics only. Never called by a process/forward method.
    pub fn manifest(&self) -> Result<Value> {
        serde_json::from_slice(&self.manifest_json).map_err(|e| e.to_string())
    }
    pub(crate) fn schema(&self) -> usize {
        self.schema
    }
    pub fn metadata_bytes(&self) -> usize {
        self.manifest_json.len()
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
        #[cfg(feature = "r10-weight-audit")]
        F32_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
        let (offset, len) = self.range(v, "f32", 4)?;
        require(len == expected, "f32 tensor shape mismatch")?;
        let a = F32s::view(self.blob.clone(), offset, len);
        require(a.iter().all(|v| v.is_finite()), "non-finite model weights")?;
        Ok(a)
    }
    pub fn i8s(&self, v: &Value, expected: usize) -> Result<I8s> {
        #[cfg(feature = "r10-weight-audit")]
        I8_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
        let (offset, len) = self.range(v, "i8", 1)?;
        require(len == expected, "i8 tensor shape mismatch")?;
        let a = I8s::view(self.blob.clone(), offset, len);
        require(
            a.iter().all(|&q| q != -128),
            "weights must be symmetric [-127,127]",
        )?;
        Ok(a)
    }
    /// Construction only. Shares one packed copy among instances using this Bundle.
    /// Prefer tools/pack_matrices.py to avoid this compatibility-copy allocation.
    pub(crate) fn packed_i8s(&self, v: &Value, rows: usize, cols: usize) -> Result<I8s> {
        require(
            rows.is_multiple_of(8) && cols.is_multiple_of(2) && rows > 0 && cols > 0,
            "pair-packed matrix dimensions",
        )?;
        let original = self.i8s(v, rows * cols)?;
        let key = (num(v, "offset")?, rows, cols);
        let mut cache = self
            .packed
            .lock()
            .map_err(|_| "weight cache poisoned".to_owned())?;
        if let Some(w) = cache.get(&key) {
            return Ok(w.clone());
        }
        let packed = crate::packed::pack_rows(&original, rows, cols);
        // Every i8 is one byte; copying its representation preserves signed values.
        let bytes =
            unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len()) };
        let w = I8s::view(Arc::new(AlignedBlob::new(bytes)), 0, packed.len());
        cache.insert(key, w.clone());
        Ok(w)
    }
    /// Diagnostics only: this locks the initialization cache, never call in process().
    pub fn compatibility_packed_bytes(&self) -> usize {
        self.packed
            .lock()
            .map(|c| c.values().map(|v| v.len).sum())
            .unwrap_or(0)
    }
    pub fn weight_bytes(&self) -> usize {
        self.blob.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bundle(bytes: &[u8]) -> Arc<Bundle> {
        let m = serde_json::json!({"schema":1,"architecture":"dpdfnet-48hr-v1",
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
    fn rejects_checksum_and_length_mismatch() {
        let b = bundle(&[0; 4]);
        let mut m = b.manifest().unwrap();
        assert!(Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &[1; 4]).is_err());
        m["weight_bytes"] = serde_json::json!(3);
        assert!(Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &[0; 4]).is_err());
    }
}

#[cfg(test)]
mod round10_tests {
    use super::*;
    #[test]
    fn owned_views_survive_move_clone_and_cross_thread_drop() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<F32s>();
        send_sync::<I8s>();
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
    #[cfg(feature = "r10-batch-cache")]
    #[test]
    fn batch_cache_is_exact_shared_and_survives_bundle_drop() {
        let bytes: Vec<u8> = (0..192 * 64)
            .map(|i| ((i % 255) as i16 - 127) as i8 as u8)
            .collect();
        let m = serde_json::json!({"schema":2,"architecture":"dpdfnet-48hr-v1", "weight_bytes":bytes.len(),
          "weights_sha256":format!("{:x}",Sha256::digest(&bytes))});
        let b = Bundle::from_bytes(&serde_json::to_vec(&m).unwrap(), &bytes).unwrap();
        let q = b
            .i8s(
                &serde_json::json!({"dtype":"i8","offset":0,"len":bytes.len()}),
                bytes.len(),
            )
            .unwrap();
        let a = b.batch_i16_cache(&q).unwrap();
        let a2 = b.batch_i16_cache(&q).unwrap();
        assert_eq!(a.as_ptr(), a2.as_ptr());
        assert_eq!(b.derived_batch_bytes(), 24576);
        for (&x, &y) in a.iter().zip(q.iter()) {
            assert_eq!(x, i16::from(y));
        }
        drop(b);
        drop(q);
        drop(a);
        std::thread::spawn(move || assert_eq!(a2[0], -127))
            .join()
            .unwrap();
    }
}
