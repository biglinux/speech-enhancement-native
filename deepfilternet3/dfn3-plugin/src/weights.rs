//! The embedded weight blob: f32 tensors, the int8 GRU matrices and their per-row
//! f32 scales, viewed in place.

use ops::pack_format::{Geometry, sections};

/// A weight blob aligned so its f32 sections can be viewed without a copy.
#[repr(C, align(64))]
pub struct AlignedBlob<const N: usize>(pub [u8; N]);

pub struct Tensors {
    floats: &'static [f32],
    integers: &'static [i8],
    scales: &'static [f32],
}

impl Tensors {
    /// Views a blob that `build.rs` packed, and locks it into RAM (best effort):
    /// a page fault in the audio callback is an xrun.
    pub fn embedded<const N: usize>(blob: &'static AlignedBlob<N>, g: Geometry) -> Self {
        const {
            assert!(
                cfg!(target_endian = "little"),
                "embedded views require little endian"
            )
        };
        let bytes = &blob.0;
        let section = sections(bytes, g).expect("invalid embedded weights");
        assert!(section.packed, "build.rs embeds packed weights");
        // SAFETY: `sections` checked the canonical ranges, which start at multiples
        // of 4 in a 64-byte aligned static blob.
        let t = unsafe {
            Self {
                floats: std::slice::from_raw_parts(
                    bytes.as_ptr().add(section.f.start).cast::<f32>(),
                    g.floats,
                ),
                integers: std::slice::from_raw_parts(
                    bytes.as_ptr().add(section.i.start).cast::<i8>(),
                    g.integers,
                ),
                scales: std::slice::from_raw_parts(
                    bytes.as_ptr().add(section.s.start).cast::<f32>(),
                    g.scales,
                ),
            }
        };
        ops::mlock_slice(t.floats);
        ops::mlock_slice(t.integers);
        ops::mlock_slice(t.scales);
        t
    }

    #[inline]
    pub fn f(&self, o: usize, n: usize) -> &[f32] {
        &self.floats[o..o + n]
    }
    #[inline]
    pub fn i(&self, o: usize, n: usize) -> &[i8] {
        &self.integers[o..o + n]
    }
    #[inline]
    pub fn sc(&self, o: usize, n: usize) -> &[f32] {
        &self.scales[o..o + n]
    }
}
