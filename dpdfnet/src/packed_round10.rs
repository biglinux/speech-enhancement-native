//! R10 exact four-vector schedules. Existing pair-output8-v1 layout is unchanged.
//! No FMA, FP32 dot products, VNNI, reassociation of dequantization, or RT allocation.
//! Expanded cache replaces signed extensions by loads; +24 KiB per 192x64 matrix.
//! Optional AVX2 broadcast cache is 4 KiB of stack, not persistent model data.
use std::arch::x86_64::*;

// Callers select ISA once. All wrappers below retain the established bounds.
#[cfg(feature = "r10-avx2-prebroadcast")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compact_four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    if cols != 64 || rows % 16 != 0 {
        return super::round8::avx2_four(out, w, sw, q, sx, rows, cols);
    }
    compact_four_64(out, w, sw, q, sx, rows);
}

#[cfg(feature = "r10-avx2-prebroadcast")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compact_four_64(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
) {
    #[cfg(feature = "r10-avx2-prebroadcast")]
    let inputs = {
        let mut cache = [_mm256_setzero_si256(); 128];
        for j in 0..32 {
            for k in 0..4 {
                cache[j * 4 + k] = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(k * 64 + 2 * j).cast::<i32>(),
                ));
            }
        }
        cache
    };
    for r in (0..rows).step_by(16) {
        let mut a0 = _mm256_setzero_si256();
        let mut b0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut b1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut b2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut b3 = _mm256_setzero_si256();
        for j in 0..32 {
            let wa = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * 64 + j * 16).cast()));
            let wb = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 8) * 64 + j * 16).cast(),
            ));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 0];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(0 * 64 + 2 * j).cast::<i32>(),
            ));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(wa, xx));
            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(wb, xx));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 1];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(1 * 64 + 2 * j).cast::<i32>(),
            ));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(wa, xx));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(wb, xx));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 2];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(2 * 64 + 2 * j).cast::<i32>(),
            ));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(wa, xx));
            b2 = _mm256_add_epi32(b2, _mm256_madd_epi16(wb, xx));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 3];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(3 * 64 + 2 * j).cast::<i32>(),
            ));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(wa, xx));
            b3 = _mm256_add_epi32(b3, _mm256_madd_epi16(wb, xx));
        }
        let sa = _mm256_loadu_ps(sw.as_ptr().add(r));
        let sb = _mm256_loadu_ps(sw.as_ptr().add(r + 8));
        let xs = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(0 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(0 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b0), sb), xs),
        );
        let xs = _mm256_set1_ps(sx[1]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(1 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(1 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b1), sb), xs),
        );
        let xs = _mm256_set1_ps(sx[2]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a2), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b2), sb), xs),
        );
        let xs = _mm256_set1_ps(sx[3]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a3), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b3), sb), xs),
        );
    }
}

#[cfg(feature = "r10-batch-cache")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn cached_four_avx2(
    out: &mut [f32],
    w: &[i16],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
) {
    #[cfg(feature = "r10-avx2-prebroadcast")]
    let inputs = {
        let mut cache = [_mm256_setzero_si256(); 128];
        for j in 0..32 {
            for k in 0..4 {
                cache[j * 4 + k] = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(k * 64 + 2 * j).cast::<i32>(),
                ));
            }
        }
        cache
    };
    for r in (0..rows).step_by(16) {
        let mut a0 = _mm256_setzero_si256();
        let mut b0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut b1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut b2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut b3 = _mm256_setzero_si256();
        for j in 0..32 {
            let wa = _mm256_loadu_si256(w.as_ptr().add(r * 64 + j * 16).cast());
            let wb = _mm256_loadu_si256(w.as_ptr().add((r + 8) * 64 + j * 16).cast());
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 0];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(0 * 64 + 2 * j).cast::<i32>(),
            ));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(wa, xx));
            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(wb, xx));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 1];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(1 * 64 + 2 * j).cast::<i32>(),
            ));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(wa, xx));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(wb, xx));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 2];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(2 * 64 + 2 * j).cast::<i32>(),
            ));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(wa, xx));
            b2 = _mm256_add_epi32(b2, _mm256_madd_epi16(wb, xx));
            #[cfg(feature = "r10-avx2-prebroadcast")]
            let xx = inputs[j * 4 + 3];
            #[cfg(not(feature = "r10-avx2-prebroadcast"))]
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(3 * 64 + 2 * j).cast::<i32>(),
            ));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(wa, xx));
            b3 = _mm256_add_epi32(b3, _mm256_madd_epi16(wb, xx));
        }
        let sa = _mm256_loadu_ps(sw.as_ptr().add(r));
        let sb = _mm256_loadu_ps(sw.as_ptr().add(r + 8));
        let xs = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(0 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(0 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b0), sb), xs),
        );
        let xs = _mm256_set1_ps(sx[1]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(1 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(1 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b1), sb), xs),
        );
        let xs = _mm256_set1_ps(sx[2]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a2), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b2), sb), xs),
        );
        let xs = _mm256_set1_ps(sx[3]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a3), sa), xs),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b3), sb), xs),
        );
    }
}

#[cfg(feature = "r10-batch-cache")]
macro_rules! cached_sse_body {
    ($out:ident, $w:ident, $sw:ident, $q:ident, $sx:ident, $rows:ident) => {{
        let (out, w, sw, q, sx, rows) = ($out, $w, $sw, $q, $sx, $rows);
        #[cfg(feature = "r8-sse-prebroadcast")]
        let inputs = {
            let mut cache = [_mm_setzero_si128(); 128];
            for j in 0..32 {
                for k in 0..4 {
                    cache[j * 4 + k] = _mm_set1_epi32(std::ptr::read_unaligned(
                        q.as_ptr().add(k * 64 + 2 * j).cast::<i32>(),
                    ));
                }
            }
            cache
        };
        for r in (0..rows).step_by(8) {
            let mut a0 = _mm_setzero_si128();
            let mut b0 = _mm_setzero_si128();
            let mut a1 = _mm_setzero_si128();
            let mut b1 = _mm_setzero_si128();
            let mut a2 = _mm_setzero_si128();
            let mut b2 = _mm_setzero_si128();
            let mut a3 = _mm_setzero_si128();
            let mut b3 = _mm_setzero_si128();
            for j in 0..32 {
                let lo = _mm_loadu_si128(w.as_ptr().add(r * 64 + j * 16).cast());
                let hi = _mm_loadu_si128(w.as_ptr().add(r * 64 + j * 16 + 8).cast());
                #[cfg(feature = "r8-sse-prebroadcast")]
                let xx = inputs[j * 4 + 0];
                #[cfg(not(feature = "r8-sse-prebroadcast"))]
                let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(0 * 64 + 2 * j).cast::<i32>(),
                ));
                a0 = _mm_add_epi32(a0, _mm_madd_epi16(lo, xx));
                b0 = _mm_add_epi32(b0, _mm_madd_epi16(hi, xx));
                #[cfg(feature = "r8-sse-prebroadcast")]
                let xx = inputs[j * 4 + 1];
                #[cfg(not(feature = "r8-sse-prebroadcast"))]
                let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(1 * 64 + 2 * j).cast::<i32>(),
                ));
                a1 = _mm_add_epi32(a1, _mm_madd_epi16(lo, xx));
                b1 = _mm_add_epi32(b1, _mm_madd_epi16(hi, xx));
                #[cfg(feature = "r8-sse-prebroadcast")]
                let xx = inputs[j * 4 + 2];
                #[cfg(not(feature = "r8-sse-prebroadcast"))]
                let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(2 * 64 + 2 * j).cast::<i32>(),
                ));
                a2 = _mm_add_epi32(a2, _mm_madd_epi16(lo, xx));
                b2 = _mm_add_epi32(b2, _mm_madd_epi16(hi, xx));
                #[cfg(feature = "r8-sse-prebroadcast")]
                let xx = inputs[j * 4 + 3];
                #[cfg(not(feature = "r8-sse-prebroadcast"))]
                let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(3 * 64 + 2 * j).cast::<i32>(),
                ));
                a3 = _mm_add_epi32(a3, _mm_madd_epi16(lo, xx));
                b3 = _mm_add_epi32(b3, _mm_madd_epi16(hi, xx));
            }
            let lo = _mm_loadu_ps(sw.as_ptr().add(r));
            let hi = _mm_loadu_ps(sw.as_ptr().add(r + 4));
            let xs = _mm_set1_ps(sx[0]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(0 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a0), lo), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(0 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b0), hi), xs),
            );
            let xs = _mm_set1_ps(sx[1]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(1 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a1), lo), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(1 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b1), hi), xs),
            );
            let xs = _mm_set1_ps(sx[2]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(2 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a2), lo), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(2 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b2), hi), xs),
            );
            let xs = _mm_set1_ps(sx[3]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(3 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a3), lo), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(3 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b3), hi), xs),
            );
        }
    }};
}
#[cfg(feature = "r10-batch-cache")]
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn cached_four_sse(
    out: &mut [f32],
    w: &[i16],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
) {
    cached_sse_body!(out, w, sw, q, sx, rows);
}
#[cfg(feature = "r10-batch-cache")]
#[target_feature(enable = "avx,sse4.1")]
pub(super) unsafe fn cached_four_vex(
    out: &mut [f32],
    w: &[i16],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
) {
    cached_sse_body!(out, w, sw, q, sx, rows);
}
