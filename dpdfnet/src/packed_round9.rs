//! R9 AVX2 schedules. Same pair-output8 weights, bounded i32 arithmetic and
//! (sum as f32 * sw) * sx epilogue. No FMA, VNNI or altered quantization.
//! Two input pairs per loop; the four-vector kernel keeps the adopted eight
//! accumulators. Const widths eliminate dynamic input strides in the hot shapes.
use std::arch::x86_64::*;

#[cfg(any(feature = "r9-packed-k4", test))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    if cols == 64 && rows % 64 == 0 {
        run_one::<64>(out, w, sw, q, sx, rows);
    } else if cols == 256 && rows % 64 == 0 {
        run_one::<256>(out, w, sw, q, sx, rows);
    } else {
        super::round8::avx2_one(out, w, sw, q, sx, rows, cols);
    }
}
#[cfg(any(feature = "r9-packed-k4", test))]
#[target_feature(enable = "avx2")]
unsafe fn run_one<const N: usize>(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
) {
    for r in (0..rows).step_by(64) {
        let mut a0_0 = _mm256_setzero_si256();
        let mut a0_1 = _mm256_setzero_si256();
        let mut a0_2 = _mm256_setzero_si256();
        let mut a0_3 = _mm256_setzero_si256();
        let mut a0_4 = _mm256_setzero_si256();
        let mut a0_5 = _mm256_setzero_si256();
        let mut a0_6 = _mm256_setzero_si256();
        let mut a0_7 = _mm256_setzero_si256();
        for j in (0..N).step_by(4) {
            {
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(j + 0).cast::<i32>(),
                ));
                let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 0) * N + (j + 0) * 8).cast(),
                ));
                a0_0 = _mm256_add_epi32(a0_0, _mm256_madd_epi16(w0, x));
                let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 8) * N + (j + 0) * 8).cast(),
                ));
                a0_1 = _mm256_add_epi32(a0_1, _mm256_madd_epi16(w1, x));
                let w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 16) * N + (j + 0) * 8).cast(),
                ));
                a0_2 = _mm256_add_epi32(a0_2, _mm256_madd_epi16(w2, x));
                let w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 24) * N + (j + 0) * 8).cast(),
                ));
                a0_3 = _mm256_add_epi32(a0_3, _mm256_madd_epi16(w3, x));
                let w4 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 32) * N + (j + 0) * 8).cast(),
                ));
                a0_4 = _mm256_add_epi32(a0_4, _mm256_madd_epi16(w4, x));
                let w5 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 40) * N + (j + 0) * 8).cast(),
                ));
                a0_5 = _mm256_add_epi32(a0_5, _mm256_madd_epi16(w5, x));
                let w6 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 48) * N + (j + 0) * 8).cast(),
                ));
                a0_6 = _mm256_add_epi32(a0_6, _mm256_madd_epi16(w6, x));
                let w7 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 56) * N + (j + 0) * 8).cast(),
                ));
                a0_7 = _mm256_add_epi32(a0_7, _mm256_madd_epi16(w7, x));
            }
            {
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(j + 2).cast::<i32>(),
                ));
                let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 0) * N + (j + 2) * 8).cast(),
                ));
                a0_0 = _mm256_add_epi32(a0_0, _mm256_madd_epi16(w0, x));
                let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 8) * N + (j + 2) * 8).cast(),
                ));
                a0_1 = _mm256_add_epi32(a0_1, _mm256_madd_epi16(w1, x));
                let w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 16) * N + (j + 2) * 8).cast(),
                ));
                a0_2 = _mm256_add_epi32(a0_2, _mm256_madd_epi16(w2, x));
                let w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 24) * N + (j + 2) * 8).cast(),
                ));
                a0_3 = _mm256_add_epi32(a0_3, _mm256_madd_epi16(w3, x));
                let w4 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 32) * N + (j + 2) * 8).cast(),
                ));
                a0_4 = _mm256_add_epi32(a0_4, _mm256_madd_epi16(w4, x));
                let w5 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 40) * N + (j + 2) * 8).cast(),
                ));
                a0_5 = _mm256_add_epi32(a0_5, _mm256_madd_epi16(w5, x));
                let w6 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 48) * N + (j + 2) * 8).cast(),
                ));
                a0_6 = _mm256_add_epi32(a0_6, _mm256_madd_epi16(w6, x));
                let w7 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 56) * N + (j + 2) * 8).cast(),
                ));
                a0_7 = _mm256_add_epi32(a0_7, _mm256_madd_epi16(w7, x));
            }
        }
        let xscale = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 0),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_0),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 0)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 8),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_1),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 8)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 16),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_2),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 16)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 24),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_3),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 24)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 32),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_4),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 32)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 40),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_5),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 40)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 48),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_6),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 48)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 56),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_7),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 56)),
                ),
                xscale,
            ),
        );
    }
}

#[cfg(any(feature = "r9-packed-k4", test))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    if cols == 64 && rows % 16 == 0 {
        run_four::<64>(out, w, sw, q, sx, rows);
    } else if cols == 256 && rows % 16 == 0 {
        run_four::<256>(out, w, sw, q, sx, rows);
    } else {
        super::round8::avx2_four(out, w, sw, q, sx, rows, cols);
    }
}
#[cfg(any(feature = "r9-packed-k4", test))]
#[target_feature(enable = "avx2")]
unsafe fn run_four<const N: usize>(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
) {
    for r in (0..rows).step_by(16) {
        let mut a0_0 = _mm256_setzero_si256();
        let mut a0_1 = _mm256_setzero_si256();
        let mut a1_0 = _mm256_setzero_si256();
        let mut a1_1 = _mm256_setzero_si256();
        let mut a2_0 = _mm256_setzero_si256();
        let mut a2_1 = _mm256_setzero_si256();
        let mut a3_0 = _mm256_setzero_si256();
        let mut a3_1 = _mm256_setzero_si256();
        for j in (0..N).step_by(4) {
            {
                let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 0) * N + (j + 0) * 8).cast(),
                ));
                let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 8) * N + (j + 0) * 8).cast(),
                ));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(j + 0).cast::<i32>(),
                ));
                a0_0 = _mm256_add_epi32(a0_0, _mm256_madd_epi16(w0, x));
                a0_1 = _mm256_add_epi32(a0_1, _mm256_madd_epi16(w1, x));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(1 * N + j + 0).cast::<i32>(),
                ));
                a1_0 = _mm256_add_epi32(a1_0, _mm256_madd_epi16(w0, x));
                a1_1 = _mm256_add_epi32(a1_1, _mm256_madd_epi16(w1, x));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(2 * N + j + 0).cast::<i32>(),
                ));
                a2_0 = _mm256_add_epi32(a2_0, _mm256_madd_epi16(w0, x));
                a2_1 = _mm256_add_epi32(a2_1, _mm256_madd_epi16(w1, x));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(3 * N + j + 0).cast::<i32>(),
                ));
                a3_0 = _mm256_add_epi32(a3_0, _mm256_madd_epi16(w0, x));
                a3_1 = _mm256_add_epi32(a3_1, _mm256_madd_epi16(w1, x));
            }
            {
                let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 0) * N + (j + 2) * 8).cast(),
                ));
                let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    w.as_ptr().add((r + 8) * N + (j + 2) * 8).cast(),
                ));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(j + 2).cast::<i32>(),
                ));
                a0_0 = _mm256_add_epi32(a0_0, _mm256_madd_epi16(w0, x));
                a0_1 = _mm256_add_epi32(a0_1, _mm256_madd_epi16(w1, x));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(1 * N + j + 2).cast::<i32>(),
                ));
                a1_0 = _mm256_add_epi32(a1_0, _mm256_madd_epi16(w0, x));
                a1_1 = _mm256_add_epi32(a1_1, _mm256_madd_epi16(w1, x));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(2 * N + j + 2).cast::<i32>(),
                ));
                a2_0 = _mm256_add_epi32(a2_0, _mm256_madd_epi16(w0, x));
                a2_1 = _mm256_add_epi32(a2_1, _mm256_madd_epi16(w1, x));
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(3 * N + j + 2).cast::<i32>(),
                ));
                a3_0 = _mm256_add_epi32(a3_0, _mm256_madd_epi16(w0, x));
                a3_1 = _mm256_add_epi32(a3_1, _mm256_madd_epi16(w1, x));
            }
        }
        let xscale = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 0),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_0),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 0)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 8),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0_1),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 8)),
                ),
                xscale,
            ),
        );
        let xscale = _mm256_set1_ps(sx[1]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(1 * rows + r + 0),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a1_0),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 0)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(1 * rows + r + 8),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a1_1),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 8)),
                ),
                xscale,
            ),
        );
        let xscale = _mm256_set1_ps(sx[2]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r + 0),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a2_0),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 0)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r + 8),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a2_1),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 8)),
                ),
                xscale,
            ),
        );
        let xscale = _mm256_set1_ps(sx[3]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r + 0),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a3_0),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 0)),
                ),
                xscale,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r + 8),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a3_1),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 8)),
                ),
                xscale,
            ),
        );
    }
}

/// Called only for validated 192x64 intra-frequency recurrent matrices.
/// The cache contains EXACT i16 extensions of original i8 values, not dequantized weights.
#[cfg(feature = "r9-recurrent-cache")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn cached_one(out: &mut [f32], w: &[i16], sw: &[f32], q: &[i16], sx: f32) {
    for r in (0..192).step_by(64) {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut a4 = _mm256_setzero_si256();
        let mut a5 = _mm256_setzero_si256();
        let mut a6 = _mm256_setzero_si256();
        let mut a7 = _mm256_setzero_si256();
        for j in (0..64).step_by(2) {
            let x = _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 0) * 64 + j * 8).cast());
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 8) * 64 + j * 8).cast());
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 16) * 64 + j * 8).cast());
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 24) * 64 + j * 8).cast());
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 32) * 64 + j * 8).cast());
            a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 40) * 64 + j * 8).cast());
            a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 48) * 64 + j * 8).cast());
            a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(ww, x));
            let ww = _mm256_loadu_si256(w.as_ptr().add((r + 56) * 64 + j * 8).cast());
            a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(ww, x));
        }
        let xs = _mm256_set1_ps(sx);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 0),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a0),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 0)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 8),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a1),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 8)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 16),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a2),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 16)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 24),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a3),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 24)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 32),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a4),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 32)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 40),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a5),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 40)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 48),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a6),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 48)),
                ),
                xs,
            ),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 56),
            _mm256_mul_ps(
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(a7),
                    _mm256_loadu_ps(sw.as_ptr().add(r + 56)),
                ),
                xs,
            ),
        );
    }
}
