//! Round-8 opt-in integer schedules. No changed weights, scales or dequant order.
//! Each vector lane is a complete output. INT32 absolute bound is unchanged.
//! Only the AVX2 wide-output schedule changes rows/iteration. Pair projection
//! uses eight accumulators (two matrices x four frequencies), no stack spilling
//! required by the algorithm. SSE pairing does not reduce broadcast count vs
//! round-7's already combined halves; its benefit is NOT assumed.
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    let mut r = 0usize;
    while r + 64 <= rows {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut a4 = _mm256_setzero_si256();
        let mut a5 = _mm256_setzero_si256();
        let mut a6 = _mm256_setzero_si256();
        let mut a7 = _mm256_setzero_si256();
        for j in (0..cols).step_by(2) {
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * cols + j * 8).cast()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, xx));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 8) * cols + j * 8).cast(),
            ));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, xx));
            let w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 16) * cols + j * 8).cast(),
            ));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(w2, xx));
            let w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 24) * cols + j * 8).cast(),
            ));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(w3, xx));
            let w4 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 32) * cols + j * 8).cast(),
            ));
            a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(w4, xx));
            let w5 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 40) * cols + j * 8).cast(),
            ));
            a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(w5, xx));
            let w6 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 48) * cols + j * 8).cast(),
            ));
            a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(w6, xx));
            let w7 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 56) * cols + j * 8).cast(),
            ));
            a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(w7, xx));
        }
        let xs = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r),
            _mm256_mul_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(a0), _mm256_loadu_ps(sw.as_ptr().add(r))),
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
        r += 64;
    }
    if r < rows {
        super::avx2::<1>(
            &mut out[r..],
            &w[r * cols..],
            &sw[r..],
            q,
            sx,
            rows - r,
            cols,
        );
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2_four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    let mut r = 0usize;
    while r + 16 <= rows {
        let mut a0 = _mm256_setzero_si256();
        let mut b0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut b1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut b2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut b3 = _mm256_setzero_si256();
        for j in (0..cols).step_by(2) {
            let wa = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * cols + j * 8).cast()));
            let wb = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                w.as_ptr().add((r + 8) * cols + j * 8).cast(),
            ));
            let x0 = _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(wa, x0));
            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(wb, x0));
            let x1 = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(cols + j).cast::<i32>(),
            ));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(wa, x1));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(wb, x1));
            let x2 = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(2 * cols + j).cast::<i32>(),
            ));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(wa, x2));
            b2 = _mm256_add_epi32(b2, _mm256_madd_epi16(wb, x2));
            let x3 = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(3 * cols + j).cast::<i32>(),
            ));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(wa, x3));
            b3 = _mm256_add_epi32(b3, _mm256_madd_epi16(wb, x3));
        }
        let sa = _mm256_loadu_ps(sw.as_ptr().add(r));
        let sb = _mm256_loadu_ps(sw.as_ptr().add(r + 8));
        let sx0 = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0), sa), sx0),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b0), sb), sx0),
        );
        let sx1 = _mm256_set1_ps(sx[1]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1), sa), sx1),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b1), sb), sx1),
        );
        let sx2 = _mm256_set1_ps(sx[2]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a2), sa), sx2),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(2 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b2), sb), sx2),
        );
        let sx3 = _mm256_set1_ps(sx[3]);
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a3), sa), sx3),
        );
        _mm256_storeu_ps(
            out.as_mut_ptr().add(3 * rows + r + 8),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b3), sb), sx3),
        );
        r += 16;
    }
    if r < rows {
        let mut a = [_mm256_setzero_si256(); 4];
        for j in (0..cols).step_by(2) {
            let ww = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * cols + j * 8).cast()));
            for (k, acc) in a.iter_mut().enumerate() {
                let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(k * cols + j).cast::<i32>(),
                ));
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(ww, xx));
            }
        }
        let ss = _mm256_loadu_ps(sw.as_ptr().add(r));
        for k in 0..4 {
            _mm256_storeu_ps(
                out.as_mut_ptr().add(k * rows + r),
                _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_cvtepi32_ps(a[k]), ss),
                    _mm256_set1_ps(sx[k]),
                ),
            );
        }
    }
}

macro_rules! cached_four {
    ($out:ident,$w:ident,$sw:ident,$q:ident,$sx:ident,$rows:ident) => {{
        let (out, w, sw, q, sx, rows) = ($out, $w, $sw, $q, $sx, $rows);
        // 2 KiB, stack-only, overwritten before use. Scope limited to cols=64.
        let mut inputs = [_mm_setzero_si128(); 128];
        for j in 0..32 {
            for k in 0..4 {
                inputs[j * 4 + k] = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(k * 64 + 2 * j).cast::<i32>(),
                ));
            }
        }
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
                let ww = w.as_ptr().add(r * 64 + j * 16);
                let lo = _mm_cvtepi8_epi16(_mm_loadl_epi64(ww.cast()));
                let hi = _mm_cvtepi8_epi16(_mm_loadl_epi64(ww.add(8).cast()));
                let xx = inputs[j * 4 + 0];
                a0 = _mm_add_epi32(a0, _mm_madd_epi16(lo, xx));
                b0 = _mm_add_epi32(b0, _mm_madd_epi16(hi, xx));
                let xx = inputs[j * 4 + 1];
                a1 = _mm_add_epi32(a1, _mm_madd_epi16(lo, xx));
                b1 = _mm_add_epi32(b1, _mm_madd_epi16(hi, xx));
                let xx = inputs[j * 4 + 2];
                a2 = _mm_add_epi32(a2, _mm_madd_epi16(lo, xx));
                b2 = _mm_add_epi32(b2, _mm_madd_epi16(hi, xx));
                let xx = inputs[j * 4 + 3];
                a3 = _mm_add_epi32(a3, _mm_madd_epi16(lo, xx));
                b3 = _mm_add_epi32(b3, _mm_madd_epi16(hi, xx));
            }
            let sl = _mm_loadu_ps(sw.as_ptr().add(r));
            let sh = _mm_loadu_ps(sw.as_ptr().add(r + 4));
            let xs = _mm_set1_ps(sx[0]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a0), sl), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b0), sh), xs),
            );
            let xs = _mm_set1_ps(sx[1]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a1), sl), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b1), sh), xs),
            );
            let xs = _mm_set1_ps(sx[2]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(2 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a2), sl), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(2 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b2), sh), xs),
            );
            let xs = _mm_set1_ps(sx[3]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(3 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a3), sl), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(3 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b3), sh), xs),
            );
        }
    }};
}

#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn sse_cached_four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    if cols == 64 {
        cached_four!(out, w, sw, q, sx, rows);
    } else {
        super::oldcpu::sse_four(out, w, sw, q, sx, rows, cols);
    }
}

#[target_feature(enable = "avx,sse4.1")]
pub(super) unsafe fn vex_cached_four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    if cols == 64 {
        cached_four!(out, w, sw, q, sx, rows);
    } else {
        super::oldcpu::vex_four(out, w, sw, q, sx, rows, cols);
    }
}

#[cfg(any(feature = "r8-fb-pair", test))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)] // both directions of one step
pub(super) unsafe fn avx2_pair(
    ya: &mut [f32],
    yb: &mut [f32],
    wa: &[i8],
    wb: &[i8],
    sa: &[f32],
    sb: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in (0..rows).step_by(8) {
        let mut a0 = _mm256_setzero_si256();
        let mut b0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut b1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut b2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut b3 = _mm256_setzero_si256();
        let at = (r / 8) * (8 * cols) + (r % 8) * 2;
        for j in (0..cols).step_by(2) {
            let aw = _mm256_cvtepi8_epi16(_mm_loadu_si128(wa.as_ptr().add(at + j * 8).cast()));
            let bw = _mm256_cvtepi8_epi16(_mm_loadu_si128(wb.as_ptr().add(at + j * 8).cast()));
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(aw, xx));
            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(bw, xx));
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(cols + j).cast::<i32>(),
            ));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(aw, xx));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(bw, xx));
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(2 * cols + j).cast::<i32>(),
            ));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(aw, xx));
            b2 = _mm256_add_epi32(b2, _mm256_madd_epi16(bw, xx));
            let xx = _mm256_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(3 * cols + j).cast::<i32>(),
            ));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(aw, xx));
            b3 = _mm256_add_epi32(b3, _mm256_madd_epi16(bw, xx));
        }
        let ascale = _mm256_loadu_ps(sa.as_ptr().add(r));
        let bscale = _mm256_loadu_ps(sb.as_ptr().add(r));
        let xs = _mm256_set1_ps(sx[0]);
        _mm256_storeu_ps(
            ya.as_mut_ptr().add(r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0), ascale), xs),
        );
        _mm256_storeu_ps(
            yb.as_mut_ptr().add(r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b0), bscale), xs),
        );
        let xs = _mm256_set1_ps(sx[1]);
        _mm256_storeu_ps(
            ya.as_mut_ptr().add(rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1), ascale), xs),
        );
        _mm256_storeu_ps(
            yb.as_mut_ptr().add(rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b1), bscale), xs),
        );
        let xs = _mm256_set1_ps(sx[2]);
        _mm256_storeu_ps(
            ya.as_mut_ptr().add(2 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a2), ascale), xs),
        );
        _mm256_storeu_ps(
            yb.as_mut_ptr().add(2 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b2), bscale), xs),
        );
        let xs = _mm256_set1_ps(sx[3]);
        _mm256_storeu_ps(
            ya.as_mut_ptr().add(3 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a3), ascale), xs),
        );
        _mm256_storeu_ps(
            yb.as_mut_ptr().add(3 * rows + r),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(b3), bscale), xs),
        );
    }
}

#[cfg(any(feature = "r8-fb-pair", test))]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)] // both directions of one step
pub(super) unsafe fn sse_pair(
    ya: &mut [f32],
    yb: &mut [f32],
    wa: &[i8],
    wb: &[i8],
    sa: &[f32],
    sb: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in (0..rows).step_by(4) {
        let mut a0 = _mm_setzero_si128();
        let mut b0 = _mm_setzero_si128();
        let mut a1 = _mm_setzero_si128();
        let mut b1 = _mm_setzero_si128();
        let mut a2 = _mm_setzero_si128();
        let mut b2 = _mm_setzero_si128();
        let mut a3 = _mm_setzero_si128();
        let mut b3 = _mm_setzero_si128();
        let at = (r / 8) * (8 * cols) + (r % 8) * 2;
        for j in (0..cols).step_by(2) {
            let aw = _mm_cvtepi8_epi16(_mm_loadl_epi64(wa.as_ptr().add(at + j * 8).cast()));
            let bw = _mm_cvtepi8_epi16(_mm_loadl_epi64(wb.as_ptr().add(at + j * 8).cast()));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            a0 = _mm_add_epi32(a0, _mm_madd_epi16(aw, xx));
            b0 = _mm_add_epi32(b0, _mm_madd_epi16(bw, xx));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(cols + j).cast::<i32>(),
            ));
            a1 = _mm_add_epi32(a1, _mm_madd_epi16(aw, xx));
            b1 = _mm_add_epi32(b1, _mm_madd_epi16(bw, xx));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(2 * cols + j).cast::<i32>(),
            ));
            a2 = _mm_add_epi32(a2, _mm_madd_epi16(aw, xx));
            b2 = _mm_add_epi32(b2, _mm_madd_epi16(bw, xx));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(3 * cols + j).cast::<i32>(),
            ));
            a3 = _mm_add_epi32(a3, _mm_madd_epi16(aw, xx));
            b3 = _mm_add_epi32(b3, _mm_madd_epi16(bw, xx));
        }
        let ascale = _mm_loadu_ps(sa.as_ptr().add(r));
        let bscale = _mm_loadu_ps(sb.as_ptr().add(r));
        let xs = _mm_set1_ps(sx[0]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a0), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b0), bscale), xs),
        );
        let xs = _mm_set1_ps(sx[1]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a1), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b1), bscale), xs),
        );
        let xs = _mm_set1_ps(sx[2]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(2 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a2), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(2 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b2), bscale), xs),
        );
        let xs = _mm_set1_ps(sx[3]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(3 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a3), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(3 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b3), bscale), xs),
        );
    }
}

#[cfg(any(feature = "r8-fb-pair", test))]
#[target_feature(enable = "avx,sse4.1")]
#[allow(clippy::too_many_arguments)] // both directions of one step
pub(super) unsafe fn vex_pair(
    ya: &mut [f32],
    yb: &mut [f32],
    wa: &[i8],
    wb: &[i8],
    sa: &[f32],
    sb: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in (0..rows).step_by(4) {
        let mut a0 = _mm_setzero_si128();
        let mut b0 = _mm_setzero_si128();
        let mut a1 = _mm_setzero_si128();
        let mut b1 = _mm_setzero_si128();
        let mut a2 = _mm_setzero_si128();
        let mut b2 = _mm_setzero_si128();
        let mut a3 = _mm_setzero_si128();
        let mut b3 = _mm_setzero_si128();
        let at = (r / 8) * (8 * cols) + (r % 8) * 2;
        for j in (0..cols).step_by(2) {
            let aw = _mm_cvtepi8_epi16(_mm_loadl_epi64(wa.as_ptr().add(at + j * 8).cast()));
            let bw = _mm_cvtepi8_epi16(_mm_loadl_epi64(wb.as_ptr().add(at + j * 8).cast()));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            a0 = _mm_add_epi32(a0, _mm_madd_epi16(aw, xx));
            b0 = _mm_add_epi32(b0, _mm_madd_epi16(bw, xx));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(cols + j).cast::<i32>(),
            ));
            a1 = _mm_add_epi32(a1, _mm_madd_epi16(aw, xx));
            b1 = _mm_add_epi32(b1, _mm_madd_epi16(bw, xx));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(2 * cols + j).cast::<i32>(),
            ));
            a2 = _mm_add_epi32(a2, _mm_madd_epi16(aw, xx));
            b2 = _mm_add_epi32(b2, _mm_madd_epi16(bw, xx));
            let xx = _mm_set1_epi32(std::ptr::read_unaligned(
                q.as_ptr().add(3 * cols + j).cast::<i32>(),
            ));
            a3 = _mm_add_epi32(a3, _mm_madd_epi16(aw, xx));
            b3 = _mm_add_epi32(b3, _mm_madd_epi16(bw, xx));
        }
        let ascale = _mm_loadu_ps(sa.as_ptr().add(r));
        let bscale = _mm_loadu_ps(sb.as_ptr().add(r));
        let xs = _mm_set1_ps(sx[0]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a0), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b0), bscale), xs),
        );
        let xs = _mm_set1_ps(sx[1]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a1), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b1), bscale), xs),
        );
        let xs = _mm_set1_ps(sx[2]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(2 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a2), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(2 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b2), bscale), xs),
        );
        let xs = _mm_set1_ps(sx[3]);
        _mm_storeu_ps(
            ya.as_mut_ptr().add(3 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a3), ascale), xs),
        );
        _mm_storeu_ps(
            yb.as_mut_ptr().add(3 * rows + r),
            _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b3), bscale), xs),
        );
    }
}
