//! 128-bit integer kernels for CPUs without AVX2: SSE4.1, and the same work in
//! AVX's VEX encoding. They read the pair-output8-v1 weights as stored.
//! Both schedules fit x86-64's 16 vector registers; inspect actual machine code
//! for spills (objdump of the release build). SSE and AVX1 bodies use only XMM
//! integer arithmetic. AVX1 changes the encoding (VEX), not the integer width.
use std::arch::x86_64::*;

// Each lane is a complete output. Independent output accumulators hide the
// integer-add latency without two stripes and without rereading the input for
// each four-output half. No change to the proven INT32 overflow bound.
macro_rules! one_body {
    ($out:ident,$w:ident,$sw:ident,$q:ident,$sx:ident,$rows:ident,$cols:ident) => {{
        let (out, w, sw, q, sx, rows, cols) = ($out, $w, $sw, $q, $sx, $rows, $cols);
        let mut r = 0usize;
        while r + 32 <= rows {
            let mut a0 = _mm_setzero_si128();
            let mut a1 = _mm_setzero_si128();
            let mut a2 = _mm_setzero_si128();
            let mut a3 = _mm_setzero_si128();
            let mut a4 = _mm_setzero_si128();
            let mut a5 = _mm_setzero_si128();
            let mut a6 = _mm_setzero_si128();
            let mut a7 = _mm_setzero_si128();
            let mut j = 0usize;
            while j < cols {
                // ONE broadcast for 32 outputs; the baseline does eight.
                let x = _mm_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
                a0 = _mm_add_epi32(
                    a0,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 0) * cols + j * 8 + 0).cast(),
                        )),
                        x,
                    ),
                );
                a1 = _mm_add_epi32(
                    a1,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 0) * cols + j * 8 + 8).cast(),
                        )),
                        x,
                    ),
                );
                a2 = _mm_add_epi32(
                    a2,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 8) * cols + j * 8 + 0).cast(),
                        )),
                        x,
                    ),
                );
                a3 = _mm_add_epi32(
                    a3,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 8) * cols + j * 8 + 8).cast(),
                        )),
                        x,
                    ),
                );
                a4 = _mm_add_epi32(
                    a4,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 16) * cols + j * 8 + 0).cast(),
                        )),
                        x,
                    ),
                );
                a5 = _mm_add_epi32(
                    a5,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 16) * cols + j * 8 + 8).cast(),
                        )),
                        x,
                    ),
                );
                a6 = _mm_add_epi32(
                    a6,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 24) * cols + j * 8 + 0).cast(),
                        )),
                        x,
                    ),
                );
                a7 = _mm_add_epi32(
                    a7,
                    _mm_madd_epi16(
                        _mm_cvtepi8_epi16(_mm_loadl_epi64(
                            w.as_ptr().add((r + 24) * cols + j * 8 + 8).cast(),
                        )),
                        x,
                    ),
                );
                j += 2;
            }
            let xs = _mm_set1_ps(sx[0]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 0),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a0), _mm_loadu_ps(sw.as_ptr().add(r + 0))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 4),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a1), _mm_loadu_ps(sw.as_ptr().add(r + 4))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 8),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a2), _mm_loadu_ps(sw.as_ptr().add(r + 8))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 12),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a3), _mm_loadu_ps(sw.as_ptr().add(r + 12))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 16),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a4), _mm_loadu_ps(sw.as_ptr().add(r + 16))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 20),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a5), _mm_loadu_ps(sw.as_ptr().add(r + 20))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 24),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a6), _mm_loadu_ps(sw.as_ptr().add(r + 24))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 28),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a7), _mm_loadu_ps(sw.as_ptr().add(r + 28))),
                    xs,
                ),
            );
            r += 32;
        }
        // General layout tail (the shipped 192/768-row matrices take no tail).
        while r < rows {
            let mut a = _mm_setzero_si128();
            let mut b = _mm_setzero_si128();
            let mut j = 0usize;
            while j < cols {
                let x = _mm_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
                let wp = w.as_ptr().add(r * cols + j * 8);
                a = _mm_add_epi32(
                    a,
                    _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_loadl_epi64(wp.cast())), x),
                );
                b = _mm_add_epi32(
                    b,
                    _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_loadl_epi64(wp.add(8).cast())), x),
                );
                j += 2;
            }
            let xs = _mm_set1_ps(sx[0]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(r),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(a), _mm_loadu_ps(sw.as_ptr().add(r))),
                    xs,
                ),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(r + 4),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(b), _mm_loadu_ps(sw.as_ptr().add(r + 4))),
                    xs,
                ),
            );
            r += 8;
        }
    }};
}
macro_rules! four_body {
    ($out:ident,$w:ident,$sw:ident,$q:ident,$sx:ident,$rows:ident,$cols:ident) => {{
        let (out, w, sw, q, sx, rows, cols) = ($out, $w, $sw, $q, $sx, $rows, $cols);
        for r in (0..rows).step_by(8) {
            let mut a0 = _mm_setzero_si128();
            let mut b0 = _mm_setzero_si128();
            let mut a1 = _mm_setzero_si128();
            let mut b1 = _mm_setzero_si128();
            let mut a2 = _mm_setzero_si128();
            let mut b2 = _mm_setzero_si128();
            let mut a3 = _mm_setzero_si128();
            let mut b3 = _mm_setzero_si128();
            let mut j = 0usize;
            while j < cols {
                let wp = w.as_ptr().add(r * cols + j * 8);
                let wl = _mm_cvtepi8_epi16(_mm_loadl_epi64(wp.cast()));
                let wh = _mm_cvtepi8_epi16(_mm_loadl_epi64(wp.add(8).cast()));
                let x0 = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(0 * cols + j).cast::<i32>(),
                ));
                a0 = _mm_add_epi32(a0, _mm_madd_epi16(wl, x0));
                b0 = _mm_add_epi32(b0, _mm_madd_epi16(wh, x0));
                let x1 = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(1 * cols + j).cast::<i32>(),
                ));
                a1 = _mm_add_epi32(a1, _mm_madd_epi16(wl, x1));
                b1 = _mm_add_epi32(b1, _mm_madd_epi16(wh, x1));
                let x2 = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(2 * cols + j).cast::<i32>(),
                ));
                a2 = _mm_add_epi32(a2, _mm_madd_epi16(wl, x2));
                b2 = _mm_add_epi32(b2, _mm_madd_epi16(wh, x2));
                let x3 = _mm_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(3 * cols + j).cast::<i32>(),
                ));
                a3 = _mm_add_epi32(a3, _mm_madd_epi16(wl, x3));
                b3 = _mm_add_epi32(b3, _mm_madd_epi16(wh, x3));
                j += 2;
            }
            let sl = _mm_loadu_ps(sw.as_ptr().add(r));
            let sh = _mm_loadu_ps(sw.as_ptr().add(r + 4));
            let xs = _mm_set1_ps(sx[0]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(0 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a0), sl), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(0 * rows + r + 4),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(b0), sh), xs),
            );
            let xs = _mm_set1_ps(sx[1]);
            _mm_storeu_ps(
                out.as_mut_ptr().add(1 * rows + r),
                _mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(a1), sl), xs),
            );
            _mm_storeu_ps(
                out.as_mut_ptr().add(1 * rows + r + 4),
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
pub(super) unsafe fn sse_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    one_body!(out, w, sw, q, sx, rows, cols)
}

#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn sse_four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    four_body!(out, w, sw, q, sx, rows, cols)
}

#[target_feature(enable = "avx,sse4.1")]
pub(super) unsafe fn vex_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    one_body!(out, w, sw, q, sx, rows, cols)
}

#[target_feature(enable = "avx,sse4.1")]
pub(super) unsafe fn vex_four(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    four_body!(out, w, sw, q, sx, rows, cols)
}
