//! Exact W8A16 packed GEMV for DFN3 (768x256) and DFN3-LL (1536x512).
//! Format [rows/8, cols/2, lane, pair member]; scales and gate order unchanged.
//! AVX2 uses 64 output accumulators per tile; SSE uses 32. No int16 cache.
use crate::simd_tier;
pub(crate) fn matvec(
    y: &mut [f32],
    w: &[i8],
    s: &[f32],
    q: &[i16],
    sx: f32,
    rows: usize,
    cols: usize,
) {
    assert!(
        rows > 0 && rows.is_multiple_of(8) && cols > 0 && cols.is_multiple_of(2) && cols <= 1024
    );
    assert!(y.len() >= rows && s.len() >= rows && q.len() >= cols && w.len() >= rows * cols);
    // Keep the public-safe arbitrary-i16 contract. The actual quantizer emits ±16383.
    let safe = cols <= 511
        || q[..cols]
            .iter()
            .fold(0u64, |a, &x| a.saturating_add(i64::from(x).unsigned_abs()))
            <= i32::MAX as u64 / 128;
    #[cfg(target_arch = "x86_64")]
    if safe {
        if !cfg!(feature = "force-sse41")
            && !cfg!(feature = "force-avx1")
            && std::is_x86_feature_detected!("avx2")
        {
            unsafe { avx2_one(y, w, s, q, &[sx], rows, cols) };
            return;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            if simd_tier() >= 2 {
                unsafe { vex_one(y, w, s, q, &[sx], rows, cols) };
            } else {
                unsafe { sse_one(y, w, s, q, &[sx], rows, cols) };
            }
            return;
        }
    }
    let _ = safe;
    for r in 0..rows {
        let mut sum = 0i128;
        for j in 0..cols {
            let off = (r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2;
            sum += i128::from(w[off]) * i128::from(q[j]);
        }
        y[r] = (sum as f32 * s[r]) * sx;
    }
}
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_one(
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
        avx2_tail(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_tail(
    y: &mut [f32],
    w: &[i8],
    s: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in (0..rows).step_by(8) {
        let mut a = _mm256_setzero_si256();
        for j in (0..cols).step_by(2) {
            let x = _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
            let ww = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * cols + j * 8).cast()));
            a = _mm256_add_epi32(a, _mm256_madd_epi16(ww, x));
        }
        _mm256_storeu_ps(
            y.as_mut_ptr().add(r),
            _mm256_mul_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_loadu_ps(s.as_ptr().add(r))),
                _mm256_set1_ps(sx[0]),
            ),
        );
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    one_body!(out, w, sw, q, sx, rows, cols);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,sse4.1")]
unsafe fn vex_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    one_body!(out, w, sw, q, sx, rows, cols);
}
