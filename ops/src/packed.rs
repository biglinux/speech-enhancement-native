//! Exact W8A16 GEMV over the pair-packed matrices of DFN3 (768x256) and
//! DFN3-LL (1536x512), laid out `[rows/8, cols/2, lane, pair member]`
//! (see `pack_format`). AVX2 keeps 64 output accumulators per tile, SSE 32. The
//! integer sums are exact, so every tier gives the scalar product's bits.
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
    // The SIMD kernels accumulate in i32: exact for any i16 input up to 511 columns,
    // beyond that only for small enough activations. The quantizer emits ±16383.
    #[cfg(target_arch = "x86_64")]
    if cols <= 511
        || q[..cols]
            .iter()
            .fold(0u64, |a, &x| a.saturating_add(i64::from(x).unsigned_abs()))
            <= i32::MAX as u64 / 128
    {
        if !cfg!(feature = "force-sse41")
            && !cfg!(feature = "force-avx1")
            && std::is_x86_feature_detected!("avx2")
        {
            // SAFETY: AVX2 was just detected; the asserts above give the shapes.
            unsafe { avx2_one(y, w, s, q, sx, rows, cols) };
            return;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            if simd_tier() >= 2 {
                // SAFETY: SSE4.1 was just detected and tier 2 means AVX; the asserts
                // above give the shapes.
                unsafe { vex_one(y, w, s, q, sx, rows, cols) };
            } else {
                // SAFETY: SSE4.1 was just detected; the asserts above give the shapes.
                unsafe { sse_one(y, w, s, q, sx, rows, cols) };
            }
            return;
        }
    }
    scalar(y, w, s, q, sx, rows, cols);
}

fn scalar(y: &mut [f32], w: &[i8], s: &[f32], q: &[i16], sx: f32, rows: usize, cols: usize) {
    for r in 0..rows {
        // At most 1024 · 128 · 32768 = 2^32 in magnitude.
        let mut sum = 0i64;
        for j in 0..cols {
            let off = (r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2;
            sum += i64::from(w[off]) * i64::from(q[j]);
        }
        y[r] = (sum as f32 * s[r]) * sx;
    }
}
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// # Safety
/// The CPU must support AVX2. `rows` must be a multiple of 8 and `cols` of 2,
/// `out` and `sw` must hold `rows` values, `q` `cols` and `w` `rows*cols`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: f32,
    rows: usize,
    cols: usize,
) {
    // SAFETY: row tile `r` reads `w[r*cols..]` for `cols*8` bytes per 8 rows, `q`
    // in pairs below `cols`, and `sw[r..]`/`out[r..]` for its rows, all below
    // `rows`, inside the caller's lengths.
    unsafe {
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
                let xx =
                    _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
                let w0 =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * cols + j * 8).cast()));
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
            let xs = _mm256_set1_ps(sx);
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
}

/// # Safety
/// The CPU must support AVX2. `rows` must be a multiple of 8 and `cols` of 2,
/// `y` and `s` must hold `rows` values, `q` `cols` and `w` `rows*cols`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_tail(
    y: &mut [f32],
    w: &[i8],
    s: &[f32],
    q: &[i16],
    sx: f32,
    rows: usize,
    cols: usize,
) {
    // SAFETY: row tile `r` reads `w[r*cols..]` for `cols*8` bytes per 8 rows, `q`
    // in pairs below `cols`, and `s[r..]`/`y[r..]` for its rows, all below
    // `rows`, inside the caller's lengths.
    unsafe {
        for r in (0..rows).step_by(8) {
            let mut a = _mm256_setzero_si256();
            for j in (0..cols).step_by(2) {
                let x =
                    _mm256_set1_epi32(std::ptr::read_unaligned(q.as_ptr().add(j).cast::<i32>()));
                let ww =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(r * cols + j * 8).cast()));
                a = _mm256_add_epi32(a, _mm256_madd_epi16(ww, x));
            }
            _mm256_storeu_ps(
                y.as_mut_ptr().add(r),
                _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_loadu_ps(s.as_ptr().add(r))),
                    _mm256_set1_ps(sx),
                ),
            );
        }
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
                // One broadcast serves all 32 outputs.
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
            let xs = _mm_set1_ps(sx);
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
        // Rows past the last 32-row tile; the shipped 768- and 1536-row matrices have none.
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
            let xs = _mm_set1_ps(sx);
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

/// # Safety
/// The CPU must support SSE4.1. `rows` must be a multiple of 8 and `cols` of 2,
/// `out` and `sw` must hold `rows` values, `q` `cols` and `w` `rows*cols`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: f32,
    rows: usize,
    cols: usize,
) {
    // SAFETY: row tile `r` reads `w[r*cols..]` for `cols*8` bytes per 8 rows, `q`
    // in pairs below `cols`, and `sw[r..]`/`out[r..]` for its rows, all below
    // `rows`, inside the caller's lengths.
    unsafe {
        one_body!(out, w, sw, q, sx, rows, cols);
    }
}

/// [`sse_one`] compiled with AVX enabled, so it uses VEX encodings: on AVX CPUs,
/// legacy SSE instructions between the 256-bit kernels of the same hop pay a
/// state-transition penalty.
///
/// # Safety
/// The CPU must support AVX and SSE4.1. `rows` must be a multiple of 8 and `cols` of 2,
/// `out` and `sw` must hold `rows` values, `q` `cols` and `w` `rows*cols`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,sse4.1")]
unsafe fn vex_one(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: f32,
    rows: usize,
    cols: usize,
) {
    // SAFETY: row tile `r` reads `w[r*cols..]` for `cols*8` bytes per 8 rows, `q`
    // in pairs below `cols`, and `sw[r..]`/`out[r..]` for its rows, all below
    // `rows`, inside the caller's lengths.
    unsafe {
        one_body!(out, w, sw, q, sx, rows, cols);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Products through the packed layout equal the plain row-major product,
    // including activations past the exact-i32 bound that take the scalar path.
    #[test]
    fn packed_products_equal_the_row_major_product() {
        for n in [2, 6, 64, 256, 512, 1024] {
            for m in [8, 24, 32, 64, 72, 192, 768, 1536] {
                let w: Vec<i8> = (0..m * n).map(|i| ((i * 79 % 256) as u8) as i8).collect();
                let bytes: Vec<u8> = w.iter().map(|&x| x as u8).collect();
                let packed = crate::pack_format::pack_matrix(&bytes, m, n).unwrap();
                let packed: Vec<i8> = packed.into_iter().map(|x| x as i8).collect();
                let scales: Vec<f32> = (0..m).map(|i| 0.001 * (i % 9 + 1) as f32).collect();
                for mode in 0..3 {
                    let q: Vec<i16> = (0..n)
                        .map(|j| match mode {
                            0 => 16383,
                            1 => ((j * 193 % 32767) as i32 - 16383) as i16,
                            _ => i16::MIN,
                        })
                        .collect();
                    let mut y = vec![0.0; m];
                    matvec(&mut y, &packed, &scales, &q, 0.004, m, n);
                    for r in 0..m {
                        let v: i64 = (0..n)
                            .map(|j| i64::from(w[r * n + j]) * i64::from(q[j]))
                            .sum();
                        let v = (v as f32 * scales[r]) * 0.004;
                        assert_eq!(v.to_bits(), y[r].to_bits(), "m={m} n={n} row={r}");
                    }
                }
            }
        }
    }

    // The dispatch only runs the host's tier, so call every kernel the host supports.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_tier_is_bit_identical_to_the_scalar_product() {
        let mut seed = 0x1234_5678_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for (rows, cols) in [
            (8, 2),
            (24, 6),
            (72, 64),
            (768, 256),
            (1536, 512),
            (64, 1024),
        ] {
            let w: Vec<i8> = (0..rows * cols).map(|_| next() as i8).collect();
            let s: Vec<f32> = (0..rows).map(|i| 0.001 * (i % 9 + 1) as f32).collect();
            let q: Vec<i16> = (0..cols).map(|_| (next() % 32767) as i16 - 16383).collect();
            let mut want = vec![0.0; rows];
            scalar(&mut want, &w, &s, &q, 0.004, rows, cols);
            let bits = |y: &[f32]| y.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let mut got = vec![0.0; rows];
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 was just detected; every shape has rows divisible by 8,
                // even cols and buffers of exactly that size.
                unsafe { avx2_one(&mut got, &w, &s, &q, 0.004, rows, cols) };
                assert_eq!(bits(&got), bits(&want), "avx2 {rows}x{cols}");
            }
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
                // SAFETY: AVX and SSE4.1 were just detected; shapes as above.
                unsafe { vex_one(&mut got, &w, &s, &q, 0.004, rows, cols) };
                assert_eq!(bits(&got), bits(&want), "avx {rows}x{cols}");
            }
            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: SSE4.1 was just detected; shapes as above.
                unsafe { sse_one(&mut got, &w, &s, &q, 0.004, rows, cols) };
                assert_eq!(bits(&got), bits(&want), "sse4.1 {rows}x{cols}");
            }
        }
    }
}
