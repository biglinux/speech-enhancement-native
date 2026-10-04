//! Pair-packed W8A16: each INT32 lane is one output, not a partial row sum.
//!
//! Layout: [rows/8, cols/2, 8 output lanes, 2 input-pair members]. It has exactly
//! rows*cols bytes. SSE reads each output tile as two four-lane halves. Neither
//! tier needs VNNI, AVX-512, FMA, a horizontal final reduction or expanded weights.
//!
//! Preconditions established by Matrix/Bundle: rows%8 == 0, cols%2 == 0,
//! cols <= 1024, weights in [-127,127], activations in [-16383,16383]. The sum
//! of absolute products is <= 2_130_576_384, so all INT32 partial sums fit.
//! Two accumulator stripes shorten the dependency chain WITHOUT changing sums.
//! Packing/dispatch happens during construction; processing neither allocates
//! nor locks. Dequantization deliberately remains (sum as f32 * sw) * sx.

#[derive(Clone, Copy)]
pub(crate) struct Plan {
    one: Run,
    four: Run,
}
type Run = unsafe fn(&mut [f32], &[i8], &[f32], &[i16], &[f32], usize, usize);
impl Plan {
    pub(crate) fn select() -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar-reference")))]
        {
            if !cfg!(feature = "force-sse41")
                && !cfg!(feature = "force-avx1")
                && std::is_x86_feature_detected!("avx2")
            {
                return Self {
                    one: wide::avx2_one,
                    four: wide::avx2_four,
                };
            }
            if std::is_x86_feature_detected!("sse4.1") {
                if !cfg!(feature = "force-sse41") && std::is_x86_feature_detected!("avx") {
                    return Self {
                        one: xmm::vex_one,
                        four: wide::vex_cached_four,
                    };
                }
                return Self {
                    one: xmm::sse_one,
                    four: wide::sse_cached_four,
                };
            }
        }
        Self {
            one: scalar::<1>,
            four: scalar::<4>,
        }
    }
    // The operands of one fused GRU step; a struct would only rename them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &self,
        out: &mut [f32],
        w: &[i8],
        sw: &[f32],
        q: &[i16],
        sx: &[f32],
        rows: usize,
        cols: usize,
        count: usize,
    ) {
        assert!(count == 1 || count == 4);
        assert!(
            rows > 0
                && rows.is_multiple_of(8)
                && cols > 0
                && cols.is_multiple_of(2)
                && cols <= 1024
        );
        assert_eq!(w.len(), rows * cols);
        assert_eq!(sw.len(), rows);
        assert_eq!(q.len(), count * cols);
        assert_eq!(out.len(), count * rows);
        assert_eq!(sx.len(), count);
        debug_assert!(q.iter().all(|&v| (-16383..=16383).contains(&v)));
        // SAFETY: private call path produces bounded q, loader validates w, sizes
        // above bound every SIMD access. Plan::select checks the required ISA.
        unsafe { (if count == 4 { self.four } else { self.one })(out, w, sw, q, sx, rows, cols) }
    }
}

pub(crate) fn pack_rows(w: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    assert!(rows > 0 && rows.is_multiple_of(8) && cols > 0 && cols.is_multiple_of(2));
    assert_eq!(w.len(), rows * cols);
    let mut packed = vec![0i8; w.len()];
    for r in 0..rows {
        for j in 0..cols {
            let dst = (r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2;
            packed[dst] = w[r * cols + j];
        }
    }
    packed
}

unsafe fn scalar<const B: usize>(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    for k in 0..B {
        for r in 0..rows {
            let mut sum = 0i32;
            for j in 0..cols {
                let at = (r / 8) * (8 * cols) + (j / 2) * 16 + (r % 8) * 2 + j % 2;
                sum += w[at] as i32 * q[k * cols + j] as i32;
            }
            out[k * rows + r] = (sum as f32 * sw[r]) * sx[k];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2<const B: usize>(
    out: &mut [f32],
    w: &[i8],
    sw: &[f32],
    q: &[i16],
    sx: &[f32],
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    for r in (0..rows).step_by(8) {
        let wp = w.as_ptr().add(r * cols);
        let mut a = [_mm256_setzero_si256(); B];
        let mut b = [_mm256_setzero_si256(); B];
        let mut j = 0;
        while j + 4 <= cols {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(j * 8).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add((j + 2) * 8).cast()));
            for (k, acc) in a.iter_mut().enumerate() {
                let x = q.as_ptr().add(k * cols + j);
                let x0 = _mm256_set1_epi32(std::ptr::read_unaligned(x.cast::<i32>()));
                let x1 = _mm256_set1_epi32(std::ptr::read_unaligned(x.add(2).cast::<i32>()));
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(w0, x0));
                b[k] = _mm256_add_epi32(b[k], _mm256_madd_epi16(w1, x1));
            }
            j += 4;
        }
        if j < cols {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(j * 8).cast()));
            for (k, acc) in a.iter_mut().enumerate() {
                let x0 = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(k * cols + j).cast::<i32>(),
                ));
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(w0, x0));
            }
        }
        let scales = _mm256_loadu_ps(sw.as_ptr().add(r));
        for k in 0..B {
            let sums = _mm256_cvtepi32_ps(_mm256_add_epi32(a[k], b[k]));
            let values = _mm256_mul_ps(_mm256_mul_ps(sums, scales), _mm256_set1_ps(sx[k]));
            _mm256_storeu_ps(out.as_mut_ptr().add(k * rows + r), values);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<const B: usize>(rows: usize, cols: usize, extreme: bool) {
        let w: Vec<i8> = (0..rows * cols)
            .map(|i| {
                if extreme {
                    127
                } else {
                    ((i * 71 + 13) % 255) as i16 as i8
                }
            })
            .collect();
        // Avoid -128 even when testing non-exported random values.
        let w: Vec<i8> = w
            .into_iter()
            .map(|v| if v == -128 { 0 } else { v })
            .collect();
        let q: Vec<i16> = (0..B * cols)
            .map(|i| {
                if extreme {
                    16383
                } else {
                    (((i * 173 + 19) % 32767) as i32 - 16383) as i16
                }
            })
            .collect();
        let sw: Vec<f32> = (0..rows).map(|i| 0.001 * (i % 17 + 1) as f32).collect();
        let sx: Vec<f32> = (0..B).map(|i| 0.0003 * (i + 1) as f32).collect();
        let p = pack_rows(&w, rows, cols);
        let mut reference = vec![0.0; B * rows];
        for k in 0..B {
            for r in 0..rows {
                let s: i64 = (0..cols)
                    .map(|j| w[r * cols + j] as i64 * q[k * cols + j] as i64)
                    .sum();
                reference[k * rows + r] = (s as f32 * sw[r]) * sx[k];
            }
        }
        let check = |run: Run| {
            let mut out = vec![123.25; B * rows + 2];
            unsafe { run(&mut out[1..B * rows + 1], &p, &sw, &q, &sx, rows, cols) };
            assert_eq!(out[0], 123.25);
            assert_eq!(out[B * rows + 1], 123.25);
            assert_eq!(
                out[1..B * rows + 1]
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        };
        check(scalar::<B>);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                if B == 1 {
                    check(xmm::sse_one);
                } else {
                    check(xmm::sse_four);
                    check(wide::sse_cached_four);
                }
            }
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("sse4.1") {
                if B == 1 {
                    check(xmm::vex_one);
                } else {
                    check(xmm::vex_four);
                    check(wide::vex_cached_four);
                }
            }
            if std::is_x86_feature_detected!("avx2") {
                check(avx2::<B>);
                if B == 1 {
                    check(wide::avx2_one);
                } else {
                    check(wide::avx2_four);
                }
            }
        }
    }
    #[test]
    fn pair_packing_and_all_supported_tiers_are_exact() {
        for n in [2, 6, 16, 64, 256, 1024] {
            for m in [8, 24, 32, 64, 72, 192, 768] {
                check::<1>(m, n, false);
                check::<4>(m, n, false);
            }
        }
    }
    #[test]
    fn maximum_int32_bound_is_exact() {
        check::<1>(8, 1024, true);
        check::<4>(8, 1024, true);
    }
}

#[cfg(target_arch = "x86_64")]
#[path = "packed_xmm.rs"]
mod xmm;

#[cfg(target_arch = "x86_64")]
#[path = "packed_cached.rs"]
mod cached;
#[cfg(target_arch = "x86_64")]
#[path = "packed_wide.rs"]
mod wide;

#[cfg(target_arch = "x86_64")]
pub(crate) fn cached_recurrent_one(y: &mut [f32], w: &[i16], sw: &[f32], q: &[i16], sx: f32) {
    assert_eq!(y.len(), 192);
    assert_eq!(w.len(), 192 * 64);
    assert_eq!(sw.len(), 192);
    assert_eq!(q.len(), 64);
    debug_assert!(q.iter().all(|&x| (-16383..=16383).contains(&x)));
    // SAFETY: only Matrix::enable_recurrent_cache enables this path after AVX2
    // detection. Bundle::recurrent_cache validates copied bounded i8 values.
    unsafe { cached::cached_one(y, w, sw, q, sx) }
}
