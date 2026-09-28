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
                if cfg!(feature = "r9-packed-k4") {
                    #[cfg(feature = "r9-packed-k4")]
                    return Self {
                        one: round9::one,
                        four: round9::four,
                    };
                }
                if cfg!(feature = "r8-packed-tiles") {
                    #[cfg(feature = "r10-avx2-prebroadcast")]
                    return Self {
                        one: round8::avx2_one,
                        four: round10::compact_four,
                    };
                    #[cfg(not(feature = "r10-avx2-prebroadcast"))]
                    return Self {
                        one: round8::avx2_one,
                        four: round8::avx2_four,
                    };
                }
                #[cfg(feature = "r10-avx2-prebroadcast")]
                return Self {
                    one: avx2::<1>,
                    four: round10::compact_four,
                };
                #[cfg(not(feature = "r10-avx2-prebroadcast"))]
                return Self {
                    one: avx2::<1>,
                    four: avx2::<4>,
                };
            }
            if std::is_x86_feature_detected!("sse4.1") {
                #[cfg(any(feature = "oldcpu-int-tiles", feature = "oldcpu-vex128"))]
                {
                    if cfg!(feature = "oldcpu-vex128")
                        && !cfg!(feature = "force-sse41")
                        && std::is_x86_feature_detected!("avx")
                    {
                        if cfg!(feature = "oldcpu-int-tiles") {
                            return Self {
                                one: oldcpu::vex_one,
                                four: if cfg!(feature = "r8-sse-prebroadcast") {
                                    round8::vex_cached_four
                                } else {
                                    oldcpu::vex_four
                                },
                            };
                        }
                        return Self {
                            one: oldcpu::vex_legacy::<1>,
                            four: oldcpu::vex_legacy::<4>,
                        };
                    }
                    if cfg!(feature = "oldcpu-int-tiles") {
                        return Self {
                            one: oldcpu::sse_one,
                            four: if cfg!(feature = "r8-sse-prebroadcast") {
                                round8::sse_cached_four
                            } else {
                                oldcpu::sse_four
                            },
                        };
                    }
                }
                return Self {
                    one: sse41::<1>,
                    four: sse41::<4>,
                };
            }
        }
        Self {
            one: scalar::<1>,
            four: scalar::<4>,
        }
    }
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
        assert!(rows > 0 && rows % 8 == 0 && cols > 0 && cols % 2 == 0 && cols <= 1024);
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
    assert!(rows > 0 && rows % 8 == 0 && cols > 0 && cols % 2 == 0);
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
            for k in 0..B {
                let x = q.as_ptr().add(k * cols + j);
                let x0 = _mm256_set1_epi32(std::ptr::read_unaligned(x.cast::<i32>()));
                let x1 = _mm256_set1_epi32(std::ptr::read_unaligned(x.add(2).cast::<i32>()));
                a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(w0, x0));
                b[k] = _mm256_add_epi32(b[k], _mm256_madd_epi16(w1, x1));
            }
            j += 4;
        }
        if j < cols {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(j * 8).cast()));
            for k in 0..B {
                let x0 = _mm256_set1_epi32(std::ptr::read_unaligned(
                    q.as_ptr().add(k * cols + j).cast::<i32>(),
                ));
                a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(w0, x0));
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse41<const B: usize>(
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
        // Eight accumulators for B=4, not sixteen; the second half is a separate
        // reduction. This keeps the SSE implementation within the register file.
        for half in [0usize, 4] {
            let mut a = [_mm_setzero_si128(); B];
            let mut b = [_mm_setzero_si128(); B];
            let mut j = 0;
            while j + 4 <= cols {
                let w0 = _mm_cvtepi8_epi16(_mm_loadl_epi64(wp.add(j * 8 + half * 2).cast()));
                let w1 = _mm_cvtepi8_epi16(_mm_loadl_epi64(wp.add((j + 2) * 8 + half * 2).cast()));
                for k in 0..B {
                    let x = q.as_ptr().add(k * cols + j);
                    let x0 = _mm_set1_epi32(std::ptr::read_unaligned(x.cast::<i32>()));
                    let x1 = _mm_set1_epi32(std::ptr::read_unaligned(x.add(2).cast::<i32>()));
                    a[k] = _mm_add_epi32(a[k], _mm_madd_epi16(w0, x0));
                    b[k] = _mm_add_epi32(b[k], _mm_madd_epi16(w1, x1));
                }
                j += 4;
            }
            if j < cols {
                let w0 = _mm_cvtepi8_epi16(_mm_loadl_epi64(wp.add(j * 8 + half * 2).cast()));
                for k in 0..B {
                    let x0 = _mm_set1_epi32(std::ptr::read_unaligned(
                        q.as_ptr().add(k * cols + j).cast::<i32>(),
                    ));
                    a[k] = _mm_add_epi32(a[k], _mm_madd_epi16(w0, x0));
                }
            }
            let scales = _mm_loadu_ps(sw.as_ptr().add(r + half));
            for k in 0..B {
                let sums = _mm_cvtepi32_ps(_mm_add_epi32(a[k], b[k]));
                let values = _mm_mul_ps(_mm_mul_ps(sums, scales), _mm_set1_ps(sx[k]));
                _mm_storeu_ps(out.as_mut_ptr().add(k * rows + r + half), values);
            }
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
                check(sse41::<B>);
                if B == 1 {
                    check(oldcpu::sse_one);
                } else {
                    check(oldcpu::sse_four);
                    check(round8::sse_cached_four);
                }
            }
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("sse4.1") {
                check(oldcpu::vex_legacy::<B>);
                if B == 1 {
                    check(oldcpu::vex_one);
                } else {
                    check(oldcpu::vex_four);
                    check(round8::vex_cached_four);
                }
            }
            if std::is_x86_feature_detected!("avx2") {
                check(avx2::<B>);
                if B == 1 {
                    check(round8::avx2_one);
                    check(round9::one);
                } else {
                    check(round8::avx2_four);
                    check(round9::four);
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
#[path = "packed_oldcpu.rs"]
mod oldcpu;

#[cfg(target_arch = "x86_64")]
#[path = "packed_round8.rs"]
mod round8;

type PairRun =
    unsafe fn(&mut [f32], &mut [f32], &[i8], &[i8], &[f32], &[f32], &[i16], &[f32], usize, usize);
/// Paired F/B projection selected at construction. All buffers remain independent.
#[derive(Clone, Copy)]
pub(crate) struct PairPlan(Option<PairRun>);
impl PairPlan {
    pub(crate) fn select() -> Self {
        #[cfg(all(
            target_arch = "x86_64",
            feature = "r8-fb-pair",
            not(feature = "scalar-reference")
        ))]
        {
            if !cfg!(feature = "force-sse41")
                && !cfg!(feature = "force-avx1")
                && std::is_x86_feature_detected!("avx2")
            {
                return Self(Some(round8::avx2_pair));
            }
            if std::is_x86_feature_detected!("sse4.1") {
                if !cfg!(feature = "force-sse41")
                    && cfg!(feature = "oldcpu-vex128")
                    && std::is_x86_feature_detected!("avx")
                {
                    return Self(Some(round8::vex_pair));
                }
                return Self(Some(round8::sse_pair));
            }
        }
        Self(None)
    }
    pub(crate) fn available(&self) -> bool {
        self.0.is_some()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &self,
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
        assert!(rows > 0 && rows % 8 == 0 && cols > 0 && cols % 2 == 0 && cols <= 1024);
        assert_eq!(wa.len(), rows * cols);
        assert_eq!(wb.len(), rows * cols);
        assert_eq!(sa.len(), rows);
        assert_eq!(sb.len(), rows);
        assert_eq!(ya.len(), 4 * rows);
        assert_eq!(yb.len(), 4 * rows);
        assert_eq!(q.len(), 4 * cols);
        assert_eq!(sx.len(), 4);
        debug_assert!(q.iter().all(|&v| (-16383..=16383).contains(&v)));
        // SAFETY: Matrix loader validates bounded weights; caller quantizes q to
        // +/-16383. Distinct mutable outputs, checked slices, ISA selected once.
        unsafe {
            (self.0.expect("paired kernel not selected"))(ya, yb, wa, wb, sa, sb, q, sx, rows, cols)
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod round8_pair_tests {
    use super::*;
    #[test]
    fn two_different_packed_matrices_match_independent_int64() {
        let mut funcs: Vec<PairRun> = Vec::new();
        if std::is_x86_feature_detected!("sse4.1") {
            funcs.push(round8::sse_pair);
        }
        if std::is_x86_feature_detected!("avx") {
            funcs.push(round8::vex_pair);
        }
        if std::is_x86_feature_detected!("avx2") {
            funcs.push(round8::avx2_pair);
        }
        for rows in [8usize, 24, 64, 72, 192, 768] {
            for cols in [2usize, 6, 16, 64, 256, 1024] {
                for extreme in [false, true] {
                    let wa: Vec<i8> = (0..rows * cols)
                        .map(|i| {
                            if extreme {
                                127
                            } else {
                                ((i * 71 % 255) as i16 - 127) as i8
                            }
                        })
                        .collect();
                    let wb: Vec<i8> = (0..rows * cols)
                        .map(|i| {
                            if extreme {
                                -127
                            } else {
                                ((i * 31 + 19) % 255) as i16 as i8
                            }
                        })
                        .map(|x| if x == -128 { 1 } else { x })
                        .collect();
                    let q: Vec<i16> = (0..4 * cols)
                        .map(|i| {
                            if extreme {
                                if i / cols % 2 == 0 {
                                    16383
                                } else {
                                    -16383
                                }
                            } else {
                                ((i * 173 % 32767) as i32 - 16383) as i16
                            }
                        })
                        .collect();
                    let sa: Vec<f32> = (0..rows).map(|i| 0.001 * (i % 17 + 1) as f32).collect();
                    let sb: Vec<f32> = (0..rows).map(|i| 0.002 * (i % 13 + 1) as f32).collect();
                    let sx = [0.0f32, 1e-4, 2e-4, 0.125];
                    let pa = pack_rows(&wa, rows, cols);
                    let pb = pack_rows(&wb, rows, cols);
                    let mut ra = vec![0.0; 4 * rows];
                    let mut rb = ra.clone();
                    for (w, s, y) in [(&wa, &sa, &mut ra), (&wb, &sb, &mut rb)] {
                        for k in 0..4 {
                            for row in 0..rows {
                                let sum: i64 = (0..cols)
                                    .map(|j| w[row * cols + j] as i64 * q[k * cols + j] as i64)
                                    .sum();
                                y[k * rows + row] = (sum as f32 * s[row]) * sx[k];
                            }
                        }
                    }
                    for &f in &funcs {
                        let mut ya = vec![123.25; 4 * rows + 2];
                        let mut yb = ya.clone();
                        unsafe {
                            f(
                                &mut ya[1..4 * rows + 1],
                                &mut yb[1..4 * rows + 1],
                                &pa,
                                &pb,
                                &sa,
                                &sb,
                                &q,
                                &sx,
                                rows,
                                cols,
                            );
                        }
                        assert_eq!(ya[0], 123.25);
                        assert_eq!(yb[4 * rows + 1], 123.25);
                        crate::test_support::assert_bits(&ya[1..4 * rows + 1], &ra);
                        crate::test_support::assert_bits(&yb[1..4 * rows + 1], &rb);
                    }
                }
            }
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "r9-packed-k4", feature = "r9-recurrent-cache", test)
))]
#[path = "packed_round9.rs"]
mod round9;

#[cfg(all(target_arch = "x86_64", feature = "r9-recurrent-cache"))]
pub(crate) fn cached_recurrent_one(y: &mut [f32], w: &[i16], sw: &[f32], q: &[i16], sx: f32) {
    assert_eq!(y.len(), 192);
    assert_eq!(w.len(), 192 * 64);
    assert_eq!(sw.len(), 192);
    assert_eq!(q.len(), 64);
    debug_assert!(q.iter().all(|&x| (-16383..=16383).contains(&x)));
    // SAFETY: only Matrix::enable_recurrent_cache enables this path after AVX2
    // detection. Bundle::recurrent_cache validates copied bounded i8 values.
    unsafe { round9::cached_one(y, w, sw, q, sx) }
}

#[cfg(all(
    target_arch = "x86_64",
    any(feature = "r10-batch-cache", feature = "r10-avx2-prebroadcast")
))]
#[path = "packed_round10.rs"]
mod round10;

#[cfg(feature = "r10-batch-cache")]
type CacheRun = unsafe fn(&mut [f32], &[i16], &[f32], &[i16], &[f32], usize);
#[cfg(feature = "r10-batch-cache")]
#[derive(Clone, Copy)]
pub(crate) struct BatchCachePlan(Option<CacheRun>);
#[cfg(feature = "r10-batch-cache")]
impl BatchCachePlan {
    pub(crate) fn select() -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar-reference")))]
        {
            if !cfg!(feature = "force-sse41")
                && !cfg!(feature = "force-avx1")
                && std::is_x86_feature_detected!("avx2")
            {
                return Self(Some(round10::cached_four_avx2));
            }
            if std::is_x86_feature_detected!("sse4.1") {
                if cfg!(feature = "oldcpu-vex128")
                    && !cfg!(feature = "force-sse41")
                    && std::is_x86_feature_detected!("avx")
                {
                    return Self(Some(round10::cached_four_vex));
                }
                return Self(Some(round10::cached_four_sse));
            }
        }
        Self(None)
    }
    pub(crate) fn available(&self) -> bool {
        self.0.is_some()
    }
    pub(crate) fn apply(&self, out: &mut [f32], w: &[i16], s: &[f32], q: &[i16], sx: &[f32; 4]) {
        assert_eq!(w.len(), 192 * 64);
        assert_eq!(s.len(), 192);
        assert_eq!(q.len(), 4 * 64);
        assert_eq!(out.len(), 4 * 192);
        debug_assert!(w.iter().all(|&x| (-127..=127).contains(&x)));
        debug_assert!(q.iter().all(|&x| (-16383..=16383).contains(&x)));
        // SAFETY: immutable loader-validated cache, bounded quantizer output,
        // checked slices and the ISA selected during construction.
        unsafe { (self.0.expect("batch cache without a supported ISA"))(out, w, s, q, sx, 192) }
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(feature = "r10-batch-cache", feature = "r10-avx2-prebroadcast")
))]
mod round10_tests {
    use super::*;
    #[test]
    fn four_vector_candidates_match_int64_oracle() {
        for seed in 0..24usize {
            let m = 192;
            let n = 64;
            let w: Vec<i8> = (0..m * n)
                .map(|i| {
                    if seed == 0 {
                        127
                    } else {
                        ((i * 73 + seed * 31) % 255) as i16 as i8
                    }
                })
                .map(|x| if x == -128 { 0 } else { x })
                .collect();
            let q: Vec<i16> = (0..4 * n)
                .map(|i| {
                    if seed == 0 {
                        16383
                    } else {
                        (((i * 337 + seed * 41) % 32767) as i32 - 16383) as i16
                    }
                })
                .collect();
            let scales: Vec<f32> = (0..m).map(|i| 0.0001 * ((i % 11 + 1) as f32)).collect();
            let sx = [0.0, 0.0003, 1.0, 0.00007];
            let packed = pack_rows(&w, m, n);
            let mut expected = vec![0.0; 4 * m];
            for k in 0..4 {
                for r in 0..m {
                    let sum: i64 = (0..n)
                        .map(|j| w[r * n + j] as i64 * q[k * n + j] as i64)
                        .sum();
                    expected[k * m + r] = (sum as f32 * scales[r]) * sx[k];
                }
            }
            #[cfg(feature = "r10-batch-cache")]
            {
                let wide: Vec<i16> = packed.iter().map(|&x| i16::from(x)).collect();
                let mut runs: Vec<CacheRun> = Vec::new();
                if std::is_x86_feature_detected!("sse4.1") {
                    runs.push(round10::cached_four_sse);
                }
                if std::is_x86_feature_detected!("sse4.1") && std::is_x86_feature_detected!("avx") {
                    runs.push(round10::cached_four_vex);
                }
                if std::is_x86_feature_detected!("avx2") {
                    runs.push(round10::cached_four_avx2);
                }
                for run in runs {
                    let mut y = vec![123.0; 4 * m + 2];
                    unsafe {
                        run(&mut y[1..1 + 4 * m], &wide, &scales, &q, &sx, m);
                    }
                    assert_eq!(y[0], 123.0);
                    assert_eq!(y[4 * m + 1], 123.0);
                    assert!(y[1..1 + 4 * m]
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                }
            }
            #[cfg(feature = "r10-avx2-prebroadcast")]
            if std::is_x86_feature_detected!("avx2") {
                let mut y = vec![0.0; 4 * m];
                unsafe {
                    round10::compact_four(&mut y, &packed, &scales, &q, &sx, m, n);
                }
                assert!(y
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
}
