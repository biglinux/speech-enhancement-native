//! R10 quantization epilogues and load-time grouped-linear dispatch.
//! No change to scale, ties-away rounding, clipping, FMA policy or reduction order.

#[cfg(all(target_arch = "x86_64", feature = "r10-quant-pack"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", feature = "r10-quant-pack"))]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn round8(prod: __m256) -> __m256i {
    let trunc = _mm256_round_ps::<0x0B>(prod);
    let frac = _mm256_sub_ps(prod, trunc);
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
    let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_and_ps(frac, abs), _mm256_set1_ps(0.5));
    let bump = _mm256_and_ps(
        _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(prod, sign)),
        ge,
    );
    let rounded = _mm256_min_ps(
        _mm256_max_ps(_mm256_add_ps(trunc, bump), _mm256_set1_ps(-16383.0)),
        _mm256_set1_ps(16383.0),
    );
    _mm256_cvtps_epi32(rounded)
}
#[cfg(all(target_arch = "x86_64", feature = "r10-quant-pack"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn quantize_avx2(x: &[f32], out: &mut [i16], inv: f32) {
    let mut i = 0;
    let scale = _mm256_set1_ps(inv);
    while i + 16 <= x.len() {
        let a = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), scale));
        let b = round8(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), scale));
        // PACKSSDW is lane-local: [a0..3,b0..3,a4..7,b4..7]. Exchange middle
        // qwords so the stored i16 sequence is a0..7,b0..7. No saturation here:
        // round8 has already bounded every value to [-16383,16383].
        let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a, b));
        _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), packed);
        i += 16;
    }
    if i < x.len() {
        super::quantize_round_avx(&x[i..], &mut out[i..], inv);
    }
}
#[cfg(all(target_arch = "x86_64", feature = "r10-quant-pack"))]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn round4(prod: __m128) -> __m128i {
    let trunc = _mm_round_ps::<0x0B>(prod);
    let frac = _mm_sub_ps(prod, trunc);
    let abs = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let sign = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
    let ge = _mm_cmpge_ps(_mm_and_ps(frac, abs), _mm_set1_ps(0.5));
    let bump = _mm_and_ps(_mm_or_ps(_mm_set1_ps(1.0), _mm_and_ps(prod, sign)), ge);
    let rounded = _mm_min_ps(
        _mm_max_ps(_mm_add_ps(trunc, bump), _mm_set1_ps(-16383.0)),
        _mm_set1_ps(16383.0),
    );
    _mm_cvtps_epi32(rounded)
}
#[cfg(all(target_arch = "x86_64", feature = "r10-quant-pack"))]
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn quantize_sse(x: &[f32], out: &mut [i16], inv: f32) {
    let scale = _mm_set1_ps(inv);
    let mut i = 0;
    while i + 8 <= x.len() {
        let a = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i)), scale));
        let b = round4(_mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(i + 4)), scale));
        _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_packs_epi32(a, b));
        i += 8;
    }
    if i < x.len() {
        super::quantize_round_sse(&x[i..], &mut out[i..], inv);
    }
}

// A preselected matvec function, NOT group fusion or a different GEMM schedule.
#[cfg(feature = "r10-linear-plan")]
type MatRun = unsafe fn(&mut [f32], &[f32], &[f32], usize, usize);
#[cfg(feature = "r10-linear-plan")]
#[derive(Clone, Copy)]
pub struct GroupedLinearPlan {
    groups: usize,
    ip: usize,
    op: usize,
    ni: usize,
    no: usize,
    nw: usize,
    run: MatRun,
    legacy_group_fusion: bool,
}
#[cfg(feature = "r10-linear-plan")]
impl GroupedLinearPlan {
    /// Initialization only. Checked dimensions, no owned buffers or allocation.
    pub fn new(groups: usize, ip: usize, op: usize) -> Option<Self> {
        if groups == 0 || ip == 0 || op == 0 {
            return None;
        }
        let ni = groups.checked_mul(ip)?;
        let no = groups.checked_mul(op)?;
        let nw = ni.checked_mul(op)?;
        let mut run: MatRun = scalar;
        #[allow(unused_mut)]
        let mut legacy_group_fusion = false;
        #[cfg(target_arch = "x86_64")]
        {
            match super::simd_tier() {
                3 => {
                    legacy_group_fusion = cfg!(feature = "r9-grouped-fusion")
                        && matches!(op, 8 | 16 | 32)
                        && groups >= 2;
                    run = if cfg!(feature = "r8-matvec-wide") && op >= 64 {
                        super::round8::matvec_avx2
                    } else {
                        super::matvec_t_avx2
                    };
                }
                2 => {
                    run = if cfg!(feature = "r8-matvec-wide") && op >= 64 {
                        super::round8::matvec_avx
                    } else {
                        super::matvec_t_avx
                    };
                }
                1 => {
                    run = if cfg!(feature = "r8-matvec-wide") && op >= 32 {
                        super::round8::matvec_sse
                    } else {
                        super::matvec_t_sse
                    };
                }
                _ => {}
            }
        }
        Some(Self {
            groups,
            ip,
            op,
            ni,
            no,
            nw,
            run,
            legacy_group_fusion,
        })
    }
    /// Same reduction function for each group as grouped_linear; bounds checked
    /// once per invocation. ISA and wide/narrow decisions are not repeated per group.
    pub fn apply(&self, y: &mut [f32], x: &[f32], w: &[f32]) {
        assert!(y.len() >= self.no && x.len() >= self.ni && w.len() >= self.nw);
        if self.legacy_group_fusion {
            super::grouped_linear(y, x, w, self.groups, self.ip, self.op);
            return;
        }
        for g in 0..self.groups {
            // SAFETY: construction checked products, slices validated above, and
            // the function is selected from the actual runtime ISA.
            unsafe {
                (self.run)(
                    &mut y[g * self.op..(g + 1) * self.op],
                    &w[g * self.ip * self.op..(g + 1) * self.ip * self.op],
                    &x[g * self.ip..(g + 1) * self.ip],
                    self.ip,
                    self.op,
                );
            }
        }
    }
}
#[cfg(feature = "r10-linear-plan")]
unsafe fn scalar(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    // Delegate instead of inventing a differently-associated scalar reduction.
    super::matvec_t(y, a, x, m, n);
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_arch = "x86_64", feature = "r10-quant-pack"))]
    #[test]
    fn packed_rounding_matches_existing_for_tails_offsets_and_ties() {
        let mut v = Vec::new();
        for i in -16383..16383 {
            let x = i as f32 + 0.5;
            v.extend_from_slice(&[
                x,
                f32::from_bits(x.to_bits().wrapping_add(1)),
                f32::from_bits(x.to_bits().wrapping_sub(1)),
            ]);
        }
        v.extend_from_slice(&[0.0, -0.0, 1e-39, -1e-39, 20000.0, -20000.0]);
        for tail in 0..17 {
            let x = &v[1..v.len() - tail];
            let mut base = vec![99i16; x.len() + 2];
            let mut y = base.clone();
            if std::is_x86_feature_detected!("sse4.1") {
                unsafe {
                    crate::quantize_round_sse(x, &mut base[1..1 + x.len()], 1.0);
                    super::quantize_sse(x, &mut y[1..1 + x.len()], 1.0);
                }
                assert_eq!(y, base);
            }
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::quantize_round_avx(x, &mut base[1..1 + x.len()], 1.0);
                    super::quantize_avx2(x, &mut y[1..1 + x.len()], 1.0);
                }
                assert_eq!(y, base);
            }
        }
    }
    #[cfg(feature = "r10-linear-plan")]
    #[test]
    fn plans_match_grouped_linear_bitwise() {
        for g in [1, 2, 16, 32] {
            for ip in [1, 7, 16, 64, 128] {
                for op in [1, 5, 8, 16, 32, 64, 128] {
                    let x: Vec<f32> = (0..g * ip)
                        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 101.0)
                        .collect();
                    let w: Vec<f32> = (0..g * ip * op)
                        .map(|i| ((i * 73 % 127) as f32 - 63.0) / 127.0)
                        .collect();
                    let mut expected = vec![0.0; g * op];
                    let mut out = expected.clone();
                    crate::grouped_linear(&mut expected, &x, &w, g, ip, op);
                    super::GroupedLinearPlan::new(g, ip, op)
                        .unwrap()
                        .apply(&mut out, &x, &w);
                    assert!(out
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                }
            }
        }
        assert!(super::GroupedLinearPlan::new(0, 1, 1).is_none());
        assert!(super::GroupedLinearPlan::new(usize::MAX, 2, 2).is_none());
    }
}
