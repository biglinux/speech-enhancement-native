//! AVX2 product of one vector with a recurrent matrix held widened to i16 (the
//! recurrent cache): bounded i32 sums and the (sum as f32 * sw) * sx epilogue
//! of the int8 kernels, without FMA.
use std::arch::x86_64::*;

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
            let ww = _mm256_loadu_si256(w.as_ptr().add(r * 64 + j * 8).cast());
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
    }
}
