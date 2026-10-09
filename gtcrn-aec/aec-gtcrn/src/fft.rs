//! Radix-2 FFTs on precomputed f32 twiddle tables, shared by the DAF front end
//! and the high-band canceller. Callers own every buffer, so no call allocates.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

/// In-place complex FFT of `re`/`im` (power-of-two length `n`). `twc`/`tws` is
/// the forward table from [`fill_twiddles`]; the inverse conjugates it and
/// scales by `1/n`. The parts are public so a caller can spread one transform
/// over several calls: [`bit_reverse`], then [`butterflies`] for
/// `len = 2, 4, ..., n`, then [`scale_inverse`] for the inverse.
#[inline(always)]
pub fn fft_inplace(re: &mut [f32], im: &mut [f32], inv: bool, twc: &[f32], tws: &[f32]) {
    bit_reverse(re, im);
    let mut len = 2;
    while len <= re.len() {
        butterflies(re, im, len, inv, twc, tws);
        len <<= 1;
    }
    if inv {
        scale_inverse(re, im);
    }
}

/// Bit-reversal permutation, the first step of [`fft_inplace`].
#[inline(always)]
pub fn bit_reverse(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    bit_reverse_part(re, im, 0..n);
}

/// The part of [`bit_reverse`] that starts at the indices in `part`; the
/// parts of any split of `0..n` together make the whole permutation.
#[inline(always)]
pub fn bit_reverse_part(re: &mut [f32], im: &mut [f32], part: std::ops::Range<usize>) {
    let n = re.len();
    if n < 2 {
        return;
    }
    let shift = usize::BITS - n.trailing_zeros();
    for i in part {
        let j = i.reverse_bits() >> shift;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

/// One radix-2 stage of [`fft_inplace`]: butterflies `len / 2` apart.
#[inline(always)]
pub fn butterflies(
    re: &mut [f32],
    im: &mut [f32],
    len: usize,
    inv: bool,
    twc: &[f32],
    tws: &[f32],
) {
    let n = re.len();
    let h = len / 2;
    let stride = n / len;
    let mut i = 0;
    while i < n {
        for k in 0..h {
            let m = k * stride;
            let cr = twc[m];
            let ci = if inv { -tws[m] } else { tws[m] };
            let ur = re[i + k];
            let ui = im[i + k];
            let br = re[i + k + h];
            let bi = im[i + k + h];
            let vr = br * cr - bi * ci;
            let vi = br * ci + bi * cr;
            re[i + k] = ur + vr;
            im[i + k] = ui + vi;
            re[i + k + h] = ur - vr;
            im[i + k + h] = ui - vi;
        }
        i += len;
    }
}

/// The `1/n` scaling that ends an inverse [`fft_inplace`].
#[inline(always)]
pub fn scale_inverse(re: &mut [f32], im: &mut [f32]) {
    let s = 1.0 / re.len() as f32;
    for i in 0..re.len() {
        re[i] *= s;
        im[i] *= s;
    }
}

/// Fill a forward twiddle table for an `n`-point FFT: `[cos(-2πm/n), sin(-2πm/n)]`
/// for `m in 0..n/2`, computed in f64 then stored f32.
pub fn fill_twiddles(n: usize, twc: &mut [f32], tws: &mut [f32]) {
    for m in 0..n / 2 {
        let th = -2.0 * std::f64::consts::PI * m as f64 / n as f64;
        twc[m] = th.cos() as f32;
        tws[m] = th.sin() as f32;
    }
}

/// Real FFT of `x[0..n]` into the half spectrum `outr/outi[0..=n/2]`: the input
/// is packed as `n/2` complex points, transformed with the `n/2` table
/// (`twc_h`/`tws_h`) and recombined with the `n` table. `sre`/`sim` are scratch
/// of length >= `n/2`.
#[inline(always)]
pub fn rfft(
    x: &[f32],
    n: usize,
    outr: &mut [f32],
    outi: &mut [f32],
    sre: &mut [f32],
    sim: &mut [f32],
    twc: &[f32],
    tws: &[f32],
    twc_h: &[f32],
    tws_h: &[f32],
) {
    let n2 = n / 2;
    rfft_pack(&x[..n], &mut sre[..n2], &mut sim[..n2]);
    fft_inplace(&mut sre[..n2], &mut sim[..n2], false, twc_h, tws_h);
    rfft_combine(&sre[..n2], &sim[..n2], outr, outi, twc, tws, 0..n2 + 1);
}

/// First step of [`rfft`]: pack real `x` as `x.len() / 2` complex points.
#[inline(always)]
pub fn rfft_pack(x: &[f32], sre: &mut [f32], sim: &mut [f32]) {
    for j in 0..sre.len() {
        sre[j] = x[2 * j];
        sim[j] = x[2 * j + 1];
    }
}

/// Last step of [`rfft`]: split the transformed packed points `sre`/`sim`
/// (length `n/2`) into the bins `part` of the half spectrum `outr/outi`,
/// `0..n/2 + 1` for all of it.
#[inline(always)]
pub fn rfft_combine(
    sre: &[f32],
    sim: &[f32],
    outr: &mut [f32],
    outi: &mut [f32],
    twc: &[f32],
    tws: &[f32],
    part: std::ops::Range<usize>,
) {
    let n2 = sre.len();
    for k in part {
        let (zr, zi) = (sre[k % n2], sim[k % n2]);
        let (mr, mi) = (sre[(n2 - k) % n2], sim[(n2 - k) % n2]);
        let xer = 0.5 * (zr + mr);
        let xei = 0.5 * (zi - mi);
        let xo_r = 0.5 * (zi + mi);
        let xo_i = -0.5 * (zr - mr);
        let (wr, wi) = if k < n2 {
            (twc[k], tws[k])
        } else {
            (-1.0, 0.0)
        };
        outr[k] = xer + wr * xo_r - wi * xo_i;
        outi[k] = xei + wr * xo_i + wi * xo_r;
    }
}

/// Inverse of [`rfft`]: half spectrum `br/bi[0..=n/2]` to real `out[0..n]`.
#[inline(always)]
pub fn irfft(
    br: &[f32],
    bi: &[f32],
    n: usize,
    out: &mut [f32],
    sre: &mut [f32],
    sim: &mut [f32],
    twc: &[f32],
    tws: &[f32],
    twc_h: &[f32],
    tws_h: &[f32],
) {
    let n2 = n / 2;
    irfft_split(br, bi, &mut sre[..n2], &mut sim[..n2], twc, tws, 0..n2);
    fft_inplace(&mut sre[..n2], &mut sim[..n2], true, twc_h, tws_h);
    irfft_unpack(&sre[..n2], &sim[..n2], &mut out[..n]);
}

/// First step of [`irfft`]: fold the half spectrum into the packed complex
/// points `part` of `sre`/`sim` (length `n/2`), ready for the inverse FFT.
#[inline(always)]
pub fn irfft_split(
    br: &[f32],
    bi: &[f32],
    sre: &mut [f32],
    sim: &mut [f32],
    twc: &[f32],
    tws: &[f32],
    part: std::ops::Range<usize>,
) {
    let n2 = sre.len();
    for k in part {
        let (xr, xi) = (br[k], bi[k]);
        let (yr, yi) = (br[n2 - k], bi[n2 - k]); // X[n2-k]
        let xer = 0.5 * (xr + yr);
        let xei = 0.5 * (xi - yi);
        let dr = xr - yr;
        let di = xi + yi;
        let (wr, wi) = (twc[k], tws[k]); // k < n2 here
        // Xo = 0.5 * conj(W_k) * (X[k] - conj(X[n2-k]))
        let xo_r = 0.5 * (wr * dr + wi * di);
        let xo_i = 0.5 * (wr * di - wi * dr);
        // Z[k] = Xe + i*Xo
        sre[k] = xer - xo_i;
        sim[k] = xei + xo_r;
    }
}

/// Last step of [`irfft`]: interleave the inverse-transformed points into `out`.
#[inline(always)]
pub fn irfft_unpack(sre: &[f32], sim: &[f32], out: &mut [f32]) {
    for j in 0..sre.len() {
        out[2 * j] = sre[j];
        out[2 * j + 1] = sim[j];
    }
}

#[cfg(test)]
mod tests {
    use super::{fft_inplace, fill_twiddles, irfft, rfft};

    // Naive DFT ground truth for real input: X[k] = Σ_t x[t] e^{-2πi k t/n}.
    fn naive(x: &[f32], n: usize, k: usize) -> (f64, f64) {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for t in 0..n {
            let a = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += x[t] as f64 * a.cos();
            im += x[t] as f64 * a.sin();
        }
        (re, im)
    }

    fn tables(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut c, mut s) = (vec![0.0; n / 2], vec![0.0; n / 2]);
        let (mut hc, mut hs) = (vec![0.0; n / 4], vec![0.0; n / 4]);
        fill_twiddles(n, &mut c, &mut s);
        fill_twiddles(n / 2, &mut hc, &mut hs);
        (c, s, hc, hs)
    }

    #[test]
    fn rfft_matches_naive_and_round_trips() {
        for &n in &[256usize, 512, 32768] {
            let (c, s, hc, hs) = tables(n);
            let mut seed = 12345u32;
            let mut g = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / 16_777_216.0 - 0.5
            };
            let x: Vec<f32> = (0..n).map(|_| g()).collect();
            let (mut outr, mut outi) = (vec![0.0f32; n / 2 + 1], vec![0.0f32; n / 2 + 1]);
            let (mut sre, mut sim) = (vec![0.0f32; n], vec![0.0f32; n]);
            rfft(
                &x, n, &mut outr, &mut outi, &mut sre, &mut sim, &c, &s, &hc, &hs,
            );
            // spectrum vs naive DFT (scale tolerance with n: bigger n accumulates more f32 error)
            let tol = 2e-3 * n as f64;
            for k in 0..=n / 2 {
                let (re, im) = naive(&x, n, k);
                assert!(
                    (outr[k] as f64 - re).abs() < tol && (outi[k] as f64 - im).abs() < tol,
                    "n={n} k={k}: ({},{}) vs naive ({re:.2},{im:.2})",
                    outr[k],
                    outi[k]
                );
            }
            // round-trip irfft(rfft(x)) == x
            let mut back = vec![0.0f32; n];
            irfft(
                &outr, &outi, n, &mut back, &mut sre, &mut sim, &c, &s, &hc, &hs,
            );
            let maxd = x
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(maxd < 1e-4, "n={n} round-trip max diff {maxd:e}");
        }
    }

    // The half-spectrum irfft must equal a full-size complex IFFT of the
    // Hermitian extension.
    #[test]
    fn irfft_matches_full_complex_ifft() {
        let n = 256;
        let (c, s, hc, hs) = tables(n);
        let mut seed = 7u32;
        let mut g = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16_777_216.0 - 0.5
        };
        let br: Vec<f32> = (0..=n / 2).map(|_| g()).collect();
        let mut bi: Vec<f32> = (0..=n / 2).map(|_| g()).collect();
        bi[0] = 0.0;
        bi[n / 2] = 0.0; // Hermitian: DC and Nyquist are real
        // reference: build the full n-point Hermitian spectrum and inverse-FFT it
        let (mut fr, mut fi) = (vec![0.0f32; n], vec![0.0f32; n]);
        fr[..=n / 2].copy_from_slice(&br);
        fi[..=n / 2].copy_from_slice(&bi);
        for k in 1..n / 2 {
            fr[n - k] = br[k];
            fi[n - k] = -bi[k];
        }
        fft_inplace(&mut fr, &mut fi, true, &c, &s);
        // our half-spectrum irfft
        let mut out = vec![0.0f32; n];
        let (mut sre, mut sim) = (vec![0.0f32; n], vec![0.0f32; n]);
        irfft(&br, &bi, n, &mut out, &mut sre, &mut sim, &c, &s, &hc, &hs);
        let maxd = out
            .iter()
            .zip(&fr)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxd < 1e-4, "irfft vs full IFFT max diff {maxd:e}");
    }
}
