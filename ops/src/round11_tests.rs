#[cfg(any(feature = "r11-packed", feature = "r11-matvec"))]
fn vectors(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 71 % 107) as f32 - 53.0) / 53.0)
        .collect()
}
#[cfg(feature = "r11-quant")]
#[test]
fn quantizer_preserves_legacy_extremes_ties_and_tails() {
    let mut data = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-39,
        -1e-39,
        16383.0,
        -16383.0,
    ];
    for i in -16383..16383 {
        let x = i as f32 + 0.5;
        data.extend_from_slice(&[
            x,
            f32::from_bits(x.to_bits().wrapping_add(1)),
            f32::from_bits(x.to_bits().wrapping_sub(1)),
        ]);
    }
    for offset in 0..4 {
        for tail in 0..17 {
            let x = &data[offset..data.len() - tail];
            let mut a = vec![11; x.len() + 2];
            let mut b = a.clone();
            let sa = super::quantize_i16_reference(x, &mut a[1..1 + x.len()]);
            let sb = super::quantize_i16(x, &mut b[1..1 + x.len()]);
            assert_eq!(sa.to_bits(), sb.to_bits());
            assert_eq!(a, b);
        }
    }
    for x in [
        vec![f32::NAN, 1.0],
        vec![f32::INFINITY, -1.0],
        vec![f32::NEG_INFINITY],
        vec![1e-40, -1e-40, 0.0],
        vec![],
        vec![-0.0; 512],
    ] {
        let mut a = vec![0; x.len()];
        let mut b = a.clone();
        let sa = super::quantize_i16_reference(&x, &mut a);
        let sb = super::quantize_i16(&x, &mut b);
        assert_eq!(sa.to_bits(), sb.to_bits());
        assert_eq!(a, b);
    }
}
#[cfg(feature = "r11-packed")]
#[test]
fn packed_products_equal_int128_reference() {
    for n in [2, 6, 64, 256, 512, 1024] {
        for m in [8, 24, 32, 64, 72, 192, 768, 1536] {
            let w: Vec<i8> = (0..m * n).map(|i| ((i * 79 % 256) as u8) as i8).collect();
            let bytes: Vec<u8> = w.iter().map(|&x| x as u8).collect();
            let packed = super::pack_format::pack_matrix(&bytes, m, n).unwrap();
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
                super::packed::matvec(&mut y, &packed, &scales, &q, 0.004, m, n);
                for r in 0..m {
                    let v: i128 = (0..n)
                        .map(|j| i128::from(w[r * n + j]) * i128::from(q[j]))
                        .sum();
                    let v = (v as f32 * scales[r]) * 0.004;
                    assert_eq!(v.to_bits(), y[r].to_bits(), "m={m} n={n} row={r}");
                }
            }
        }
    }
}
#[cfg(feature = "r11-packed")]
#[test]
fn packed_gru_matches_row_major_same_tier() {
    for h in [64, 256, 512] {
        let m = 3 * h;
        let w: Vec<i8> = (0..m * h)
            .map(|i| ((i * 13 % 255) as i16 - 127) as i8)
            .collect();
        let p: Vec<i8> =
            super::pack_format::pack_matrix(&w.iter().map(|&x| x as u8).collect::<Vec<_>>(), m, h)
                .unwrap()
                .into_iter()
                .map(|x| x as i8)
                .collect();
        let s = vec![0.002; m];
        let b = vectors(6 * h);
        let mut old = vectors(h);
        let mut new = old.clone();
        let mut qa = vec![0; h];
        let mut qb = qa.clone();
        let mut a = vec![0.0; 6 * h];
        let mut c = a.clone();
        let x = vectors(h);
        for _ in 0..12 {
            super::gru_cell_q(&mut old, &x, &w, &s, &w, &s, &b, h, h, &mut a, &mut qa);
            super::gru_cell_packed(&mut new, &x, &p, &s, &p, &s, &b, h, h, &mut c, &mut qb);
            assert!(old
                .iter()
                .zip(&new)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}
#[cfg(all(feature = "r11-matvec", target_arch = "x86_64"))]
#[test]
fn matvec_reblocking_same_tier_bits() {
    for m in [1, 7, 16, 32, 64, 96, 256, 512] {
        for n in [1, 5, 8, 16, 32, 60, 64, 96, 512] {
            let x = vectors(m);
            let w = vectors(m * n);
            let mut a = vec![0.0; n];
            let mut b = a.clone();
            match super::simd_tier() {
                3 => unsafe { super::matvec_t_avx2(&mut a, &w, &x, m, n) },
                2 => unsafe { super::matvec_t_avx(&mut a, &w, &x, m, n) },
                _ => {
                    for i in 0..m {
                        for j in 0..n {
                            a[j] += x[i] * w[i * n + j];
                        }
                    }
                }
            }
            super::matvec_t(&mut b, &w, &x, m, n);
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}
#[test]
fn pack_format_roundtrip_and_header_validation() {
    for g in [
        super::pack_format::Geometry::DFN3,
        super::pack_format::Geometry::DFN3LL,
    ] {
        let raw: Vec<u8> = (0..g.raw_len()).map(|i| (i * 79 % 256) as u8).collect();
        let packed = super::pack_format::convert(&raw, g).unwrap();
        let s = super::pack_format::sections(&packed, g).unwrap();
        let old = super::pack_format::sections(&raw, g).unwrap();
        assert_eq!(&packed[s.f], &raw[old.f]);
        assert_eq!(&packed[s.s], &raw[old.s]);
        for m in 0..g.matrices() {
            let k = m * g.matrix_len();
            let u = super::pack_format::unpack_matrix(
                &packed[s.i.start + k..s.i.start + k + g.matrix_len()],
                3 * g.hidden,
                g.hidden,
            )
            .unwrap();
            assert_eq!(u, raw[old.i.start + k..old.i.start + k + g.matrix_len()]);
        }
        let mut broken = packed.clone();
        broken[16] ^= 1;
        assert!(super::pack_format::sections(&broken, g).is_err());
        assert!(super::pack_format::sections(&packed[..packed.len() - 1], g).is_err());
    }
}
