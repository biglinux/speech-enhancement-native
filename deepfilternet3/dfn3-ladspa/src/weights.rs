// Auto-generated hybrid weight table: non-GRU tensors f32, GRU W/R int8 + per-row scale.
#![allow(dead_code, non_snake_case)]
use std::borrow::Cow;
pub struct W {
    f32buf: Cow<'static, [f32]>,
    i8buf: Cow<'static, [i8]>,
    scbuf: Cow<'static, [f32]>,
}
impl W {
    /// Owning decode (copies into Vecs); used by the CLIs and file-loading tests
    /// where a second copy costs nothing.
    pub fn load(bytes: &[u8]) -> Self {
        let g = dfn_ops::pack_format::Geometry::DFN3;
        let section = dfn_ops::pack_format::sections(bytes, g).expect("invalid weights");
        assert!(
            !section.packed || cfg!(feature = "r11-packed"),
            "DFNPAIR1 requires r11-packed"
        );
        let f32buf = bytes[section.f]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let scbuf = bytes[section.s]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let mut integers = bytes[section.i].to_vec();
        if cfg!(feature = "r11-packed") && !section.packed {
            for m in 0..g.matrices() {
                let k = m * g.matrix_len();
                let p = dfn_ops::pack_format::pack_matrix(
                    &integers[k..k + g.matrix_len()],
                    3 * g.hidden,
                    g.hidden,
                )
                .unwrap();
                integers[k..k + p.len()].copy_from_slice(&p);
            }
        }
        Self {
            f32buf: Cow::Owned(f32buf),
            i8buf: Cow::Owned(integers.into_iter().map(|b| b as i8).collect()),
            scbuf: Cow::Owned(scbuf),
        }
    }
    /// Embedded blob is immutably borrowed; packing was done by build.rs.
    /// Header and section offsets are checked before constructing typed views.
    pub fn from_static_aligned(bytes: &'static [u8]) -> Self {
        const {
            assert!(
                cfg!(target_endian = "little"),
                "embedded views require little endian"
            )
        };
        assert_eq!(bytes.as_ptr() as usize % 4, 0, "embedded alignment");
        let g = dfn_ops::pack_format::Geometry::DFN3;
        let section = dfn_ops::pack_format::sections(bytes, g).expect("invalid embedded weights");
        assert_eq!(
            section.packed,
            cfg!(feature = "r11-packed"),
            "embedded layout/feature mismatch"
        );
        // SAFETY: canonical ranges validated above, aligned f32 sections, static owner.
        let f32buf = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(section.f.start).cast::<f32>(), g.floats)
        };
        let i8buf = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(section.i.start).cast::<i8>(), g.integers)
        };
        let scbuf = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(section.s.start).cast::<f32>(), g.scales)
        };
        Self {
            f32buf: Cow::Borrowed(f32buf),
            i8buf: Cow::Borrowed(i8buf),
            scbuf: Cow::Borrowed(scbuf),
        }
    }
    /// Pin the weight buffers in RAM (best effort) so realtime processing never
    /// faults on them under memory pressure.
    pub fn mlock(&self) {
        dfn_ops::mlock_slice(&self.f32buf);
        dfn_ops::mlock_slice(&self.i8buf);
        dfn_ops::mlock_slice(&self.scbuf);
    }
    #[inline]
    fn f(&self, o: usize, n: usize) -> &[f32] {
        &self.f32buf[o..o + n]
    }
    #[inline]
    fn i(&self, o: usize, n: usize) -> &[i8] {
        &self.i8buf[o..o + n]
    }
    #[inline]
    fn sc(&self, o: usize, n: usize) -> &[f32] {
        &self.scbuf[o..o + n]
    }
}
impl W {
    #[inline]
    pub fn enc_erb_conv1_dw_w(&self) -> &[f32] {
        self.f(0, 192)
    }
    #[inline]
    pub fn enc_erb_conv2_dw_w(&self) -> &[f32] {
        self.f(192, 192)
    }
    #[inline]
    pub fn enc_erb_conv3_dw_w(&self) -> &[f32] {
        self.f(384, 192)
    }
    #[inline]
    pub fn enc_df_conv0_dw_w(&self) -> &[f32] {
        self.f(576, 576)
    }
    #[inline]
    pub fn enc_df_conv1_dw_w(&self) -> &[f32] {
        self.f(1152, 192)
    }
    #[inline]
    pub fn enc_df_fc_emb_w(&self) -> &[f32] {
        self.f(1344, 49152)
    }
    #[inline]
    pub fn enc_emb_gru_lin_in_w(&self) -> &[f32] {
        self.f(50496, 8192)
    }
    #[inline]
    pub fn enc_emb_gru_lin_out_w(&self) -> &[f32] {
        self.f(58688, 8192)
    }
    #[inline]
    pub fn enc_lsnr_fc_b(&self) -> &[f32] {
        self.f(66880, 1)
    }
    #[inline]
    pub fn enc_erb_conv0_dw_w(&self) -> &[f32] {
        self.f(66881, 576)
    }
    #[inline]
    pub fn enc_erb_conv0_dw_b(&self) -> &[f32] {
        self.f(67457, 64)
    }
    #[inline]
    pub fn enc_erb_conv1_pw_w(&self) -> &[f32] {
        self.f(67521, 4096)
    }
    #[inline]
    pub fn enc_erb_conv1_pw_b(&self) -> &[f32] {
        self.f(71617, 64)
    }
    #[inline]
    pub fn enc_erb_conv2_pw_w(&self) -> &[f32] {
        self.f(71681, 4096)
    }
    #[inline]
    pub fn enc_erb_conv2_pw_b(&self) -> &[f32] {
        self.f(75777, 64)
    }
    #[inline]
    pub fn enc_erb_conv3_pw_w(&self) -> &[f32] {
        self.f(75841, 4096)
    }
    #[inline]
    pub fn enc_erb_conv3_pw_b(&self) -> &[f32] {
        self.f(79937, 64)
    }
    #[inline]
    pub fn enc_df_conv0_pw_w(&self) -> &[f32] {
        self.f(80001, 4096)
    }
    #[inline]
    pub fn enc_df_conv0_pw_b(&self) -> &[f32] {
        self.f(84097, 64)
    }
    #[inline]
    pub fn enc_df_conv1_pw_w(&self) -> &[f32] {
        self.f(84161, 4096)
    }
    #[inline]
    pub fn enc_df_conv1_pw_b(&self) -> &[f32] {
        self.f(88257, 64)
    }
    #[inline]
    pub fn enc_lsnr_fc_w(&self) -> &[f32] {
        self.f(88321, 512)
    }
    #[inline]
    pub fn enc_emb_gru0_B(&self) -> &[f32] {
        self.f(88833, 1536)
    }
    #[inline]
    pub fn erb_emb_gru_lin_in_w(&self) -> &[f32] {
        self.f(90369, 8192)
    }
    #[inline]
    pub fn erb_emb_gru_lin_out_w(&self) -> &[f32] {
        self.f(98561, 8192)
    }
    #[inline]
    pub fn erb_convt3_dw_w(&self) -> &[f32] {
        self.f(106753, 192)
    }
    #[inline]
    pub fn erb_convt2_dw_w(&self) -> &[f32] {
        self.f(106945, 192)
    }
    #[inline]
    pub fn erb_convt1_dw_w(&self) -> &[f32] {
        self.f(107137, 192)
    }
    #[inline]
    pub fn erb_conv3p_dw_w(&self) -> &[f32] {
        self.f(107329, 64)
    }
    #[inline]
    pub fn erb_conv3p_dw_b(&self) -> &[f32] {
        self.f(107393, 64)
    }
    #[inline]
    pub fn erb_convt3_pw_w(&self) -> &[f32] {
        self.f(107457, 4096)
    }
    #[inline]
    pub fn erb_convt3_pw_b(&self) -> &[f32] {
        self.f(111553, 64)
    }
    #[inline]
    pub fn erb_conv2p_dw_w(&self) -> &[f32] {
        self.f(111617, 64)
    }
    #[inline]
    pub fn erb_conv2p_dw_b(&self) -> &[f32] {
        self.f(111681, 64)
    }
    #[inline]
    pub fn erb_convt2_pw_w(&self) -> &[f32] {
        self.f(111745, 4096)
    }
    #[inline]
    pub fn erb_convt2_pw_b(&self) -> &[f32] {
        self.f(115841, 64)
    }
    #[inline]
    pub fn erb_conv1p_dw_w(&self) -> &[f32] {
        self.f(115905, 64)
    }
    #[inline]
    pub fn erb_conv1p_dw_b(&self) -> &[f32] {
        self.f(115969, 64)
    }
    #[inline]
    pub fn erb_convt1_pw_w(&self) -> &[f32] {
        self.f(116033, 4096)
    }
    #[inline]
    pub fn erb_convt1_pw_b(&self) -> &[f32] {
        self.f(120129, 64)
    }
    #[inline]
    pub fn erb_conv0p_dw_w(&self) -> &[f32] {
        self.f(120193, 64)
    }
    #[inline]
    pub fn erb_conv0p_dw_b(&self) -> &[f32] {
        self.f(120257, 64)
    }
    #[inline]
    pub fn erb_conv0_out_w(&self) -> &[f32] {
        self.f(120321, 192)
    }
    #[inline]
    pub fn erb_conv0_out_b(&self) -> &[f32] {
        self.f(120513, 1)
    }
    #[inline]
    pub fn erb_emb_gru0_B(&self) -> &[f32] {
        self.f(120514, 1536)
    }
    #[inline]
    pub fn erb_emb_gru1_B(&self) -> &[f32] {
        self.f(122050, 1536)
    }
    #[inline]
    pub fn df_convp_dw_w(&self) -> &[f32] {
        self.f(123586, 1600)
    }
    #[inline]
    pub fn df_gru_lin_in_w(&self) -> &[f32] {
        self.f(125186, 16384)
    }
    #[inline]
    pub fn df_skip_w(&self) -> &[f32] {
        self.f(141570, 8192)
    }
    #[inline]
    pub fn df_out_w(&self) -> &[f32] {
        self.f(149762, 15360)
    }
    #[inline]
    pub fn df_fc_a_b(&self) -> &[f32] {
        self.f(165122, 1)
    }
    #[inline]
    pub fn df_convp_pw_w(&self) -> &[f32] {
        self.f(165123, 100)
    }
    #[inline]
    pub fn df_convp_pw_b(&self) -> &[f32] {
        self.f(165223, 10)
    }
    #[inline]
    pub fn df_gru0_B(&self) -> &[f32] {
        self.f(165233, 1536)
    }
    #[inline]
    pub fn df_gru1_B(&self) -> &[f32] {
        self.f(166769, 1536)
    }
    #[inline]
    pub fn df_fc_a_w(&self) -> &[f32] {
        self.f(168305, 256)
    }
    #[inline]
    pub fn enc_emb_gru0_W_q(&self) -> &[i8] {
        self.i(0, 196608)
    }
    #[inline]
    pub fn enc_emb_gru0_W_s(&self) -> &[f32] {
        self.sc(0, 768)
    }
    #[inline]
    pub fn enc_emb_gru0_R_q(&self) -> &[i8] {
        self.i(196608, 196608)
    }
    #[inline]
    pub fn enc_emb_gru0_R_s(&self) -> &[f32] {
        self.sc(768, 768)
    }
    #[inline]
    pub fn erb_emb_gru0_W_q(&self) -> &[i8] {
        self.i(393216, 196608)
    }
    #[inline]
    pub fn erb_emb_gru0_W_s(&self) -> &[f32] {
        self.sc(1536, 768)
    }
    #[inline]
    pub fn erb_emb_gru0_R_q(&self) -> &[i8] {
        self.i(589824, 196608)
    }
    #[inline]
    pub fn erb_emb_gru0_R_s(&self) -> &[f32] {
        self.sc(2304, 768)
    }
    #[inline]
    pub fn erb_emb_gru1_W_q(&self) -> &[i8] {
        self.i(786432, 196608)
    }
    #[inline]
    pub fn erb_emb_gru1_W_s(&self) -> &[f32] {
        self.sc(3072, 768)
    }
    #[inline]
    pub fn erb_emb_gru1_R_q(&self) -> &[i8] {
        self.i(983040, 196608)
    }
    #[inline]
    pub fn erb_emb_gru1_R_s(&self) -> &[f32] {
        self.sc(3840, 768)
    }
    #[inline]
    pub fn df_gru0_W_q(&self) -> &[i8] {
        self.i(1179648, 196608)
    }
    #[inline]
    pub fn df_gru0_W_s(&self) -> &[f32] {
        self.sc(4608, 768)
    }
    #[inline]
    pub fn df_gru0_R_q(&self) -> &[i8] {
        self.i(1376256, 196608)
    }
    #[inline]
    pub fn df_gru0_R_s(&self) -> &[f32] {
        self.sc(5376, 768)
    }
    #[inline]
    pub fn df_gru1_W_q(&self) -> &[i8] {
        self.i(1572864, 196608)
    }
    #[inline]
    pub fn df_gru1_W_s(&self) -> &[f32] {
        self.sc(6144, 768)
    }
    #[inline]
    pub fn df_gru1_R_q(&self) -> &[i8] {
        self.i(1769472, 196608)
    }
    #[inline]
    pub fn df_gru1_R_s(&self) -> &[f32] {
        self.sc(6912, 768)
    }
}
