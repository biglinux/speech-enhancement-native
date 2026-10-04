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
        let g = dfn_ops::pack_format::Geometry::DFN3LL;
        let section = dfn_ops::pack_format::sections(bytes, g).expect("invalid weights");
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
        if !section.packed {
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
        let g = dfn_ops::pack_format::Geometry::DFN3LL;
        let section = dfn_ops::pack_format::sections(bytes, g).expect("invalid embedded weights");
        assert!(section.packed, "build.rs embeds packed weights");
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
    pub fn enc_erb_conv0_dw_w(&self) -> &[f32] {
        self.f(0, 576)
    }
    #[inline]
    pub fn enc_erb_conv0_dw_b(&self) -> &[f32] {
        self.f(576, 64)
    }
    #[inline]
    pub fn enc_erb_conv1_dw_w(&self) -> &[f32] {
        self.f(640, 384)
    }
    #[inline]
    pub fn enc_erb_conv1_pw_w(&self) -> &[f32] {
        self.f(1024, 4096)
    }
    #[inline]
    pub fn enc_erb_conv1_pw_b(&self) -> &[f32] {
        self.f(5120, 64)
    }
    #[inline]
    pub fn enc_erb_conv2_dw_w(&self) -> &[f32] {
        self.f(5184, 384)
    }
    #[inline]
    pub fn enc_erb_conv2_pw_w(&self) -> &[f32] {
        self.f(5568, 4096)
    }
    #[inline]
    pub fn enc_erb_conv2_pw_b(&self) -> &[f32] {
        self.f(9664, 64)
    }
    #[inline]
    pub fn enc_erb_conv3_dw_w(&self) -> &[f32] {
        self.f(9728, 384)
    }
    #[inline]
    pub fn enc_erb_conv3_pw_w(&self) -> &[f32] {
        self.f(10112, 4096)
    }
    #[inline]
    pub fn enc_erb_conv3_pw_b(&self) -> &[f32] {
        self.f(14208, 64)
    }
    #[inline]
    pub fn enc_df_conv0_dw_w(&self) -> &[f32] {
        self.f(14272, 576)
    }
    #[inline]
    pub fn enc_df_conv0_pw_w(&self) -> &[f32] {
        self.f(14848, 4096)
    }
    #[inline]
    pub fn enc_df_conv0_pw_b(&self) -> &[f32] {
        self.f(18944, 64)
    }
    #[inline]
    pub fn enc_df_conv1_dw_w(&self) -> &[f32] {
        self.f(19008, 384)
    }
    #[inline]
    pub fn enc_df_conv1_pw_w(&self) -> &[f32] {
        self.f(19392, 4096)
    }
    #[inline]
    pub fn enc_df_conv1_pw_b(&self) -> &[f32] {
        self.f(23488, 64)
    }
    #[inline]
    pub fn enc_df_fc_emb_w(&self) -> &[f32] {
        self.f(23552, 98304)
    }
    #[inline]
    pub fn enc_emb_gru_lin_in_w(&self) -> &[f32] {
        self.f(121856, 16384)
    }
    #[inline]
    pub fn enc_emb_gru_B(&self) -> &[f32] {
        self.f(138240, 3072)
    }
    #[inline]
    pub fn enc_emb_gru_lin_out_w(&self) -> &[f32] {
        self.f(141312, 16384)
    }
    #[inline]
    pub fn enc_lsnr_fc_w(&self) -> &[f32] {
        self.f(157696, 512)
    }
    #[inline]
    pub fn enc_lsnr_fc_b(&self) -> &[f32] {
        self.f(158208, 1)
    }
    #[inline]
    pub fn erb_emb_gru_lin_in_w(&self) -> &[f32] {
        self.f(158209, 16384)
    }
    #[inline]
    pub fn erb_emb_gru0_B(&self) -> &[f32] {
        self.f(174593, 3072)
    }
    #[inline]
    pub fn erb_emb_gru1_B(&self) -> &[f32] {
        self.f(177665, 3072)
    }
    #[inline]
    pub fn erb_emb_gru_lin_out_w(&self) -> &[f32] {
        self.f(180737, 16384)
    }
    #[inline]
    pub fn erb_conv3p_dw_w(&self) -> &[f32] {
        self.f(197121, 64)
    }
    #[inline]
    pub fn erb_conv3p_dw_b(&self) -> &[f32] {
        self.f(197185, 64)
    }
    #[inline]
    pub fn erb_convt3_dw_w(&self) -> &[f32] {
        self.f(197249, 384)
    }
    #[inline]
    pub fn erb_convt3_pw_w(&self) -> &[f32] {
        self.f(197633, 4096)
    }
    #[inline]
    pub fn erb_convt3_pw_b(&self) -> &[f32] {
        self.f(201729, 64)
    }
    #[inline]
    pub fn erb_conv2p_dw_w(&self) -> &[f32] {
        self.f(201793, 64)
    }
    #[inline]
    pub fn erb_conv2p_dw_b(&self) -> &[f32] {
        self.f(201857, 64)
    }
    #[inline]
    pub fn erb_convt2_pw_w(&self) -> &[f32] {
        self.f(201921, 4096)
    }
    #[inline]
    pub fn erb_convt2_pw_b(&self) -> &[f32] {
        self.f(206017, 64)
    }
    #[inline]
    pub fn erb_conv1p_dw_w(&self) -> &[f32] {
        self.f(206081, 64)
    }
    #[inline]
    pub fn erb_conv1p_dw_b(&self) -> &[f32] {
        self.f(206145, 64)
    }
    #[inline]
    pub fn erb_convt1_pw_w(&self) -> &[f32] {
        self.f(206209, 4096)
    }
    #[inline]
    pub fn erb_convt1_pw_b(&self) -> &[f32] {
        self.f(210305, 64)
    }
    #[inline]
    pub fn erb_conv0p_dw_w(&self) -> &[f32] {
        self.f(210369, 64)
    }
    #[inline]
    pub fn erb_conv0p_dw_b(&self) -> &[f32] {
        self.f(210433, 64)
    }
    #[inline]
    pub fn erb_conv0_out_w(&self) -> &[f32] {
        self.f(210497, 384)
    }
    #[inline]
    pub fn erb_conv0_out_b(&self) -> &[f32] {
        self.f(210881, 1)
    }
    #[inline]
    pub fn erb_convt2_dw_w(&self) -> &[f32] {
        self.f(210882, 192)
    }
    #[inline]
    pub fn erb_convt1_dw_w(&self) -> &[f32] {
        self.f(211074, 192)
    }
    #[inline]
    pub fn df_gru_lin_in_w(&self) -> &[f32] {
        self.f(211266, 32768)
    }
    #[inline]
    pub fn df_gru0_B(&self) -> &[f32] {
        self.f(244034, 3072)
    }
    #[inline]
    pub fn df_gru1_B(&self) -> &[f32] {
        self.f(247106, 3072)
    }
    #[inline]
    pub fn df_gru2_B(&self) -> &[f32] {
        self.f(250178, 3072)
    }
    #[inline]
    pub fn df_skip_w(&self) -> &[f32] {
        self.f(253250, 16384)
    }
    #[inline]
    pub fn df_out_w(&self) -> &[f32] {
        self.f(269634, 30720)
    }
    #[inline]
    pub fn df_convp_dw_w(&self) -> &[f32] {
        self.f(300354, 1600)
    }
    #[inline]
    pub fn df_convp_pw_w(&self) -> &[f32] {
        self.f(301954, 100)
    }
    #[inline]
    pub fn df_convp_pw_b(&self) -> &[f32] {
        self.f(302054, 10)
    }
    #[inline]
    pub fn enc_emb_gru_W_q(&self) -> &[i8] {
        self.i(0, 786432)
    }
    #[inline]
    pub fn enc_emb_gru_W_s(&self) -> &[f32] {
        self.sc(0, 1536)
    }
    #[inline]
    pub fn enc_emb_gru_R_q(&self) -> &[i8] {
        self.i(786432, 786432)
    }
    #[inline]
    pub fn enc_emb_gru_R_s(&self) -> &[f32] {
        self.sc(1536, 1536)
    }
    #[inline]
    pub fn erb_emb_gru0_W_q(&self) -> &[i8] {
        self.i(1572864, 786432)
    }
    #[inline]
    pub fn erb_emb_gru0_W_s(&self) -> &[f32] {
        self.sc(3072, 1536)
    }
    #[inline]
    pub fn erb_emb_gru0_R_q(&self) -> &[i8] {
        self.i(2359296, 786432)
    }
    #[inline]
    pub fn erb_emb_gru0_R_s(&self) -> &[f32] {
        self.sc(4608, 1536)
    }
    #[inline]
    pub fn erb_emb_gru1_W_q(&self) -> &[i8] {
        self.i(3145728, 786432)
    }
    #[inline]
    pub fn erb_emb_gru1_W_s(&self) -> &[f32] {
        self.sc(6144, 1536)
    }
    #[inline]
    pub fn erb_emb_gru1_R_q(&self) -> &[i8] {
        self.i(3932160, 786432)
    }
    #[inline]
    pub fn erb_emb_gru1_R_s(&self) -> &[f32] {
        self.sc(7680, 1536)
    }
    #[inline]
    pub fn df_gru0_W_q(&self) -> &[i8] {
        self.i(4718592, 786432)
    }
    #[inline]
    pub fn df_gru0_W_s(&self) -> &[f32] {
        self.sc(9216, 1536)
    }
    #[inline]
    pub fn df_gru0_R_q(&self) -> &[i8] {
        self.i(5505024, 786432)
    }
    #[inline]
    pub fn df_gru0_R_s(&self) -> &[f32] {
        self.sc(10752, 1536)
    }
    #[inline]
    pub fn df_gru1_W_q(&self) -> &[i8] {
        self.i(6291456, 786432)
    }
    #[inline]
    pub fn df_gru1_W_s(&self) -> &[f32] {
        self.sc(12288, 1536)
    }
    #[inline]
    pub fn df_gru1_R_q(&self) -> &[i8] {
        self.i(7077888, 786432)
    }
    #[inline]
    pub fn df_gru1_R_s(&self) -> &[f32] {
        self.sc(13824, 1536)
    }
    #[inline]
    pub fn df_gru2_W_q(&self) -> &[i8] {
        self.i(7864320, 786432)
    }
    #[inline]
    pub fn df_gru2_W_s(&self) -> &[f32] {
        self.sc(15360, 1536)
    }
    #[inline]
    pub fn df_gru2_R_q(&self) -> &[i8] {
        self.i(8650752, 786432)
    }
    #[inline]
    pub fn df_gru2_R_s(&self) -> &[f32] {
        self.sc(16896, 1536)
    }
}
