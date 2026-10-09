//! Where each tensor sits in the weight blob, in the order
//! `deepfilternet3/scripts/quantize_int8.py` wrote them. Names follow the upstream
//! ONNX tensors.
#![allow(non_snake_case)]

pub(crate) struct W(pub(crate) dfn3_plugin::Tensors);

impl W {
    #[inline]
    pub(crate) fn enc_erb_conv1_dw_w(&self) -> &[f32] {
        self.0.f(0, 192)
    }
    #[inline]
    pub(crate) fn enc_erb_conv2_dw_w(&self) -> &[f32] {
        self.0.f(192, 192)
    }
    #[inline]
    pub(crate) fn enc_erb_conv3_dw_w(&self) -> &[f32] {
        self.0.f(384, 192)
    }
    #[inline]
    pub(crate) fn enc_df_conv0_dw_w(&self) -> &[f32] {
        self.0.f(576, 576)
    }
    #[inline]
    pub(crate) fn enc_df_conv1_dw_w(&self) -> &[f32] {
        self.0.f(1152, 192)
    }
    #[inline]
    pub(crate) fn enc_df_fc_emb_w(&self) -> &[f32] {
        self.0.f(1344, 49152)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_lin_in_w(&self) -> &[f32] {
        self.0.f(50496, 8192)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_lin_out_w(&self) -> &[f32] {
        self.0.f(58688, 8192)
    }
    #[inline]
    pub(crate) fn enc_lsnr_fc_b(&self) -> &[f32] {
        self.0.f(66880, 1)
    }
    #[inline]
    pub(crate) fn enc_erb_conv0_dw_w(&self) -> &[f32] {
        self.0.f(66881, 576)
    }
    #[inline]
    pub(crate) fn enc_erb_conv0_dw_b(&self) -> &[f32] {
        self.0.f(67457, 64)
    }
    #[inline]
    pub(crate) fn enc_erb_conv1_pw_w(&self) -> &[f32] {
        self.0.f(67521, 4096)
    }
    #[inline]
    pub(crate) fn enc_erb_conv1_pw_b(&self) -> &[f32] {
        self.0.f(71617, 64)
    }
    #[inline]
    pub(crate) fn enc_erb_conv2_pw_w(&self) -> &[f32] {
        self.0.f(71681, 4096)
    }
    #[inline]
    pub(crate) fn enc_erb_conv2_pw_b(&self) -> &[f32] {
        self.0.f(75777, 64)
    }
    #[inline]
    pub(crate) fn enc_erb_conv3_pw_w(&self) -> &[f32] {
        self.0.f(75841, 4096)
    }
    #[inline]
    pub(crate) fn enc_erb_conv3_pw_b(&self) -> &[f32] {
        self.0.f(79937, 64)
    }
    #[inline]
    pub(crate) fn enc_df_conv0_pw_w(&self) -> &[f32] {
        self.0.f(80001, 4096)
    }
    #[inline]
    pub(crate) fn enc_df_conv0_pw_b(&self) -> &[f32] {
        self.0.f(84097, 64)
    }
    #[inline]
    pub(crate) fn enc_df_conv1_pw_w(&self) -> &[f32] {
        self.0.f(84161, 4096)
    }
    #[inline]
    pub(crate) fn enc_df_conv1_pw_b(&self) -> &[f32] {
        self.0.f(88257, 64)
    }
    #[inline]
    pub(crate) fn enc_lsnr_fc_w(&self) -> &[f32] {
        self.0.f(88321, 512)
    }
    #[inline]
    pub(crate) fn enc_emb_gru0_B(&self) -> &[f32] {
        self.0.f(88833, 1536)
    }
    #[inline]
    pub(crate) fn erb_emb_gru_lin_in_w(&self) -> &[f32] {
        self.0.f(90369, 8192)
    }
    #[inline]
    pub(crate) fn erb_emb_gru_lin_out_w(&self) -> &[f32] {
        self.0.f(98561, 8192)
    }
    #[inline]
    pub(crate) fn erb_convt3_dw_w(&self) -> &[f32] {
        self.0.f(106753, 192)
    }
    #[inline]
    pub(crate) fn erb_convt2_dw_w(&self) -> &[f32] {
        self.0.f(106945, 192)
    }
    #[inline]
    pub(crate) fn erb_convt1_dw_w(&self) -> &[f32] {
        self.0.f(107137, 192)
    }
    #[inline]
    pub(crate) fn erb_conv3p_dw_w(&self) -> &[f32] {
        self.0.f(107329, 64)
    }
    #[inline]
    pub(crate) fn erb_conv3p_dw_b(&self) -> &[f32] {
        self.0.f(107393, 64)
    }
    #[inline]
    pub(crate) fn erb_convt3_pw_w(&self) -> &[f32] {
        self.0.f(107457, 4096)
    }
    #[inline]
    pub(crate) fn erb_convt3_pw_b(&self) -> &[f32] {
        self.0.f(111553, 64)
    }
    #[inline]
    pub(crate) fn erb_conv2p_dw_w(&self) -> &[f32] {
        self.0.f(111617, 64)
    }
    #[inline]
    pub(crate) fn erb_conv2p_dw_b(&self) -> &[f32] {
        self.0.f(111681, 64)
    }
    #[inline]
    pub(crate) fn erb_convt2_pw_w(&self) -> &[f32] {
        self.0.f(111745, 4096)
    }
    #[inline]
    pub(crate) fn erb_convt2_pw_b(&self) -> &[f32] {
        self.0.f(115841, 64)
    }
    #[inline]
    pub(crate) fn erb_conv1p_dw_w(&self) -> &[f32] {
        self.0.f(115905, 64)
    }
    #[inline]
    pub(crate) fn erb_conv1p_dw_b(&self) -> &[f32] {
        self.0.f(115969, 64)
    }
    #[inline]
    pub(crate) fn erb_convt1_pw_w(&self) -> &[f32] {
        self.0.f(116033, 4096)
    }
    #[inline]
    pub(crate) fn erb_convt1_pw_b(&self) -> &[f32] {
        self.0.f(120129, 64)
    }
    #[inline]
    pub(crate) fn erb_conv0p_dw_w(&self) -> &[f32] {
        self.0.f(120193, 64)
    }
    #[inline]
    pub(crate) fn erb_conv0p_dw_b(&self) -> &[f32] {
        self.0.f(120257, 64)
    }
    #[inline]
    pub(crate) fn erb_conv0_out_w(&self) -> &[f32] {
        self.0.f(120321, 192)
    }
    #[inline]
    pub(crate) fn erb_conv0_out_b(&self) -> &[f32] {
        self.0.f(120513, 1)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_B(&self) -> &[f32] {
        self.0.f(120514, 1536)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_B(&self) -> &[f32] {
        self.0.f(122050, 1536)
    }
    #[inline]
    pub(crate) fn df_convp_dw_w(&self) -> &[f32] {
        self.0.f(123586, 1600)
    }
    #[inline]
    pub(crate) fn df_gru_lin_in_w(&self) -> &[f32] {
        self.0.f(125186, 16384)
    }
    #[inline]
    pub(crate) fn df_skip_w(&self) -> &[f32] {
        self.0.f(141570, 8192)
    }
    #[inline]
    pub(crate) fn df_out_w(&self) -> &[f32] {
        self.0.f(149762, 15360)
    }
    #[inline]
    pub(crate) fn df_convp_pw_w(&self) -> &[f32] {
        self.0.f(165123, 100)
    }
    #[inline]
    pub(crate) fn df_convp_pw_b(&self) -> &[f32] {
        self.0.f(165223, 10)
    }
    #[inline]
    pub(crate) fn df_gru0_B(&self) -> &[f32] {
        self.0.f(165233, 1536)
    }
    #[inline]
    pub(crate) fn df_gru1_B(&self) -> &[f32] {
        self.0.f(166769, 1536)
    }
    #[inline]
    pub(crate) fn enc_emb_gru0_W_q(&self) -> &[i8] {
        self.0.i(0, 196608)
    }
    #[inline]
    pub(crate) fn enc_emb_gru0_W_s(&self) -> &[f32] {
        self.0.sc(0, 768)
    }
    #[inline]
    pub(crate) fn enc_emb_gru0_R_q(&self) -> &[i8] {
        self.0.i(196608, 196608)
    }
    #[inline]
    pub(crate) fn enc_emb_gru0_R_s(&self) -> &[f32] {
        self.0.sc(768, 768)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_W_q(&self) -> &[i8] {
        self.0.i(393216, 196608)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_W_s(&self) -> &[f32] {
        self.0.sc(1536, 768)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_R_q(&self) -> &[i8] {
        self.0.i(589824, 196608)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_R_s(&self) -> &[f32] {
        self.0.sc(2304, 768)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_W_q(&self) -> &[i8] {
        self.0.i(786432, 196608)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_W_s(&self) -> &[f32] {
        self.0.sc(3072, 768)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_R_q(&self) -> &[i8] {
        self.0.i(983040, 196608)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_R_s(&self) -> &[f32] {
        self.0.sc(3840, 768)
    }
    #[inline]
    pub(crate) fn df_gru0_W_q(&self) -> &[i8] {
        self.0.i(1179648, 196608)
    }
    #[inline]
    pub(crate) fn df_gru0_W_s(&self) -> &[f32] {
        self.0.sc(4608, 768)
    }
    #[inline]
    pub(crate) fn df_gru0_R_q(&self) -> &[i8] {
        self.0.i(1376256, 196608)
    }
    #[inline]
    pub(crate) fn df_gru0_R_s(&self) -> &[f32] {
        self.0.sc(5376, 768)
    }
    #[inline]
    pub(crate) fn df_gru1_W_q(&self) -> &[i8] {
        self.0.i(1572864, 196608)
    }
    #[inline]
    pub(crate) fn df_gru1_W_s(&self) -> &[f32] {
        self.0.sc(6144, 768)
    }
    #[inline]
    pub(crate) fn df_gru1_R_q(&self) -> &[i8] {
        self.0.i(1769472, 196608)
    }
    #[inline]
    pub(crate) fn df_gru1_R_s(&self) -> &[f32] {
        self.0.sc(6912, 768)
    }
}
