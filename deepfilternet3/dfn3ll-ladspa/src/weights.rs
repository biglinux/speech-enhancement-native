//! Where each tensor sits in the weight blob, in the order
//! `deepfilternet3/scripts/quantize_int8.py` wrote them. Names follow the upstream
//! ONNX tensors.
#![allow(non_snake_case)]

pub(crate) struct W(pub(crate) dfn3_plugin::Tensors);

impl W {
    #[inline]
    pub(crate) fn enc_erb_conv0_dw_w(&self) -> &[f32] {
        self.0.f(0, 576)
    }
    #[inline]
    pub(crate) fn enc_erb_conv0_dw_b(&self) -> &[f32] {
        self.0.f(576, 64)
    }
    #[inline]
    pub(crate) fn enc_erb_conv1_dw_w(&self) -> &[f32] {
        self.0.f(640, 384)
    }
    #[inline]
    pub(crate) fn enc_erb_conv1_pw_w(&self) -> &[f32] {
        self.0.f(1024, 4096)
    }
    #[inline]
    pub(crate) fn enc_erb_conv1_pw_b(&self) -> &[f32] {
        self.0.f(5120, 64)
    }
    #[inline]
    pub(crate) fn enc_erb_conv2_dw_w(&self) -> &[f32] {
        self.0.f(5184, 384)
    }
    #[inline]
    pub(crate) fn enc_erb_conv2_pw_w(&self) -> &[f32] {
        self.0.f(5568, 4096)
    }
    #[inline]
    pub(crate) fn enc_erb_conv2_pw_b(&self) -> &[f32] {
        self.0.f(9664, 64)
    }
    #[inline]
    pub(crate) fn enc_erb_conv3_dw_w(&self) -> &[f32] {
        self.0.f(9728, 384)
    }
    #[inline]
    pub(crate) fn enc_erb_conv3_pw_w(&self) -> &[f32] {
        self.0.f(10112, 4096)
    }
    #[inline]
    pub(crate) fn enc_erb_conv3_pw_b(&self) -> &[f32] {
        self.0.f(14208, 64)
    }
    #[inline]
    pub(crate) fn enc_df_conv0_dw_w(&self) -> &[f32] {
        self.0.f(14272, 576)
    }
    #[inline]
    pub(crate) fn enc_df_conv0_pw_w(&self) -> &[f32] {
        self.0.f(14848, 4096)
    }
    #[inline]
    pub(crate) fn enc_df_conv0_pw_b(&self) -> &[f32] {
        self.0.f(18944, 64)
    }
    #[inline]
    pub(crate) fn enc_df_conv1_dw_w(&self) -> &[f32] {
        self.0.f(19008, 384)
    }
    #[inline]
    pub(crate) fn enc_df_conv1_pw_w(&self) -> &[f32] {
        self.0.f(19392, 4096)
    }
    #[inline]
    pub(crate) fn enc_df_conv1_pw_b(&self) -> &[f32] {
        self.0.f(23488, 64)
    }
    #[inline]
    pub(crate) fn enc_df_fc_emb_w(&self) -> &[f32] {
        self.0.f(23552, 98304)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_lin_in_w(&self) -> &[f32] {
        self.0.f(121856, 16384)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_B(&self) -> &[f32] {
        self.0.f(138240, 3072)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_lin_out_w(&self) -> &[f32] {
        self.0.f(141312, 16384)
    }
    #[inline]
    pub(crate) fn enc_lsnr_fc_w(&self) -> &[f32] {
        self.0.f(157696, 512)
    }
    #[inline]
    pub(crate) fn enc_lsnr_fc_b(&self) -> &[f32] {
        self.0.f(158208, 1)
    }
    #[inline]
    pub(crate) fn erb_emb_gru_lin_in_w(&self) -> &[f32] {
        self.0.f(158209, 16384)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_B(&self) -> &[f32] {
        self.0.f(174593, 3072)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_B(&self) -> &[f32] {
        self.0.f(177665, 3072)
    }
    #[inline]
    pub(crate) fn erb_emb_gru_lin_out_w(&self) -> &[f32] {
        self.0.f(180737, 16384)
    }
    #[inline]
    pub(crate) fn erb_conv3p_dw_w(&self) -> &[f32] {
        self.0.f(197121, 64)
    }
    #[inline]
    pub(crate) fn erb_conv3p_dw_b(&self) -> &[f32] {
        self.0.f(197185, 64)
    }
    #[inline]
    pub(crate) fn erb_convt3_dw_w(&self) -> &[f32] {
        self.0.f(197249, 384)
    }
    #[inline]
    pub(crate) fn erb_convt3_pw_w(&self) -> &[f32] {
        self.0.f(197633, 4096)
    }
    #[inline]
    pub(crate) fn erb_convt3_pw_b(&self) -> &[f32] {
        self.0.f(201729, 64)
    }
    #[inline]
    pub(crate) fn erb_conv2p_dw_w(&self) -> &[f32] {
        self.0.f(201793, 64)
    }
    #[inline]
    pub(crate) fn erb_conv2p_dw_b(&self) -> &[f32] {
        self.0.f(201857, 64)
    }
    #[inline]
    pub(crate) fn erb_convt2_pw_w(&self) -> &[f32] {
        self.0.f(201921, 4096)
    }
    #[inline]
    pub(crate) fn erb_convt2_pw_b(&self) -> &[f32] {
        self.0.f(206017, 64)
    }
    #[inline]
    pub(crate) fn erb_conv1p_dw_w(&self) -> &[f32] {
        self.0.f(206081, 64)
    }
    #[inline]
    pub(crate) fn erb_conv1p_dw_b(&self) -> &[f32] {
        self.0.f(206145, 64)
    }
    #[inline]
    pub(crate) fn erb_convt1_pw_w(&self) -> &[f32] {
        self.0.f(206209, 4096)
    }
    #[inline]
    pub(crate) fn erb_convt1_pw_b(&self) -> &[f32] {
        self.0.f(210305, 64)
    }
    #[inline]
    pub(crate) fn erb_conv0p_dw_w(&self) -> &[f32] {
        self.0.f(210369, 64)
    }
    #[inline]
    pub(crate) fn erb_conv0p_dw_b(&self) -> &[f32] {
        self.0.f(210433, 64)
    }
    #[inline]
    pub(crate) fn erb_conv0_out_w(&self) -> &[f32] {
        self.0.f(210497, 384)
    }
    #[inline]
    pub(crate) fn erb_conv0_out_b(&self) -> &[f32] {
        self.0.f(210881, 1)
    }
    #[inline]
    pub(crate) fn erb_convt2_dw_w(&self) -> &[f32] {
        self.0.f(210882, 192)
    }
    #[inline]
    pub(crate) fn erb_convt1_dw_w(&self) -> &[f32] {
        self.0.f(211074, 192)
    }
    #[inline]
    pub(crate) fn df_gru_lin_in_w(&self) -> &[f32] {
        self.0.f(211266, 32768)
    }
    #[inline]
    pub(crate) fn df_gru0_B(&self) -> &[f32] {
        self.0.f(244034, 3072)
    }
    #[inline]
    pub(crate) fn df_gru1_B(&self) -> &[f32] {
        self.0.f(247106, 3072)
    }
    #[inline]
    pub(crate) fn df_gru2_B(&self) -> &[f32] {
        self.0.f(250178, 3072)
    }
    #[inline]
    pub(crate) fn df_skip_w(&self) -> &[f32] {
        self.0.f(253250, 16384)
    }
    #[inline]
    pub(crate) fn df_out_w(&self) -> &[f32] {
        self.0.f(269634, 30720)
    }
    #[inline]
    pub(crate) fn df_convp_dw_w(&self) -> &[f32] {
        self.0.f(300354, 1600)
    }
    #[inline]
    pub(crate) fn df_convp_pw_w(&self) -> &[f32] {
        self.0.f(301954, 100)
    }
    #[inline]
    pub(crate) fn df_convp_pw_b(&self) -> &[f32] {
        self.0.f(302054, 10)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_W_q(&self) -> &[i8] {
        self.0.i(0, 786432)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_W_s(&self) -> &[f32] {
        self.0.sc(0, 1536)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_R_q(&self) -> &[i8] {
        self.0.i(786432, 786432)
    }
    #[inline]
    pub(crate) fn enc_emb_gru_R_s(&self) -> &[f32] {
        self.0.sc(1536, 1536)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_W_q(&self) -> &[i8] {
        self.0.i(1572864, 786432)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_W_s(&self) -> &[f32] {
        self.0.sc(3072, 1536)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_R_q(&self) -> &[i8] {
        self.0.i(2359296, 786432)
    }
    #[inline]
    pub(crate) fn erb_emb_gru0_R_s(&self) -> &[f32] {
        self.0.sc(4608, 1536)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_W_q(&self) -> &[i8] {
        self.0.i(3145728, 786432)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_W_s(&self) -> &[f32] {
        self.0.sc(6144, 1536)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_R_q(&self) -> &[i8] {
        self.0.i(3932160, 786432)
    }
    #[inline]
    pub(crate) fn erb_emb_gru1_R_s(&self) -> &[f32] {
        self.0.sc(7680, 1536)
    }
    #[inline]
    pub(crate) fn df_gru0_W_q(&self) -> &[i8] {
        self.0.i(4718592, 786432)
    }
    #[inline]
    pub(crate) fn df_gru0_W_s(&self) -> &[f32] {
        self.0.sc(9216, 1536)
    }
    #[inline]
    pub(crate) fn df_gru0_R_q(&self) -> &[i8] {
        self.0.i(5505024, 786432)
    }
    #[inline]
    pub(crate) fn df_gru0_R_s(&self) -> &[f32] {
        self.0.sc(10752, 1536)
    }
    #[inline]
    pub(crate) fn df_gru1_W_q(&self) -> &[i8] {
        self.0.i(6291456, 786432)
    }
    #[inline]
    pub(crate) fn df_gru1_W_s(&self) -> &[f32] {
        self.0.sc(12288, 1536)
    }
    #[inline]
    pub(crate) fn df_gru1_R_q(&self) -> &[i8] {
        self.0.i(7077888, 786432)
    }
    #[inline]
    pub(crate) fn df_gru1_R_s(&self) -> &[f32] {
        self.0.sc(13824, 1536)
    }
    #[inline]
    pub(crate) fn df_gru2_W_q(&self) -> &[i8] {
        self.0.i(7864320, 786432)
    }
    #[inline]
    pub(crate) fn df_gru2_W_s(&self) -> &[f32] {
        self.0.sc(15360, 1536)
    }
    #[inline]
    pub(crate) fn df_gru2_R_q(&self) -> &[i8] {
        self.0.i(8650752, 786432)
    }
    #[inline]
    pub(crate) fn df_gru2_R_s(&self) -> &[f32] {
        self.0.sc(16896, 1536)
    }
}
