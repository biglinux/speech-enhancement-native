//! DeepFilterNet3 as a mono LADSPA plugin: the network (GRU hidden size 256, two
//! frames of lookahead) over the shared pipeline in `dfn3-plugin`. The weights are
//! derived from the upstream DeepFilterNet3 ONNX and embedded in the `.so`.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design

mod weights;

use dfn3_plugin::ladspa::{Descriptor, descriptor};
use dfn3_plugin::layers::{DfConvp, conv_p_add, convt_up, df_conv0, erb_conv0};
use dfn3_plugin::{AlignedBlob, CH, DF_COEFS, Denoiser, NB_DF, NB_ERB, Tensors};
use ops::gru_cell_packed as gru;
use ops::{dw_row_k3s1_accum, dw_row_k3s2_accum, grouped_linear, pointwise_conv2d};
use ops::{relu_inplace, sigmoid, vadd};
use std::os::raw::c_ulong;
use std::sync::OnceLock;
use weights::W;

pub use dfn3_plugin::HOP;

/// The DeepFilterNet3 denoiser.
pub type Dfn3 = Denoiser<Dfn3Network>;

const EMB: usize = 512;
const HID: usize = 256;

static WEIGHTS: AlignedBlob<
    { include_bytes!(concat!(env!("OUT_DIR"), "/embedded_weights.bin")).len() },
> = AlignedBlob(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/embedded_weights.bin"
)));

/// The plugin's descriptor, which `ladspa_descriptor` hands to hosts.
pub static DESCRIPTOR: Descriptor = descriptor::<Dfn3Network>(
    0x00DF_3130,
    c"deep_filter_net3_rs_mono",
    c"DeepFilterNet3 (Rust) mono noise reducer",
);

/// LADSPA host entry point.
#[unsafe(no_mangle)]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 {
        &DESCRIPTOR
    } else {
        std::ptr::null()
    }
}

/// The DeepFilterNet3 encoder, ERB decoder and deep-filter decoder.
pub struct Dfn3Network {
    w: &'static W,
    erb0_pad: Vec<f32>,
    df0_pad: Vec<f32>,
    e0: Vec<f32>,
    e1: Vec<f32>,
    e2: Vec<f32>,
    e3: Vec<f32>,
    c0: Vec<f32>,
    emb: Vec<f32>,
    enc_h: Vec<f32>,
    erb0_h: Vec<f32>,
    erb1_h: Vec<f32>,
    df0_h: Vec<f32>,
    df1_h: Vec<f32>,
    gru_scratch: Vec<f32>,
    i16q: Vec<i16>,
    convp: DfConvp,
    // Largest live planes: the DF input (CH x NB_DF) and strided DF (CH x NB_DF/2).
    s1: Vec<f32>,
    s2: Vec<f32>,
    sc_hid: Vec<f32>,
    sc_dfcomb: Vec<f32>,
    sc_skip: Vec<f32>,
    sc_emb: Vec<f32>,
    sc_flat: Vec<f32>,
    sc_emb2d: Vec<f32>,
    sc_trans: Vec<f32>,
}

impl dfn3_plugin::Network for Dfn3Network {
    const LOOKAHEAD: usize = 2;

    fn embedded() -> Self {
        static SHARED: OnceLock<W> = OnceLock::new();
        let w = SHARED.get_or_init(|| {
            W(Tensors::embedded(
                &WEIGHTS,
                ops::pack_format::Geometry::DFN3,
            ))
        });
        Self {
            w,
            erb0_pad: vec![0.0; 2 * NB_ERB],
            df0_pad: vec![0.0; 2 * 2 * NB_DF],
            e0: vec![0.0; CH * NB_ERB],
            e1: vec![0.0; CH * (NB_ERB / 2)],
            e2: vec![0.0; CH * (NB_ERB / 4)],
            e3: vec![0.0; CH * (NB_ERB / 4)],
            c0: vec![0.0; CH * NB_DF],
            emb: vec![0.0; EMB],
            enc_h: vec![0.0; HID],
            erb0_h: vec![0.0; HID],
            erb1_h: vec![0.0; HID],
            df0_h: vec![0.0; HID],
            df1_h: vec![0.0; HID],
            gru_scratch: vec![0.0; 6 * HID],
            i16q: vec![0; HID],
            convp: DfConvp::new(),
            s1: vec![0.0; CH * NB_DF],
            s2: vec![0.0; CH * (NB_DF / 2)],
            sc_hid: vec![0.0; HID],
            sc_dfcomb: vec![0.0; HID],
            sc_skip: vec![0.0; HID],
            sc_emb: vec![0.0; EMB],
            sc_flat: vec![0.0; 48 * CH],
            sc_emb2d: vec![0.0; CH * 8],
            sc_trans: vec![0.0; CH * 33],
        }
    }

    fn infer(
        &mut self,
        feat_erb: &[f32; NB_ERB],
        feat_spec: &[f32; 2 * NB_DF],
        erb_mask: &mut [f32; NB_ERB],
        coefs: &mut [f32; DF_COEFS],
    ) -> f32 {
        let lsnr = self.encoder(feat_erb, feat_spec);
        self.erb_decoder(erb_mask);
        self.df_decoder(coefs);
        lsnr
    }
}

impl Dfn3Network {
    fn encoder(&mut self, feat_erb: &[f32], feat_spec: &[f32]) -> f32 {
        let w = self.w;
        erb_conv0(
            &mut self.e0,
            &mut self.erb0_pad,
            feat_erb,
            w.enc_erb_conv0_dw_w(),
            w.enc_erb_conv0_dw_b(),
        );
        // erb_conv1/2: depthwise stride 2 + pointwise + ReLU; erb_conv3 stride 1.
        sep_s2(
            &mut self.e1,
            &mut self.s1,
            &self.e0,
            w.enc_erb_conv1_dw_w(),
            w.enc_erb_conv1_pw_w(),
            w.enc_erb_conv1_pw_b(),
            NB_ERB,
            NB_ERB / 2,
        );
        sep_s2(
            &mut self.e2,
            &mut self.s1,
            &self.e1,
            w.enc_erb_conv2_dw_w(),
            w.enc_erb_conv2_pw_w(),
            w.enc_erb_conv2_pw_b(),
            NB_ERB / 2,
            NB_ERB / 4,
        );
        sep_s1(
            &mut self.e3,
            &mut self.s1,
            &self.e2,
            w.enc_erb_conv3_dw_w(),
            w.enc_erb_conv3_pw_w(),
            w.enc_erb_conv3_pw_b(),
            NB_ERB / 4,
        );
        df_conv0(
            &mut self.c0,
            &mut self.s1,
            &mut self.df0_pad,
            feat_spec,
            w.enc_df_conv0_dw_w(),
            w.enc_df_conv0_pw_w(),
            w.enc_df_conv0_pw_b(),
        );
        // df_conv1: depthwise stride 2 + pointwise + ReLU into s2 [CH, 48].
        sep_s2(
            &mut self.s2,
            &mut self.s1,
            &self.c0,
            w.enc_df_conv1_dw_w(),
            w.enc_df_conv1_pw_w(),
            w.enc_df_conv1_pw_b(),
            NB_DF,
            NB_DF / 2,
        );

        // Embedding: the DF path transposed to [48, CH] through a grouped linear,
        // plus e3 transposed to [8, CH].
        for ch in 0..CH {
            for f in 0..48 {
                self.sc_flat[f * CH + ch] = self.s2[ch * 48 + f];
            }
        }
        grouped_linear(
            &mut self.sc_emb,
            &self.sc_flat,
            w.enc_df_fc_emb_w(),
            32,
            96,
            16,
        );
        relu_inplace(&mut self.sc_emb);
        for ch in 0..CH {
            for f in 0..8 {
                self.emb[f * CH + ch] = self.e3[ch * 8 + f];
            }
        }
        vadd(&mut self.emb, &self.sc_emb);

        // Squeezed GRU.
        grouped_linear(
            &mut self.sc_hid,
            &self.emb,
            w.enc_emb_gru_lin_in_w(),
            16,
            32,
            16,
        );
        relu_inplace(&mut self.sc_hid);
        gru(
            &mut self.enc_h,
            &self.sc_hid,
            w.enc_emb_gru0_W_q(),
            w.enc_emb_gru0_W_s(),
            w.enc_emb_gru0_R_q(),
            w.enc_emb_gru0_R_s(),
            w.enc_emb_gru0_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        grouped_linear(
            &mut self.sc_emb,
            &self.enc_h,
            w.enc_emb_gru_lin_out_w(),
            16,
            16,
            32,
        );
        relu_inplace(&mut self.sc_emb);
        self.emb.copy_from_slice(&self.sc_emb);

        dfn3_plugin::lsnr(&self.emb, w.enc_lsnr_fc_w(), w.enc_lsnr_fc_b()[0])
    }

    fn erb_decoder(&mut self, erb_mask: &mut [f32; NB_ERB]) {
        let w = self.w;
        grouped_linear(
            &mut self.sc_hid,
            &self.emb,
            w.erb_emb_gru_lin_in_w(),
            16,
            32,
            16,
        );
        relu_inplace(&mut self.sc_hid);
        gru(
            &mut self.erb0_h,
            &self.sc_hid,
            w.erb_emb_gru0_W_q(),
            w.erb_emb_gru0_W_s(),
            w.erb_emb_gru0_R_q(),
            w.erb_emb_gru0_R_s(),
            w.erb_emb_gru0_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        gru(
            &mut self.erb1_h,
            &self.erb0_h,
            w.erb_emb_gru1_W_q(),
            w.erb_emb_gru1_W_s(),
            w.erb_emb_gru1_R_q(),
            w.erb_emb_gru1_R_s(),
            w.erb_emb_gru1_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        grouped_linear(
            &mut self.sc_emb,
            &self.erb1_h,
            w.erb_emb_gru_lin_out_w(),
            16,
            16,
            32,
        );
        relu_inplace(&mut self.sc_emb);
        for f in 0..8 {
            for ch in 0..CH {
                self.sc_emb2d[ch * 8 + f] = self.sc_emb[f * CH + ch];
            }
        }
        // Each decoder stage: a 1x1 on the encoder skip plus the stage below.
        conv_p_add(
            &mut self.s1,
            &self.e3,
            &self.sc_emb2d,
            w.erb_conv3p_dw_w(),
            w.erb_conv3p_dw_b(),
            8,
        );
        convt_dw_pw(
            &mut self.s1,
            &mut self.s2,
            w.erb_convt3_dw_w(),
            w.erb_convt3_pw_w(),
            w.erb_convt3_pw_b(),
            8,
        );
        conv_p_add(
            &mut self.s2[..CH * 8],
            &self.e2,
            &self.s1[..CH * 8],
            w.erb_conv2p_dw_w(),
            w.erb_conv2p_dw_b(),
            8,
        );
        convt_up(
            &mut self.s2,
            &mut self.s1,
            &mut self.sc_trans,
            w.erb_convt2_dw_w(),
            w.erb_convt2_pw_w(),
            w.erb_convt2_pw_b(),
            8,
            16,
        );
        conv_p_add(
            &mut self.s1[..CH * 16],
            &self.e1,
            &self.s2[..CH * 16],
            w.erb_conv1p_dw_w(),
            w.erb_conv1p_dw_b(),
            16,
        );
        convt_up(
            &mut self.s1,
            &mut self.s2,
            &mut self.sc_trans,
            w.erb_convt1_dw_w(),
            w.erb_convt1_pw_w(),
            w.erb_convt1_pw_b(),
            16,
            32,
        );
        conv_p_add(
            &mut self.s2[..CH * NB_ERB],
            &self.e0,
            &self.s1[..CH * NB_ERB],
            w.erb_conv0p_dw_w(),
            w.erb_conv0p_dw_b(),
            NB_ERB,
        );
        // conv0_out: [1, CH, 1, 3] + sigmoid.
        let wt = w.erb_conv0_out_w();
        erb_mask.fill(w.erb_conv0_out_b()[0]);
        for ci in 0..CH {
            let wc = &wt[ci * 3..ci * 3 + 3];
            dw_row_k3s1_accum(
                erb_mask,
                &self.s2[ci * NB_ERB..ci * NB_ERB + NB_ERB],
                wc[0],
                wc[1],
                wc[2],
            );
        }
        for v in erb_mask.iter_mut() {
            *v = sigmoid(*v);
        }
    }

    fn df_decoder(&mut self, coefs: &mut [f32; DF_COEFS]) {
        let w = self.w;
        grouped_linear(&mut self.sc_hid, &self.emb, w.df_gru_lin_in_w(), 8, 64, 32);
        relu_inplace(&mut self.sc_hid);
        gru(
            &mut self.df0_h,
            &self.sc_hid,
            w.df_gru0_W_q(),
            w.df_gru0_W_s(),
            w.df_gru0_R_q(),
            w.df_gru0_R_s(),
            w.df_gru0_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        gru(
            &mut self.df1_h,
            &self.df0_h,
            w.df_gru1_W_q(),
            w.df_gru1_W_s(),
            w.df_gru1_R_q(),
            w.df_gru1_R_s(),
            w.df_gru1_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );

        self.sc_dfcomb.copy_from_slice(&self.df1_h);
        grouped_linear(&mut self.sc_skip, &self.emb, w.df_skip_w(), 16, 32, 16);
        vadd(&mut self.sc_dfcomb, &self.sc_skip);
        grouped_linear(coefs, &self.sc_dfcomb, w.df_out_w(), 16, 16, 60);
        for v in coefs.iter_mut() {
            *v = v.tanh();
        }
        self.convp.add_to(
            coefs,
            &self.c0,
            w.df_convp_dw_w(),
            w.df_convp_pw_w(),
            w.df_convp_pw_b(),
        );
    }
}

/// Separable conv: depthwise kernel 3 stride 2 (pad 1), pointwise, ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_s2(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_in: usize,
    w_out: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_out..c * w_out + w_out];
        out_c.fill(0.0);
        let wk = &dw[c * 3..c * 3 + 3];
        dw_row_k3s2_accum(out_c, &src[c * w_in..c * w_in + w_in], wk[0], wk[1], wk[2]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_out);
    relu_inplace(&mut dst[..CH * w_out]);
}

/// Separable conv: depthwise kernel 3 stride 1 (pad 1), pointwise, ReLU.
fn sep_s1(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_io: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 3..c * 3 + 3];
        dw_row_k3s1_accum(out_c, &src[c * w_io..c * w_io + w_io], wk[0], wk[1], wk[2]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut dst[..CH * w_io]);
}

/// Depthwise kernel 3 stride 1 into `scr`, pointwise back into `sio`, ReLU.
fn convt_dw_pw(sio: &mut [f32], scr: &mut [f32], dw: &[f32], pw: &[f32], pb: &[f32], w_io: usize) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 3..c * 3 + 3];
        dw_row_k3s1_accum(out_c, &sio[c * w_io..c * w_io + w_io], wk[0], wk[1], wk[2]);
    }
    pointwise_conv2d(sio, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut sio[..CH * w_io]);
}
