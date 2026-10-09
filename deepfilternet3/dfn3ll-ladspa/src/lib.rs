//! DeepFilterNet3-LL (low latency) as a mono LADSPA plugin: the network over the
//! shared pipeline in `dfn3-plugin`, with the LL hyper-parameters: embedding and
//! GRU hidden size 512, three DF GRU layers, grouped linears at 16 and 8 groups,
//! separable convs with temporal kernel 2 (one past frame), depthwise transposed
//! convs and no lookahead. The weights are traced from the upstream LL ONNX and
//! embedded in the `.so`; the port was validated stage by stage against ONNX Runtime.

#![allow(clippy::needless_range_loop)] // numeric kernels index by design

mod weights;

use dfn3_plugin::ladspa::{descriptor, Descriptor};
use dfn3_plugin::layers::{conv_p_add, convt_up, df_conv0, erb_conv0, DfConvp};
use dfn3_plugin::{AlignedBlob, Denoiser, Tensors, CH, DF_COEFS, NB_DF, NB_ERB};
use dfn_ops::gru_cell_packed as gru;
use dfn_ops::{dw_row_k3s1_accum, dw_row_k3s2_accum, grouped_linear, pointwise_conv2d};
use dfn_ops::{relu_inplace, sigmoid, vadd};
use std::os::raw::c_ulong;
use std::sync::{Arc, OnceLock};
use weights::W;

pub use dfn3_plugin::{atten_lim_from_db, HOP, SR};

/// The DeepFilterNet3-LL denoiser.
pub type Dfn3Ll = Denoiser<Dfn3LlNetwork>;

const EMB: usize = 512;
const HID: usize = 512;

static WEIGHTS: AlignedBlob<
    { include_bytes!(concat!(env!("OUT_DIR"), "/embedded_weights.bin")).len() },
> = AlignedBlob(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/embedded_weights.bin"
)));

/// The plugin's descriptor, which `ladspa_descriptor` hands to hosts.
pub static DESCRIPTOR: Descriptor = descriptor::<Dfn3LlNetwork>(
    0x00DF_3159,
    c"deep_filter_net3_ll_rs_mono",
    c"DeepFilterNet3-LL (Rust) mono noise reducer",
);

/// LADSPA host entry point.
#[no_mangle]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 {
        &DESCRIPTOR
    } else {
        std::ptr::null()
    }
}

/// The DeepFilterNet3-LL encoder, ERB decoder and deep-filter decoder.
pub struct Dfn3LlNetwork {
    w: Arc<W>,
    // The previous input frames of the temporal convolutions.
    erb0_pad: Vec<f32>,
    df0_pad: Vec<f32>,
    erb1_pad: Vec<f32>,
    erb2_pad: Vec<f32>,
    erb3_pad: Vec<f32>,
    df1_pad: Vec<f32>,
    convt3_pad: Vec<f32>,
    conv0out_pad: Vec<f32>,
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
    df2_h: Vec<f32>,
    gru_scratch: Vec<f32>,
    i16q: Vec<i16>,
    convp: DfConvp,
    // Largest live planes: the DF input (CH x NB_DF) and strided DF (CH x NB_DF/2).
    s1: Vec<f32>,
    s2: Vec<f32>,
    sc_flat: Vec<f32>,
    sc_512a: Vec<f32>,
    sc_512b: Vec<f32>,
    sc_512c: Vec<f32>,
    sc_trans: Vec<f32>,
}

impl dfn3_plugin::Network for Dfn3LlNetwork {
    const LOOKAHEAD: usize = 0;

    fn embedded() -> Self {
        static SHARED: OnceLock<Arc<W>> = OnceLock::new();
        let w = SHARED.get_or_init(|| {
            Arc::new(W(Tensors::embedded(
                &WEIGHTS,
                dfn_ops::pack_format::Geometry::DFN3LL,
            )))
        });
        Self {
            w: w.clone(),
            erb0_pad: vec![0.0; 2 * NB_ERB],
            df0_pad: vec![0.0; 2 * 2 * NB_DF],
            erb1_pad: vec![0.0; CH * NB_ERB],
            erb2_pad: vec![0.0; CH * (NB_ERB / 2)],
            erb3_pad: vec![0.0; CH * (NB_ERB / 4)],
            df1_pad: vec![0.0; CH * NB_DF],
            convt3_pad: vec![0.0; CH * (NB_ERB / 4)],
            conv0out_pad: vec![0.0; CH * NB_ERB],
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
            df2_h: vec![0.0; HID],
            gru_scratch: vec![0.0; 6 * HID],
            i16q: vec![0; HID],
            convp: DfConvp::new(),
            s1: vec![0.0; CH * NB_DF],
            s2: vec![0.0; CH * (NB_DF / 2)],
            sc_flat: vec![0.0; 48 * CH],
            sc_512a: vec![0.0; EMB],
            sc_512b: vec![0.0; EMB],
            sc_512c: vec![0.0; EMB],
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

impl Dfn3LlNetwork {
    fn encoder(&mut self, feat_erb: &[f32], feat_spec: &[f32]) -> f32 {
        let w = &*self.w;
        erb_conv0(
            &mut self.e0,
            &mut self.erb0_pad,
            feat_erb,
            w.enc_erb_conv0_dw_w(),
            w.enc_erb_conv0_dw_b(),
        );
        // erb_conv1/2: temporal kernel 2, depthwise stride 2 + pointwise + ReLU;
        // erb_conv3 stride 1.
        sep_k2_s2(
            &mut self.e1,
            &mut self.s1,
            &self.e0,
            &mut self.erb1_pad,
            w.enc_erb_conv1_dw_w(),
            w.enc_erb_conv1_pw_w(),
            w.enc_erb_conv1_pw_b(),
            NB_ERB,
            NB_ERB / 2,
        );
        sep_k2_s2(
            &mut self.e2,
            &mut self.s1,
            &self.e1,
            &mut self.erb2_pad,
            w.enc_erb_conv2_dw_w(),
            w.enc_erb_conv2_pw_w(),
            w.enc_erb_conv2_pw_b(),
            NB_ERB / 2,
            NB_ERB / 4,
        );
        sep_k2_s1(
            &mut self.e3,
            &mut self.s1,
            &self.e2,
            &mut self.erb3_pad,
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
        // df_conv1: temporal kernel 2, stride 2, into s2 [CH, 48].
        sep_k2_s2(
            &mut self.s2,
            &mut self.s1,
            &self.c0,
            &mut self.df1_pad,
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
            &mut self.sc_512a,
            &self.sc_flat,
            w.enc_df_fc_emb_w(),
            16,
            192,
            32,
        );
        relu_inplace(&mut self.sc_512a);
        for ch in 0..CH {
            for f in 0..8 {
                self.emb[f * CH + ch] = self.e3[ch * 8 + f];
            }
        }
        vadd(&mut self.emb, &self.sc_512a);

        // Squeezed GRU.
        grouped_linear(
            &mut self.sc_512b,
            &self.emb,
            w.enc_emb_gru_lin_in_w(),
            16,
            32,
            32,
        );
        relu_inplace(&mut self.sc_512b);
        gru(
            &mut self.enc_h,
            &self.sc_512b,
            w.enc_emb_gru_W_q(),
            w.enc_emb_gru_W_s(),
            w.enc_emb_gru_R_q(),
            w.enc_emb_gru_R_s(),
            w.enc_emb_gru_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );
        grouped_linear(
            &mut self.sc_512c,
            &self.enc_h,
            w.enc_emb_gru_lin_out_w(),
            16,
            32,
            32,
        );
        relu_inplace(&mut self.sc_512c);
        self.emb.copy_from_slice(&self.sc_512c);

        dfn3_plugin::lsnr(&self.emb, w.enc_lsnr_fc_w(), w.enc_lsnr_fc_b()[0])
    }

    fn erb_decoder(&mut self, erb_mask: &mut [f32; NB_ERB]) {
        let w = &*self.w;
        grouped_linear(
            &mut self.sc_512b,
            &self.emb,
            w.erb_emb_gru_lin_in_w(),
            16,
            32,
            32,
        );
        relu_inplace(&mut self.sc_512b);
        gru(
            &mut self.erb0_h,
            &self.sc_512b,
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
            &mut self.sc_512a,
            &self.erb1_h,
            w.erb_emb_gru_lin_out_w(),
            16,
            32,
            32,
        );
        relu_inplace(&mut self.sc_512a);
        for f in 0..8 {
            for ch in 0..CH {
                self.sc_512c[ch * 8 + f] = self.sc_512a[f * CH + ch];
            }
        }
        // Each decoder stage: a 1x1 on the encoder skip plus the stage below.
        conv_p_add(
            &mut self.s1,
            &self.e3,
            &self.sc_512c[..CH * 8],
            w.erb_conv3p_dw_w(),
            w.erb_conv3p_dw_b(),
            8,
        );
        convt3_k2(
            &mut self.s1,
            &mut self.s2,
            &mut self.convt3_pad,
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
        // conv0_out: temporal kernel 2, weights [1, CH, 2 (past, current), 3], + sigmoid.
        let wt = w.erb_conv0_out_w();
        erb_mask.fill(w.erb_conv0_out_b()[0]);
        for ci in 0..CH {
            let base = ci * 6;
            let past = &self.conv0out_pad[ci * NB_ERB..ci * NB_ERB + NB_ERB];
            let cur = &self.s2[ci * NB_ERB..ci * NB_ERB + NB_ERB];
            dw_row_k3s1_accum(erb_mask, past, wt[base], wt[base + 1], wt[base + 2]);
            dw_row_k3s1_accum(erb_mask, cur, wt[base + 3], wt[base + 4], wt[base + 5]);
        }
        for v in erb_mask.iter_mut() {
            *v = sigmoid(*v);
        }
        self.conv0out_pad.copy_from_slice(&self.s2[..CH * NB_ERB]);
    }

    fn df_decoder(&mut self, coefs: &mut [f32; DF_COEFS]) {
        let w = &*self.w;
        grouped_linear(&mut self.sc_512b, &self.emb, w.df_gru_lin_in_w(), 8, 64, 64);
        relu_inplace(&mut self.sc_512b);
        gru(
            &mut self.df0_h,
            &self.sc_512b,
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
        gru(
            &mut self.df2_h,
            &self.df1_h,
            w.df_gru2_W_q(),
            w.df_gru2_W_s(),
            w.df_gru2_R_q(),
            w.df_gru2_R_s(),
            w.df_gru2_B(),
            HID,
            HID,
            &mut self.gru_scratch,
            &mut self.i16q,
        );

        // The GRU output plus the skip from the embedding.
        self.sc_512a.copy_from_slice(&self.df2_h);
        grouped_linear(&mut self.sc_512c, &self.emb, w.df_skip_w(), 16, 32, 32);
        vadd(&mut self.sc_512a, &self.sc_512c);
        grouped_linear(coefs, &self.sc_512a, w.df_out_w(), 16, 32, 60);
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

/// Separable conv: temporal kernel 2 (`pad` holds the previous `src`), frequency
/// kernel 3 stride 2, pointwise, ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_k2_s2(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    pad: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_in: usize,
    w_out: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_out..c * w_out + w_out];
        out_c.fill(0.0);
        let wk = &dw[c * 6..c * 6 + 6];
        let past = &pad[c * w_in..c * w_in + w_in];
        let cur = &src[c * w_in..c * w_in + w_in];
        dw_row_k3s2_accum(out_c, past, wk[0], wk[1], wk[2]);
        dw_row_k3s2_accum(out_c, cur, wk[3], wk[4], wk[5]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_out);
    relu_inplace(&mut dst[..CH * w_out]);
    pad[..CH * w_in].copy_from_slice(&src[..CH * w_in]);
}

/// Separable conv: temporal kernel 2, frequency kernel 3 stride 1, pointwise, ReLU.
#[allow(clippy::too_many_arguments)]
fn sep_k2_s1(
    dst: &mut [f32],
    scr: &mut [f32],
    src: &[f32],
    pad: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_io: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 6..c * 6 + 6];
        let past = &pad[c * w_io..c * w_io + w_io];
        let cur = &src[c * w_io..c * w_io + w_io];
        dw_row_k3s1_accum(out_c, past, wk[0], wk[1], wk[2]);
        dw_row_k3s1_accum(out_c, cur, wk[3], wk[4], wk[5]);
    }
    pointwise_conv2d(dst, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut dst[..CH * w_io]);
    pad[..CH * w_io].copy_from_slice(&src[..CH * w_io]);
}

/// `convt3`: temporal kernel 2 depthwise, frequency kernel 3 stride 1, into `scr`;
/// pointwise back into `sio`, ReLU.
fn convt3_k2(
    sio: &mut [f32],
    scr: &mut [f32],
    pad: &mut [f32],
    dw: &[f32],
    pw: &[f32],
    pb: &[f32],
    w_io: usize,
) {
    for c in 0..CH {
        let out_c = &mut scr[c * w_io..c * w_io + w_io];
        out_c.fill(0.0);
        let wk = &dw[c * 6..c * 6 + 6];
        let past = &pad[c * w_io..c * w_io + w_io];
        let cur = &sio[c * w_io..c * w_io + w_io];
        dw_row_k3s1_accum(out_c, past, wk[0], wk[1], wk[2]);
        dw_row_k3s1_accum(out_c, cur, wk[3], wk[4], wk[5]);
    }
    pad[..CH * w_io].copy_from_slice(&sio[..CH * w_io]);
    pointwise_conv2d(sio, scr, pw, pb, CH, CH, w_io);
    relu_inplace(&mut sio[..CH * w_io]);
}
