//! Streaming: one frame per call, recurrent state carried across calls.
//! `streaming_core_matches_batch` checks it against `forward`.

use super::*;

struct GtRing {
    cap: usize,
    frames: Vec<f32>, // cap * hidden * f, time-major (oldest first)
    tra_h: Vec<f32>,
}

impl GtRing {
    fn new(dil: usize) -> Self {
        Self {
            cap: 2 * dil + 1,
            frames: Vec::new(),
            tra_h: vec![0.0; 16],
        }
    }
}

/// Per-frame streaming core. `process_frame` takes one `(257,1,2)` spec_e/spec_y
/// frame (freq-major `[fb*2 + {re,im}]`) and returns the masked frame, carrying
/// all time-recurrent state (gt-block dw rings + temporal-attention GRUs, and the
/// dual-path inter GRUs) across calls.
pub struct StreamCore {
    state: CoreState,
    arena: Workspace,
    out_frame: Vec<f32>,
}

struct CoreState {
    gt: Vec<GtRing>,
    inter_h: Vec<Vec<f32>>,
}

impl StreamCore {
    fn new() -> Self {
        Self {
            state: CoreState {
                gt: GT_DILATIONS.iter().map(|&d| GtRing::new(d)).collect(),
                inter_h: vec![Vec::new(), Vec::new()],
            },
            arena: Workspace::default(),
            out_frame: vec![0.0; FB * 2],
        }
    }

    /// Build every shape/state before streaming, then retain and zero it.
    /// Every operation of `process_frame` runs on every frame whatever the samples,
    /// and the rings start at their full dilation span, so one traversal records
    /// every buffer shape.
    #[must_use]
    pub fn prepared(net: &Net) -> Self {
        let mut core = Self::new();
        core.process_frame(net, &[0.0; FB * 2], &[0.0; FB * 2]);
        core.reset();
        core.arena.seal();
        core
    }

    pub fn reset(&mut self) {
        for r in &mut self.state.gt {
            r.frames.fill(0.0);
            r.tra_h.fill(0.0);
        }
        for h in &mut self.state.inter_h {
            h.fill(0.0);
        }
        self.out_frame.fill(0.0);
    }
}

impl CoreState {
    fn gt_step<'a>(
        &mut self,
        arena: &'a Workspace,
        x: &GTensor,
        g: &GtBlock,
        idx: usize,
    ) -> GTensor<'a> {
        let x1 = chunk(arena, x, false);
        let x2 = chunk(arena, x, true);
        let s = sfe(arena, &x1);
        let mut h = conv(arena, &s, &g.pc1, 1, 1, 0, 0, 1);
        prelu_(&mut h, &g.pc1.prelu); // (hidden,1,F)
        let hidden = h.c;
        let fdim = h.f;
        // dw time ring (oldest first, length cap)
        let r = &mut self.gt[idx];
        if r.frames.is_empty() {
            r.frames = vec![0.0; r.cap * hidden * fdim];
        }
        // shift down one frame, append current at the newest slot
        let fsz = hidden * fdim;
        r.frames.copy_within(fsz.., 0);
        r.frames[(r.cap - 1) * fsz..].copy_from_slice(&h.d);
        let mut ring = GTensor::new(arena, hidden, r.cap, fdim);
        // ring as (hidden, cap, F): ring[(c*cap+t)*F+f] = frames[t*fsz + c*F + f].
        // For fixed (c,t) the F-run is contiguous in both, so this is a copy.
        for t in 0..r.cap {
            for c in 0..hidden {
                let src = t * fsz + c * fdim;
                let dst = (c * r.cap + t) * fdim;
                ring.d[dst..dst + fdim].copy_from_slice(&r.frames[src..src + fdim]);
            }
        }
        // pad_t 0: the ring already holds the context
        let mut dw = conv(
            arena,
            &ring,
            &g.dw,
            hidden,
            1,
            1,
            0,
            GT_DILATIONS[idx] as i64,
        );
        prelu_(&mut dw, &g.dw.prelu); // (hidden,1,F)
        let mut pc2 = conv(arena, &dw, &g.pc2, 1, 1, 0, 0, 1);
        // temporal attention (streaming GRU)
        let c = pc2.c;
        let mut seq = Scratch::zeros(arena, c);
        for ci in 0..c {
            let row = &pc2.d[ci * fdim..ci * fdim + fdim];
            let sm: f32 = row.iter().map(|&v| v * v).sum();
            seq[ci] = sm / fdim as f32;
        }
        let h = &mut self.gt[idx].tra_h[..g.gru.hidden];
        gru_cell(&seq, h, &g.gru.wih, &g.gru.whh, &g.gru.bih, &g.gru.bhh);
        let at = linear(arena, h, 1, &g.fc);
        for ci in 0..c {
            let g = sigmoid(at[ci]);
            for v in &mut pc2.d[ci * fdim..ci * fdim + fdim] {
                *v *= g;
            }
        }
        shuffle2(arena, &pc2, &x2)
    }

    fn dpgrnn_step<'a>(
        &mut self,
        arena: &'a Workspace,
        x: &GTensor,
        d: &DpGrnn,
        idx: usize,
    ) -> GTensor<'a> {
        let (c, fd) = (x.c, x.f);
        let mut xp = Scratch::zeros(arena, fd * c);
        for f in 0..fd {
            for ci in 0..c {
                xp[f * c + ci] = x.at(ci, 0, f);
            }
        }
        // intra: bidirectional grnn over freq (per-frame, no carried state)
        let mut intra = grnn(arena, &xp, 1, fd, c, &d.intra);
        intra = linear(arena, &intra, fd, &d.intra_fc);
        layernorm_last2(&mut intra, 1, fd, c, &d.intra_ln);
        for i in 0..intra.len() {
            intra[i] += xp[i];
        }
        // inter: per-freq unidirectional GRU carrying state across frames
        if self.inter_h[idx].is_empty() {
            self.inter_h[idx] = vec![0.0; fd * c];
        }
        let half = c / 2;
        // Per-freq GRUs are independent across `f` and share weights, so batch them
        // 8 freqs per SIMD lane (feature-major), same trick as the DAF controllers.
        let mut inter = Scratch::zeros(arena, fd * c);
        let ih = &mut self.inter_h[idx];
        for (rnn, g) in d.inter.fwd.iter().enumerate() {
            let off = rnn * half;
            let h1 = g.hidden;
            let mut f0 = 0;
            while f0 < fd {
                let lanes = (fd - f0).min(8);
                let mut x8 = [0.0f32; 16 * 8];
                let mut h8 = [0.0f32; 16 * 8];
                for lane in 0..lanes {
                    let f = f0 + lane;
                    for i in 0..half {
                        x8[i * 8 + lane] = intra[f * c + off + i];
                    }
                    for j in 0..h1 {
                        h8[j * 8 + lane] = ih[f * c + off + j];
                    }
                }
                ops::gru8(&x8, half, &mut h8, h1, &g.wih, &g.whh, &g.bih, &g.bhh);
                for lane in 0..lanes {
                    let f = f0 + lane;
                    for j in 0..h1 {
                        let v = h8[j * 8 + lane];
                        ih[f * c + off + j] = v;
                        inter[f * c + off + j] = v;
                    }
                }
                f0 += 8;
            }
        }
        inter = linear(arena, &inter, fd, &d.inter_fc);
        layernorm_last2(&mut inter, 1, fd, c, &d.inter_ln);
        for i in 0..inter.len() {
            inter[i] += intra[i];
        }
        let mut out = GTensor::new(arena, c, 1, fd);
        for f in 0..fd {
            for ci in 0..c {
                out.set(ci, 0, f, inter[f * c + ci]);
            }
        }
        out
    }
}

impl StreamCore {
    /// Process one spec frame: `spec_e`/`spec_y` are `[fb*2 + {re,im}]` (257 bins).
    pub fn process_frame(&mut self, net: &Net, spec_e: &[f32], spec_y: &[f32]) -> &[f32] {
        let arena = &self.arena;
        let fe = feat(arena, spec_e, 1, &net.erb_bm);
        let fy = feat(arena, spec_y, 1, &net.erb_bm);
        let mut ft = GTensor::new(arena, 18, 1, 129);
        for c in 0..9 {
            for f in 0..129 {
                ft.set(c, 0, f, fe.at(c, 0, f));
                ft.set(9 + c, 0, f, fy.at(c, 0, f));
            }
        }
        let st = &mut self.state;
        let en0 = conv_block(arena, &ft, &net.en0, 1, 2, false, false);
        let en1 = conv_block(arena, &en0, &net.en1, 2, 2, false, false);
        let en2 = st.gt_step(arena, &en1, &net.gt[0], 0);
        let en3 = st.gt_step(arena, &en2, &net.gt[1], 1);
        let en4 = st.gt_step(arena, &en3, &net.gt[2], 2);
        let d1 = st.dpgrnn_step(arena, &en4, &net.dp[0], 0);
        let d2 = st.dpgrnn_step(arena, &d1, &net.dp[1], 1);
        let mut x = st.gt_step(arena, &add(arena, &d2, &en4), &net.gt[3], 3);
        x = st.gt_step(arena, &add(arena, &x, &en3), &net.gt[4], 4);
        x = st.gt_step(arena, &add(arena, &x, &en2), &net.gt[5], 5);
        x = conv_block(arena, &add(arena, &x, &en1), &net.de3, 2, 2, true, false);
        x = conv_block(arena, &add(arena, &x, &en0), &net.de4, 1, 2, true, true);
        let m = erb_bs(arena, &x, &net.erb_bs);
        // The output is written into the reused frame buffer and returned as a view.
        for fb in 0..FB {
            let er = spec_e[fb * 2];
            let ei = spec_e[fb * 2 + 1];
            let mr = m.at(0, 0, fb);
            let mi = m.at(1, 0, fb);
            self.out_frame[fb * 2] = er * mr - ei * mi;
            self.out_frame[fb * 2 + 1] = ei * mr + er * mi;
        }
        &self.out_frame
    }
}
