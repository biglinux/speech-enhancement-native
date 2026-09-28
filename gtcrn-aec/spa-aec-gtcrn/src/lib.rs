//! PipeWire SPA AEC plugin exposing `spa_audio_aec`, backed by the native Rust
//! GTCRN-AEC (`aec-gtcrn`). `module-echo-cancel` loads it via
//! `library.name = aec/libspa-aec-gtcrn`. PipeWire runs the graph at 48 kHz; this
//! resamples to the model's 16 kHz, streams per 256-sample hop, and resamples back,
//! buffering through FIFOs so any host block size works.
//!
//! Only the pieces `module-echo-cancel` calls are implemented: factory enum,
//! handle `get_interface`, and the AEC methods `init`/`init2`/`run`/`activate`/
//! `deactivate`. Everything else returns `-ENOTSUP`.

#![allow(non_camel_case_types, unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

mod reference_bypass;
use reference_bypass::ReferenceBypass;

use aec_gtcrn::{
    Model, Streamer,
    hbaec::HbAec,
    resample::{Down3, Up3},
};

// ── SPA C ABI structs (mirror spa/support/plugin.h + interfaces/audio/aec.h) ──

#[repr(C)]
struct spa_dict_item {
    key: *const c_char,
    value: *const c_char,
}
#[repr(C)]
struct spa_dict {
    flags: u32,
    n_items: u32,
    items: *const spa_dict_item,
}
#[repr(C)]
struct spa_audio_info_raw {
    format: u32,
    flags: u32,
    rate: u32,
    channels: u32,
    position: [u32; 64],
}
#[repr(C)]
struct spa_callbacks {
    funcs: *const c_void,
    data: *mut c_void,
}
#[repr(C)]
struct spa_interface {
    type_: *const c_char,
    version: u32,
    cb: spa_callbacks,
}
#[repr(C)]
struct spa_audio_aec {
    iface: spa_interface,
    name: *const c_char,
    info: *const spa_dict,
    latency: *const c_char,
}
#[repr(C)]
struct spa_handle {
    version: u32,
    get_interface:
        Option<unsafe extern "C" fn(*mut spa_handle, *const c_char, *mut *mut c_void) -> c_int>,
    clear: Option<unsafe extern "C" fn(*mut spa_handle) -> c_int>,
}
#[repr(C)]
struct spa_interface_info {
    type_: *const c_char,
}
#[repr(C)]
struct spa_handle_factory {
    version: u32,
    name: *const c_char,
    info: *const spa_dict,
    get_size: Option<unsafe extern "C" fn(*const spa_handle_factory, *const spa_dict) -> usize>,
    init: Option<
        unsafe extern "C" fn(
            *const spa_handle_factory,
            *mut spa_handle,
            *const spa_dict,
            *const c_void,
            u32,
        ) -> c_int,
    >,
    enum_interface_info: Option<
        unsafe extern "C" fn(
            *const spa_handle_factory,
            *mut *const spa_interface_info,
            *mut u32,
        ) -> c_int,
    >,
}

// Opaque C pointees: only their pointers cross this ABI.
#[repr(C)]
struct spa_hook {
    _private: [u8; 0],
}
#[repr(C)]
struct spa_audio_aec_events {
    _private: [u8; 0],
}
#[repr(C)]
struct spa_pod_builder {
    _private: [u8; 0],
}
#[repr(C)]
struct spa_pod {
    _private: [u8; 0],
}

/// `spa_audio_aec_methods` — order matches aec.h (version 1:3).
#[repr(C)]
struct spa_audio_aec_methods {
    version: u32,
    add_listener: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut spa_hook,
            *const spa_audio_aec_events,
            *mut c_void,
        ) -> c_int,
    >,
    init: Option<
        unsafe extern "C" fn(*mut c_void, *const spa_dict, *const spa_audio_info_raw) -> c_int,
    >,
    run: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const *const f32,
            *const *const f32,
            *mut *mut f32,
            u32,
        ) -> c_int,
    >,
    set_props: Option<unsafe extern "C" fn(*mut c_void, *const spa_dict) -> c_int>,
    activate: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    deactivate: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    enum_props: Option<unsafe extern "C" fn(*mut c_void, c_int, *mut spa_pod_builder) -> c_int>,
    get_params: Option<unsafe extern "C" fn(*mut c_void, *mut spa_pod_builder) -> c_int>,
    set_params: Option<unsafe extern "C" fn(*mut c_void, *const spa_pod) -> c_int>,
    init2: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const spa_dict,
            *mut spa_audio_info_raw,
            *mut spa_audio_info_raw,
            *mut spa_audio_info_raw,
        ) -> c_int,
    >,
}

/// Raw-pointer statics are not `Sync`; the C ABI only reads them. This wrapper
/// asserts that (the plugin's statics are immutable vtables).
struct SyncWrap<T>(T);
// SAFETY: these three immutable statics point only to immutable static data
// and function entry points. This is NOT a blanket guarantee for arbitrary T.
unsafe impl Sync for SyncWrap<spa_audio_aec_methods> {}
unsafe impl Sync for SyncWrap<spa_interface_info> {}
unsafe impl Sync for SyncWrap<spa_handle_factory> {}

const AEC_TYPE: &[u8] = b"Spa:Pointer:Interface:Audio:AEC\0";
const FACTORY_NAME: &[u8] = b"audio.aec\0";
// These values are checked against the installed PipeWire C headers by
// gtcrn-aec/tools/spa-aec-abi-check.c. This ratio is a block requirement, NOT a measured
// input-to-output group delay.
const AEC_BLOCK: &[u8] = b"768/48000\0";
const AEC_NAME: &[u8] = b"gtcrn\0";
const SPA_AUDIO_FORMAT_F32P: u32 = 0x206;
const DEFAULT_MODEL: &str = "/usr/share/gtcrn-aec-native/localvqe-pi-aec-v1-49k-f32.gguf";

// ── engine state ──────────────────────────────────────────────────────────

// One buffered neural hop, one STFT hop, and the two resampling FIR delays.
// Both bands must use this delay regardless of the host's callback size.
const AEC_LATENCY_48K: usize = 2 * MAX_CHUNK + 2 * aec_gtcrn::resample::LPF_DELAY;
const MIC_RING: usize = 8192; // > 2·latency + LPF span + one bounded chunk
const MAX_CHUNK: usize = 768; // one 16 kHz neural hop at 48 kHz

struct Engine {
    reference_bypass: Option<ReferenceBypass>,
    model: Model,
    streamer: Streamer,
    down_mic: Down3,
    down_ref: Down3,
    up_out: Up3,
    fifo_mic: Vec<f32>, // 16 kHz, awaiting a full hop
    fifo_ref: Vec<f32>,
    fifo_out: Vec<f32>,     // 48 kHz, awaiting emit
    lpf: Vec<f32>,          // 7.6 kHz low-pass prototype (complement gives the hi band)
    mic_hist: Box<[f32]>,   // raw 48 kHz mic ring, indexed by absolute sample count
    in_count: usize,        // total mic samples pushed
    out_count: usize,       // total samples emitted
    rs: Vec<f32>,           // reused resampler output scratch (RT: no per-quantum alloc)
    fifo_gain: Vec<f32>,    // per-output-sample low-band suppression ratio, parallel to fifo_out
    gain_scratch: Vec<f32>, // reused per-emit gain buffer (RT: no per-quantum alloc)
    hb_gain: f32,           // smoothed high-band reinjection gain (echo-only duck)
    // High-band linear AEC: cancels >8 kHz echo the
    // 16 kHz core never sees, from the loopback reference.
    ref_hist: Box<[f32]>, // raw 48 kHz reference ring (for the high-band reference)
    hb: HbAec,            // partitioned block frequency-domain adaptive filter
    hb_mic: Vec<f32>,     // accumulating high-band mic block (HB_BLOCK)
    hb_ref: Vec<f32>,     // accumulating high-band reference block
    hbout: Box<[f32]>,    // echo-cancelled high band, indexed by absolute sample count
    hb_t: usize,          // next absolute sample fed into the high-band filter
    hbout_ready: usize,   // absolute count up to which hbout is valid
    last_g: f32,          // legacy suppression heuristic, NOT a DTD
    previous_mic_energy: f64,
    hb_blk: Vec<f32>, // reused per-block output scratch
}

/// High-band filter block size and partition count (48 kHz): 64 partitions of
/// 256 samples ≈ 341 ms of echo-tail coverage.
const HB_BLOCK: usize = 256;
const HB_PARTS: usize = 64;

impl Engine {
    fn new(model: Model) -> Self {
        let streamer = Streamer::new(&model);
        // Reserve a complete hop before emitting audio. Starting empty inserts
        // zeros at callback boundaries until enough slack accumulates, making
        // latency depend on the callback size and desynchronizing the bands.
        let mut fifo_out = Vec::with_capacity(2 * MAX_CHUNK);
        fifo_out.resize(MAX_CHUNK, 0.0);
        let mut fifo_gain = Vec::with_capacity(2 * MAX_CHUNK);
        fifo_gain.resize(MAX_CHUNK, 1.0);
        Self {
            reference_bypass: None,
            model,
            streamer,
            down_mic: Down3::with_capacity(MAX_CHUNK),
            down_ref: Down3::with_capacity(MAX_CHUNK),
            up_out: Up3::with_capacity(Streamer::HOP),
            fifo_mic: Vec::with_capacity(2 * Streamer::HOP),
            fifo_ref: Vec::with_capacity(2 * Streamer::HOP),
            fifo_out,
            lpf: aec_gtcrn::resample::lpf_prototype_48k(),
            mic_hist: vec![0.0; MIC_RING].into_boxed_slice(),
            in_count: 0,
            out_count: 0,
            rs: Vec::with_capacity(MAX_CHUNK),
            fifo_gain,
            gain_scratch: Vec::with_capacity(MAX_CHUNK),
            hb_gain: 1.0,
            ref_hist: vec![0.0; MIC_RING].into_boxed_slice(),
            hb: HbAec::new(HB_BLOCK, HB_PARTS),
            hb_mic: Vec::with_capacity(HB_BLOCK),
            hb_ref: Vec::with_capacity(HB_BLOCK),
            hbout: vec![0.0; MIC_RING].into_boxed_slice(),
            hb_t: 0,
            hbout_ready: 0,
            last_g: 1.0,
            previous_mic_energy: 0.0,
            hb_blk: vec![0.0; HB_BLOCK],
        }
    }

    /// High-band (>~8 kHz) of `hist[bi]` = the sample minus its zero-phase
    /// low-pass. A neural low branch does NOT guarantee perfect reconstruction. Reads
    /// forward to `bi + LPF_DELAY`, so the caller must have that much history.
    fn band_hi(&self, hist: &[f32], bi: i64) -> f32 {
        let d = aec_gtcrn::resample::LPF_DELAY as i64;
        if bi < d {
            return 0.0; // startup: not enough history yet
        }
        let taps = self.lpf.len();
        let mut lp = 0.0;
        for k in 0..taps {
            lp += self.lpf[k] * hist[((bi + d - k as i64) as usize) % MIC_RING];
        }
        hist[bi as usize % MIC_RING] - lp
    }

    /// Run the high-band linear AEC over every sample whose low-pass window is
    /// fully available, filling `hbout` with the echo-cancelled high band.
    /// Legacy mode uses suppression as a heuristic; the optional aligned guard
    /// ignores that permission. Neither mode has a double-talk safety proof.
    fn feed_high_band(&mut self) {
        let d = aec_gtcrn::resample::LPF_DELAY;
        while (self.hb_t + d) < self.in_count {
            let t = self.hb_t as i64;
            self.hb_mic.push(self.band_hi(&self.mic_hist, t));
            self.hb_ref.push(self.band_hi(&self.ref_hist, t));
            self.hb_t += 1;
            if self.hb_mic.len() == HB_BLOCK {
                let adapt = self.last_g < 0.5;
                let (mic, refb) = (
                    std::mem::take(&mut self.hb_mic),
                    std::mem::take(&mut self.hb_ref),
                );
                self.hb.process_block(&mic, &refb, adapt, &mut self.hb_blk);
                let start = self.hb_t - HB_BLOCK;
                for (i, &v) in self.hb_blk.iter().enumerate() {
                    self.hbout[(start + i) % MIC_RING] = v;
                }
                self.hbout_ready = self.hb_t;
                self.hb_mic = mic;
                self.hb_ref = refb;
                self.hb_mic.clear();
                self.hb_ref.clear();
            }
        }
    }

    fn run(&mut self, rec: &[f32], play: &[f32], out: &mut [f32]) {
        assert_eq!(rec.len(), play.len());
        assert_eq!(rec.len(), out.len());
        // Never fill more history than the consumer can process before wrap.
        // Ordinary host blocks <=768 keep their previous arithmetic order.
        for ((m, r), o) in rec
            .chunks(MAX_CHUNK)
            .zip(play.chunks(MAX_CHUNK))
            .zip(out.chunks_mut(MAX_CHUNK))
        {
            self.run_chunk(m, r, o);
        }
    }

    fn run_chunk(&mut self, rec: &[f32], play: &[f32], out: &mut [f32]) {
        if self.streamer.is_faulted() || self.hb.is_faulted() {
            out.fill(0.0);
            return;
        }
        for (&m, &r) in rec.iter().zip(play.iter()) {
            if let Some(bypass) = &mut self.reference_bypass {
                bypass.observe(r, self.in_count);
            }
            self.mic_hist[self.in_count % MIC_RING] = if m.is_finite() { m } else { 0.0 };
            self.ref_hist[self.in_count % MIC_RING] = if r.is_finite() { r } else { 0.0 };
            self.in_count += 1;
        }
        self.down_mic.process(rec, &mut self.rs);
        self.fifo_mic.extend_from_slice(&self.rs);
        self.down_ref.process(play, &mut self.rs);
        self.fifo_ref.extend_from_slice(&self.rs);
        let hop = Streamer::HOP;
        let mut consumed = 0;
        let ready = self.fifo_mic.len().min(self.fifo_ref.len());
        while consumed + hop <= ready {
            let m = &self.fifo_mic[consumed..consumed + hop];
            let r = &self.fifo_ref[consumed..consumed + hop];
            let mut o16 = self.streamer.process_hop(&self.model, m, r);
            // Numerically bounded legacy duck signal. Input/output frames need
            // not refer to the same acoustic instant; never call this a DTD.
            let energy =
                |samples: &[f32]| samples.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            let me = energy(m);
            let oe = energy(&o16);
            // The STFT output includes the previous microphone hop. Keep that
            // hop's energy in the ceiling so a speech offset is not cut short.
            // Apply the ceiling at every level, including digital silence.
            const MAX_GROWTH: f64 = 4.0; // energy ratio, i.e. 2x amplitude
            let ceiling = me.max(self.previous_mic_energy) * MAX_GROWTH;
            self.previous_mic_energy = me;
            let oe = if oe > ceiling {
                let s = (ceiling / oe).sqrt() as f32;
                for v in &mut o16 {
                    *v *= s;
                }
                ceiling
            } else {
                oe
            };
            let g = if me > 1e-9 {
                (oe / me).sqrt().clamp(0.0, 50.0) as f32
            } else {
                1.0
            };
            self.last_g = g;
            self.up_out.process(&o16, &mut self.rs);
            self.fifo_out.extend_from_slice(&self.rs);
            self.fifo_gain.resize(self.fifo_out.len(), g);
            consumed += hop;
        }
        if consumed > 0 {
            self.fifo_mic.drain(..consumed);
            self.fifo_ref.drain(..consumed);
        }
        // Cancel the high-band echo linearly from the reference.
        self.feed_high_band();
        // emit out.len() samples; zero-fill during the startup fill
        let n = out.len();
        let avail = self.fifo_out.len().min(n);
        out[..avail].copy_from_slice(&self.fifo_out[..avail]);
        out[avail..].fill(0.0);
        // Gains aligned with the emitted samples (passthrough where zero-filled).
        // Reuse the scratch buffer so the RT path allocates nothing after warmup.
        self.gain_scratch.clear();
        self.gain_scratch.resize(n, 1.0);
        self.gain_scratch[..avail].copy_from_slice(&self.fifo_gain[..avail]);
        self.fifo_out.drain(..avail);
        self.fifo_gain.drain(..avail);
        // Add the delayed mic high band (output sample p aligns with mic[p - latency]),
        // ducked when the low band was strongly suppressed — i.e. confident far-end
        // echo — so the uncancellable >8 kHz echo does not pass through raw, while
        // near-end and double-talk (low band retained → g high) keep the band intact
        //. The one-pole smoothing avoids zipper on g transitions.
        const G_LO: f32 = 0.15; // full duck at/below this low-band ratio
        const G_HI: f32 = 0.45; // full passthrough at/above
        const SMOOTH: f32 = 0.0021; // ~10 ms one-pole at 48 kHz
        let mut hb = self.hb_gain;
        for (j, o) in out.iter_mut().enumerate() {
            let target = ((self.gain_scratch[j] - G_LO) / (G_HI - G_LO)).clamp(0.0, 1.0);
            hb += (target - hb) * SMOOTH;
            let idx = (self.out_count + j) as i64 - AEC_LATENCY_48K as i64;
            // Prefer the linearly echo-cancelled high band; fall back to the raw
            // high band only until the filter has produced that sample (startup).
            let hi = if idx >= 0 && (idx as usize) < self.hbout_ready {
                self.hbout[idx as usize % MIC_RING]
            } else {
                self.band_hi(&self.mic_hist, idx)
            };
            *o += hb * hi;
        }
        self.hb_gain = if hb.is_finite() { hb } else { 1.0 };
        if self.streamer.is_faulted() || self.hb.is_faulted() {
            out.fill(0.0); // never reinject raw high band around a faulted core
        } else {
            for (j, sample) in out.iter_mut().enumerate() {
                // Full-band dry path uses EXACTLY the latency of the wet path.
                // Faults above still fail closed; bypass cannot mask them.
                if let Some(index) = (self.out_count + j).checked_sub(AEC_LATENCY_48K)
                    && let Some(bypass) = &mut self.reference_bypass
                {
                    *sample = bypass.mix(*sample, self.mic_hist[index % MIC_RING], index);
                }
                if !sample.is_finite() {
                    *sample = 0.0;
                }
            }
        }
        self.out_count += n;
    }
}

#[repr(C)]
struct Handle {
    handle: spa_handle,
    aec: spa_audio_aec,
    engine: *mut Engine, // null until init/init2
}

// ── method impls ──────────────────────────────────────────────────────────

fn dict_get(args: *const spa_dict, key: &str) -> Option<String> {
    if args.is_null() {
        return None;
    }
    // SAFETY: PipeWire passes a valid spa_dict or null.
    let d = unsafe { &*args };
    for i in 0..d.n_items as isize {
        let it = unsafe { &*d.items.offset(i) };
        if it.key.is_null() {
            continue;
        }
        let k = unsafe { CStr::from_ptr(it.key) }.to_string_lossy();
        if k == key && !it.value.is_null() {
            return Some(
                unsafe { CStr::from_ptr(it.value) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    None
}

fn model_path(args: *const spa_dict) -> String {
    dict_get(args, "gtcrn.model")
        .or_else(|| std::env::var("AEC_GTCRN_MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

unsafe fn engine_from(object: *mut c_void) -> Option<*mut Engine> {
    let h = object as *mut Handle;
    if h.is_null() || (*h).engine.is_null() {
        None
    } else {
        Some((*h).engine)
    }
}

unsafe fn do_init(object: *mut c_void, args: *const spa_dict) -> c_int {
    let h = object as *mut Handle;
    if h.is_null() {
        return -22; // -EINVAL
    }
    if !(*h).engine.is_null() {
        return 0;
    }
    match Model::load(&model_path(args)) {
        Ok(m) => {
            let mut engine = Engine::new(m);
            // Initialization only: no spawn/join or allocation in m_run.
            let flag = |name| match dict_get(args, name).as_deref() {
                None | Some("false" | "0") => Ok(false),
                Some("true" | "1") => Ok(true),
                _ => Err(()),
            };
            let (Ok(async_delay), Ok(track_delay), Ok(hb_guard), Ok(reference_bypass)) = (
                flag("gtcrn.async-delay"),
                flag("gtcrn.track-delay"),
                flag("gtcrn.hb-adaptation-guard"),
                flag("gtcrn.reference-silence-bypass"),
            ) else {
                return -22;
            };
            if reference_bypass {
                engine.reference_bypass = Some(ReferenceBypass::new(MIC_RING));
            }
            let result = if track_delay {
                engine.streamer.enable_delay_tracking()
            } else if async_delay {
                engine.streamer.enable_async_delay()
            } else {
                Ok(())
            };
            if result.is_err() {
                return -11;
            }
            if hb_guard {
                engine.hb.enable_adaptation_guard();
            }
            (*h).engine = Box::into_raw(Box::new(engine));
            0
        }
        Err(_) => -2, // -ENOENT
    }
}

unsafe fn valid_format(info: *const spa_audio_info_raw) -> bool {
    !info.is_null()
        && (*info).format == SPA_AUDIO_FORMAT_F32P
        && (*info).rate == 48_000
        && (*info).channels == 1
}

unsafe extern "C" fn m_init(
    object: *mut c_void,
    args: *const spa_dict,
    info: *const spa_audio_info_raw,
) -> c_int {
    if !valid_format(info) {
        return -22;
    }
    do_init(object, args)
}
unsafe extern "C" fn m_init2(
    object: *mut c_void,
    args: *const spa_dict,
    first: *mut spa_audio_info_raw,
    second: *mut spa_audio_info_raw,
    third: *mut spa_audio_info_raw,
) -> c_int {
    // The header and module-echo-cancel disagree about parameter *names* in
    // current upstream. All three streams have the same supported contract;
    // validate all of them, without relying on their names/order.
    if !valid_format(first) || !valid_format(second) || !valid_format(third) {
        return -22;
    }
    do_init(object, args)
}
unsafe extern "C" fn m_activate(_object: *mut c_void) -> c_int {
    0
}
unsafe extern "C" fn m_deactivate(_object: *mut c_void) -> c_int {
    0
}

unsafe extern "C" fn m_run(
    object: *mut c_void,
    rec: *const *const f32,
    play: *const *const f32,
    out: *mut *mut f32,
    n: u32,
) -> c_int {
    let Some(engine) = engine_from(object) else {
        return -22;
    };
    // No buffers are accessed for a zero-size callback.
    if n == 0 {
        return 0;
    }
    if rec.is_null() || play.is_null() || out.is_null() {
        return -22;
    }
    let (rec0, play0, out0) = (*rec, *play, *out);
    let n = n as usize;
    if !valid_audio_span(rec0, n)
        || !valid_audio_span(play0, n)
        || !valid_audio_span(out0, n)
        || partial_overlap(rec0, out0, n)
        || partial_overlap(play0, out0, n)
    {
        return -22;
    }
    // The normal PipeWire host supplies disjoint buffers. Staging also permits
    // exact in-place operation, without ever making overlapping Rust slices.
    // Partial overlap is rejected before writing: it could overwrite a future
    // chunk's unread input. Pointer validity/length remain the C caller's duty.
    let eng = &mut *engine; // lifetime scoped to this serialized SPA call
    let mut mic = [0.0f32; MAX_CHUNK];
    let mut render = [0.0f32; MAX_CHUNK];
    let mut result = [0.0f32; MAX_CHUNK];
    let mut offset = 0;
    while offset < n {
        let count = (n - offset).min(MAX_CHUNK);
        for i in 0..count {
            let m = rec0.add(offset + i).read();
            let r = play0.add(offset + i).read();
            mic[i] = if m.is_finite() { m } else { 0.0 };
            render[i] = if r.is_finite() { r } else { 0.0 };
        }
        eng.run(&mic[..count], &render[..count], &mut result[..count]);
        for (i, &sample) in result[..count].iter().enumerate() {
            out0.add(offset + i)
                .write(if sample.is_finite() { sample } else { 0.0 });
        }
        offset += count;
    }
    0
}

fn valid_audio_span(p: *const f32, n: usize) -> bool {
    !p.is_null()
        && (p as usize).is_multiple_of(std::mem::align_of::<f32>())
        && n.checked_mul(std::mem::size_of::<f32>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .and_then(|bytes| (p as usize).checked_add(bytes))
            .is_some()
}

fn partial_overlap(input: *const f32, output: *const f32, n: usize) -> bool {
    if input == output {
        return false;
    }
    // valid_audio_span has already established non-overflowing ranges.
    let bytes = n * std::mem::size_of::<f32>();
    let (a, b) = (input as usize, output as usize);
    a < b + bytes && b < a + bytes
}

static METHODS: SyncWrap<spa_audio_aec_methods> = SyncWrap(spa_audio_aec_methods {
    version: 3,
    add_listener: None,
    init: Some(m_init),
    run: Some(m_run),
    set_props: None,
    activate: Some(m_activate),
    deactivate: Some(m_deactivate),
    enum_props: None,
    get_params: None,
    set_params: None,
    init2: Some(m_init2),
});

unsafe extern "C" fn h_get_interface(
    handle: *mut spa_handle,
    type_: *const c_char,
    iface: *mut *mut c_void,
) -> c_int {
    if handle.is_null() || type_.is_null() || iface.is_null() {
        return -22;
    }
    *iface = ptr::null_mut();
    let want = CStr::from_ptr(type_).to_bytes_with_nul();
    if want != AEC_TYPE {
        return -95; // -ENOTSUP, as specified by spa_handle.get_interface
    }
    let h = handle as *mut Handle;
    *iface = &mut (*h).aec as *mut spa_audio_aec as *mut c_void;
    0
}

unsafe extern "C" fn h_clear(handle: *mut spa_handle) -> c_int {
    let h = handle as *mut Handle;
    if !h.is_null() && !(*h).engine.is_null() {
        drop(Box::from_raw((*h).engine));
        (*h).engine = ptr::null_mut();
    }
    0
}

unsafe extern "C" fn f_get_size(_f: *const spa_handle_factory, _p: *const spa_dict) -> usize {
    std::mem::size_of::<Handle>()
}

unsafe extern "C" fn f_init(
    _f: *const spa_handle_factory,
    handle: *mut spa_handle,
    _info: *const spa_dict,
    _support: *const c_void,
    _n: u32,
) -> c_int {
    if handle.is_null() {
        return -22;
    }
    let h = handle as *mut Handle;
    ptr::write(
        h,
        Handle {
            handle: spa_handle {
                version: 0,
                get_interface: Some(h_get_interface),
                clear: Some(h_clear),
            },
            aec: spa_audio_aec {
                iface: spa_interface {
                    type_: AEC_TYPE.as_ptr() as *const c_char,
                    version: 1,
                    cb: spa_callbacks {
                        funcs: &METHODS.0 as *const spa_audio_aec_methods as *const c_void,
                        data: h as *mut c_void, // methods receive this as `object`
                    },
                },
                name: AEC_NAME.as_ptr().cast(),
                info: ptr::null(),
                latency: AEC_BLOCK.as_ptr().cast(),
            },
            engine: ptr::null_mut(),
        },
    );
    0
}

static AEC_IFACE_INFO: SyncWrap<spa_interface_info> = SyncWrap(spa_interface_info {
    type_: AEC_TYPE.as_ptr() as *const c_char,
});

unsafe extern "C" fn f_enum_interface_info(
    _f: *const spa_handle_factory,
    info: *mut *const spa_interface_info,
    index: *mut u32,
) -> c_int {
    if info.is_null() || index.is_null() {
        return -22;
    }
    if *index == 0 {
        *info = &AEC_IFACE_INFO.0;
        *index += 1;
        1
    } else {
        0
    }
}

static FACTORY: SyncWrap<spa_handle_factory> = SyncWrap(spa_handle_factory {
    version: 1,
    name: FACTORY_NAME.as_ptr() as *const c_char,
    info: ptr::null(),
    get_size: Some(f_get_size),
    init: Some(f_init),
    enum_interface_info: Some(f_enum_interface_info),
});

/// SPA plugin entry point: enumerate this plugin's handle factories.
///
/// # Safety
/// `factory` and `index` must be valid non-null pointers per the SPA
/// `spa_handle_factory_enum` contract. Called by the PipeWire loader.
#[unsafe(no_mangle)]
#[allow(private_interfaces)]
pub unsafe extern "C" fn spa_handle_factory_enum(
    factory: *mut *const spa_handle_factory,
    index: *mut u32,
) -> c_int {
    if factory.is_null() || index.is_null() {
        return -22;
    }
    if *index == 0 {
        *factory = &FACTORY.0;
        *index += 1;
        1
    } else {
        0
    }
}

#[cfg(test)]
mod split_band_tests {
    use super::Engine;
    use aec_gtcrn::Model;

    fn model_or_skip() -> Model {
        model().expect("AEC_GTCRN_GGUF")
    }
    fn model() -> Option<Model> {
        let p = std::env::var("AEC_GTCRN_GGUF")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())?;
        Model::load(&p).ok()
    }
    fn drive(eng: &mut Engine, mic: &[f32]) -> Vec<f32> {
        let zero = vec![0.0f32; 480];
        let mut out = vec![0.0f32; mic.len()];
        let mut o = 0;
        while o + 480 <= mic.len() {
            let mut ob = [0.0f32; 480];
            eng.run(&mic[o..o + 480], &zero, &mut ob);
            out[o..o + 480].copy_from_slice(&ob);
            o += 480;
        }
        out
    }

    #[test]
    fn output_does_not_depend_on_host_block_size() {
        let Some(m) = model() else { return };
        let mic: Vec<f32> = (0..48_000)
            .map(|i| 0.1 * (i as f32 * 0.037).sin() + 0.03 * (i as f32 * 1.7).sin())
            .collect();
        let reference = vec![0.0; mic.len()];
        let render = |mut engine: Engine, quantum: usize| {
            let mut out = vec![0.0; mic.len()];
            for ((rec, play), out) in mic
                .chunks(quantum)
                .zip(reference.chunks(quantum))
                .zip(out.chunks_mut(quantum))
            {
                engine.run(rec, play, out);
            }
            out
        };
        let expected = render(Engine::new(m), 768);
        for quantum in [1, 128, 256, 480, 1024] {
            let actual = render(Engine::new(model_or_skip()), quantum);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 1e-6, "quantum {quantum}: max sample error {error}");
        }
    }

    #[test]
    fn silent_microphone_does_not_emit_the_predicted_echo() {
        let Some(m) = model() else { return };
        let mut engine = Engine::new(m);
        let mut seed = 17_u32;
        let reference: Vec<f32> = (0..3 * 48_000)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                0.2 * ((seed >> 9) as f32 / 8_388_608.0 - 1.0)
            })
            .collect();
        let mic: Vec<f32> = reference
            .iter()
            .enumerate()
            .map(|(i, &r)| if i < 48_000 { 0.4 * r } else { 0.0 })
            .collect();
        let out = drive_ref(&mut engine, &mic, &reference);
        let peak = out[2 * 48_000..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        assert!(peak < 1e-7, "silent microphone emitted echo: peak {peak}");
    }

    // hi_band must be the mic's >8 kHz complement: ~0 for an in-band tone, ~full
    // for a tone above the 8 kHz ceiling. Isolates the split filter from the AEC.
    fn hiband_rms(freq: f32) -> f32 {
        let sr = 48000usize;
        let mut eng = Engine::new(model_or_skip());
        let mic: Vec<f32> = (0..sr)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let (s, e) = (sr / 2, 9 * sr / 10);
        let mut acc = 0.0f32;
        let delay = aec_gtcrn::resample::LPF_DELAY;
        for (i, &sample) in mic.iter().enumerate() {
            eng.mic_hist[eng.in_count % super::MIC_RING] = sample;
            eng.in_count += 1;
            // Consume while the FIR window is still present in the ring.
            if i >= delay {
                let bi = i - delay;
                if (s..e).contains(&bi) {
                    let value = eng.band_hi(&eng.mic_hist, bi as i64);
                    acc += value * value;
                }
            }
        }
        (acc / (e - s) as f32).sqrt()
    }

    #[test]
    fn hi_band_is_the_complement() {
        if std::env::var("AEC_GTCRN_GGUF")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
            .is_none()
        {
            eprintln!("skip");
            return;
        }
        let inrms = 0.2 / 2f32.sqrt();
        let lo = hiband_rms(1000.0);
        let hi = hiband_rms(12000.0);
        assert!(lo < 0.05 * inrms, "1 kHz leaked into hi band: {lo:.4}");
        assert!(hi > 0.9 * inrms, "12 kHz missing from hi band: {hi:.4}");
    }

    // Split `x` into (low, high) band energy using the same LPF the plugin uses
    // for hi_band, so the measurement matches the deployed split point.
    fn band_energies(x: &[f32]) -> (f64, f64) {
        let lpf = aec_gtcrn::resample::lpf_prototype_48k();
        let d = aec_gtcrn::resample::LPF_DELAY;
        let taps = lpf.len();
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for bi in taps..x.len().saturating_sub(d) {
            let mut low = 0.0f32;
            for k in 0..taps {
                low += lpf[k] * x[bi + d - k];
            }
            let high = x[bi] - low;
            lo += f64::from(low) * f64::from(low);
            hi += f64::from(high) * f64::from(high);
        }
        (lo, hi)
    }

    fn drive_ref(eng: &mut Engine, mic: &[f32], far: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; mic.len()];
        let mut o = 0;
        while o + 480 <= mic.len() {
            let mut ob = [0.0f32; 480];
            eng.run(&mic[o..o + 480], &far[o..o + 480], &mut ob);
            out[o..o + 480].copy_from_slice(&ob);
            o += 480;
        }
        out
    }

    // High-band mitigation: the core cancels only <=8 kHz, but the high-band
    // reinjection is now ducked when the low band is confidently cancelled
    // (far-end echo). On a far-end-only echo both bands must therefore be
    // suppressed — the >8 kHz echo no longer passes through raw.
    #[test]
    fn residual_echo_is_cancelled_below_8k_but_not_above() {
        let Some(m) = model() else {
            eprintln!("skip");
            return;
        };
        let mut eng = Engine::new(m);
        let sr = 48000usize;
        let n = 4 * sr;
        // Broadband far-end: tones in both bands + light noise.
        let mut seed = 99u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / 8_388_608.0 - 1.0
        };
        let far: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let tau = 2.0 * std::f32::consts::PI * t;
                0.12 * ((tau * 500.0).sin()
                    + (tau * 3000.0).sin()
                    + (tau * 11000.0).sin()
                    + (tau * 14000.0).sin())
                    + 0.02 * rng()
            })
            .collect();
        // Echo into the mic: attenuated, delayed copy of the far-end, no near-end.
        let delay = 240;
        let mic: Vec<f32> = (0..n)
            .map(|i| {
                if i >= delay {
                    0.4 * far[i - delay]
                } else {
                    0.0
                }
            })
            .collect();

        let out = drive_ref(&mut eng, &mic, &far);
        // Measure the converged tail only.
        let tail = 2 * sr;
        let (mic_lo, mic_hi) = band_energies(&mic[tail..]);
        let (out_lo, out_hi) = band_energies(&out[tail..]);
        let erle = |i: f64, o: f64| 10.0 * (i / o.max(1e-12)).log10();
        let erle_lo = erle(mic_lo, out_lo);
        let erle_hi = erle(mic_hi, out_hi);
        eprintln!("per-band ERLE: <8kHz = {erle_lo:.1} dB, >8kHz = {erle_hi:.1} dB");
        // Both bands are now cancelled on a far-end-only echo: the low band by the
        // core, the high band by the suppression-coupled duck. The
        // >8 kHz echo used to pass through at ~0 dB; the duck must cut it clearly.
        assert!(erle_lo > 12.0, "low band not cancelled: {erle_lo:.1} dB");
        assert!(
            erle_hi > 12.0,
            ">8kHz echo not ducked during far-end-only: {erle_hi:.1} dB"
        );
    }

    /// Minimal mono 32-bit-float WAV reader (the testset's format), stdlib only:
    /// walk the RIFF chunks to the `data` block and reinterpret it as f32 LE.
    fn read_wav_f32(path: &std::path::Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        let mut i = 12; // skip "RIFF"<size>"WAVE"
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            let body_end = (i + 8 + sz).min(bytes.len());
            if id == b"data" {
                return Some(
                    bytes[i + 8..body_end]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            }
            i += 8 + sz + (sz & 1);
        }
        None
    }

    /// Corpus baseline on real speech (Microsoft-AEC-style scenarios). Pure
    /// measurement of the *current* raw-reinjection code — no behaviour change —
    /// so the full-band decision rests on real-speech per-band numbers, not only
    /// the synthetic-tone measurement above. Point `AEC_TESTSET` at a 48 kHz
    /// scenario root (default: the local aec-eval testset); skips when the model
    /// or the corpus is absent.
    /// (mic, ref) wav paths in a scenario dir: a single `mic.wav`/`ref.wav`
    /// (the local synthetic testset), or every `*_mic.wav` paired with its
    /// `*_ref.wav` (a multi-clip real corpus like the MS blind test set).
    /// Write a mono 32-bit-float WAV (RIFF, IEEE-float fmt) for AECMOS/soundfile.
    fn write_wav_f32(path: &std::path::Path, x: &[f32], sr: u32) {
        let mut b = Vec::with_capacity(44 + x.len() * 4);
        let data_len = (x.len() * 4) as u32;
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data_len).to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        b.extend_from_slice(&1u16.to_le_bytes()); // mono
        b.extend_from_slice(&sr.to_le_bytes());
        b.extend_from_slice(&(sr * 4).to_le_bytes()); // byte rate
        b.extend_from_slice(&4u16.to_le_bytes()); // block align
        b.extend_from_slice(&32u16.to_le_bytes()); // bits
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data_len.to_le_bytes());
        for &s in x {
            b.extend_from_slice(&s.to_le_bytes());
        }
        let _ = std::fs::write(path, b);
    }

    /// Render the Engine's enhanced output for every `*_mic.wav` in `AEC_RENDER`
    /// (ref = the sibling `*_ref.wav` or `*_farend.wav`), writing `*_enh.wav`
    /// next to it — the input to an external perceptual scorer (AECMOS).
    #[ignore = "renderer: needs AEC_GTCRN_GGUF + AEC_RENDER dir of *_mic/*_ref|farend"]
    #[test]
    fn aec01_render() {
        let Some(dir) = std::env::var("AEC_RENDER").ok() else {
            eprintln!("AEC_RENDER unset, skip");
            return;
        };
        let dir = std::path::Path::new(&dir);
        if model().is_none() {
            eprintln!("model absent, skip");
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut mics: Vec<String> = entries
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .filter(|n| n.ends_with("_mic.wav"))
            .collect();
        mics.sort();
        let mut done = 0;
        for mic_name in mics {
            let id = &mic_name[..mic_name.len() - "_mic.wav".len()];
            let ref_ref = dir.join(format!("{id}_ref.wav"));
            let ref_far = dir.join(format!("{id}_farend.wav"));
            let refp = if ref_ref.is_file() { ref_ref } else { ref_far };
            let (Some(mic), Some(far)) = (read_wav_f32(&dir.join(&mic_name)), read_wav_f32(&refp))
            else {
                continue;
            };
            let Some(m) = model() else { return };
            let n = mic.len().min(far.len());
            let out = drive_ref(&mut Engine::new(m), &mic[..n], &far[..n]);
            write_wav_f32(&dir.join(format!("{id}_enh.wav")), &out, 48000);
            done += 1;
        }
        eprintln!("rendered {done} enhanced clips into {}", dir.display());
    }

    fn clip_pairs(dir: &std::path::Path) -> Vec<(std::path::PathBuf, std::path::PathBuf)> {
        let single = dir.join("mic.wav");
        if single.is_file() {
            return vec![(single, dir.join("ref.wav"))];
        }
        let mut pairs = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return pairs;
        };
        let mut mics: Vec<String> = entries
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .filter(|n| n.ends_with("_mic.wav"))
            .collect();
        mics.sort();
        for mic in mics {
            let refn = format!("{}_ref.wav", &mic[..mic.len() - "_mic.wav".len()]);
            if dir.join(&refn).is_file() {
                pairs.push((dir.join(&mic), dir.join(&refn)));
            }
        }
        pairs
    }

    /// Sum (mic, out) low/high band energies over every clip in a scenario,
    /// each through a fresh Engine, measuring only the converged second half.
    fn measure_scenario(dir: &std::path::Path) -> Option<(f64, f64, f64, f64)> {
        let (mut mlo, mut mhi, mut olo, mut ohi) = (0.0, 0.0, 0.0, 0.0);
        let mut clips = 0;
        for (mic_p, ref_p) in clip_pairs(dir) {
            let (Some(mic), Some(far)) = (read_wav_f32(&mic_p), read_wav_f32(&ref_p)) else {
                continue;
            };
            // mic and reference clips can differ in length; drive_ref indexes both.
            let n = mic.len().min(far.len());
            let (mic, far) = (&mic[..n], &far[..n]);
            let m = model()?;
            let out = drive_ref(&mut Engine::new(m), mic, far);
            let tail = mic.len() / 2;
            let (a, b) = band_energies(&mic[tail..]);
            let (c, d) = band_energies(&out[tail..]);
            mlo += a;
            mhi += b;
            olo += c;
            ohi += d;
            clips += 1;
        }
        (clips > 0).then_some((mlo, mhi, olo, ohi))
    }

    #[ignore = "measurement harness: needs AEC_GTCRN_GGUF + a 48 kHz AEC testset"]
    #[test]
    fn aec01_corpus_baseline() {
        let Some(home) = std::env::var("HOME").ok() else {
            return;
        };
        let root = std::env::var("AEC_TESTSET")
            .unwrap_or_else(|_| format!("{home}/.cache/aec-eval/testset/48000"));
        let root = std::path::Path::new(&root);
        if model().is_none() {
            eprintln!("corpus: model absent, skip");
            return;
        }
        let erle = |i: f64, o: f64| 10.0 * (i / o.max(1e-12)).log10();

        // Far-end only: the mic is pure echo, so any output energy is residual echo.
        if let Some((mlo, mhi, olo, ohi)) = measure_scenario(&root.join("farend_only")) {
            eprintln!(
                "corpus farend ERLE: <8kHz {:.1} dB, >8kHz {:.1} dB",
                erle(mlo, olo),
                erle(mhi, ohi)
            );
        } else {
            eprintln!("corpus: farend_only absent, skip");
        }

        // Near-end only: no echo present, so the >8 kHz voice band must survive to
        // the output — this is what any high-band echo duck must not damage.
        if let Some((nmlo, nmhi, nolo, nohi)) = measure_scenario(&root.join("nearend_only")) {
            eprintln!(
                "corpus nearend preservation: <8kHz {:.2}x, >8kHz {:.2}x",
                (nolo / nmlo.max(1e-12)).sqrt(),
                (nohi / nmhi.max(1e-12)).sqrt()
            );
        }
    }

    /// Delay that best aligns `est` (Engine output, latency-shifted) to `target`,
    /// by correlation over a mid window — so SI-SDR compares aligned signals.
    /// Searched in `[lo, hi)`; the Engine latency is deterministic (~1632), so a
    /// tight window avoids spurious lags that tank SI-SDR on some clips.
    fn best_shift(est: &[f32], target: &[f32], lo: usize, hi: usize) -> usize {
        let win = 40_000.min(target.len() / 2);
        let start = target.len() / 4;
        let (mut best_s, mut best_c) = (lo, f64::MIN);
        for s in lo..hi {
            let mut c = 0.0f64;
            for i in 0..win {
                let e = est[(start + i + s).min(est.len() - 1)];
                c += f64::from(e) * f64::from(target[start + i]);
            }
            if c > best_c {
                best_c = c;
                best_s = s;
            }
        }
        best_s
    }

    /// Scale-invariant SDR of `est` against `target` at a given alignment shift.
    fn si_sdr(est: &[f32], target: &[f32], shift: usize) -> f64 {
        let n = (est.len().saturating_sub(shift)).min(target.len());
        let (e, t) = (&est[shift..shift + n], &target[..n]);
        let (mut em, mut tm) = (0.0f64, 0.0f64);
        for i in 0..n {
            em += f64::from(e[i]);
            tm += f64::from(t[i]);
        }
        em /= n as f64;
        tm /= n as f64;
        let (mut dot, mut tt) = (0.0f64, 0.0f64);
        for i in 0..n {
            let (ev, tv) = (f64::from(e[i]) - em, f64::from(t[i]) - tm);
            dot += ev * tv;
            tt += tv * tv;
        }
        let alpha = dot / (tt + 1e-12);
        let (mut sp, mut np) = (0.0f64, 0.0f64);
        for i in 0..n {
            let (ev, tv) = (f64::from(e[i]) - em, f64::from(t[i]) - tm);
            let proj = alpha * tv;
            sp += proj * proj;
            np += (ev - proj) * (ev - proj);
        }
        10.0 * (sp / (np + 1e-12)).log10()
    }

    /// Intrusive A+B quality on the synthetic-fullband set (ground-truth clean
    /// near-end): SI-SDR of the Engine output vs the clean target, against the
    /// raw mic baseline. Point `AEC_SYNTH` at a dir of `*_mic.wav` / `*_farend.wav`
    /// / `*_target.wav` triplets. Answers how much near-end quality the AEC keeps
    /// while removing echo.
    #[ignore = "measurement harness: needs AEC_GTCRN_GGUF + a synthetic AEC triplet dir"]
    #[test]
    fn aec01_synth_sisdr() {
        let Some(home) = std::env::var("HOME").ok() else {
            return;
        };
        let dir = std::env::var("AEC_SYNTH")
            .unwrap_or_else(|_| format!("{home}/.cache/aec-eval/ms-synth"));
        let dir = std::path::Path::new(&dir);
        if model().is_none() {
            eprintln!("synth: model absent, skip");
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            eprintln!("synth: dir absent, skip");
            return;
        };
        let mut ids: Vec<String> = entries
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .filter_map(|n| n.strip_suffix("_mic.wav").map(str::to_owned))
            .collect();
        ids.sort();
        let (mut in_sdr, mut out_sdr, mut clips) = (0.0f64, 0.0f64, 0usize);
        for id in &ids {
            let mic = read_wav_f32(&dir.join(format!("{id}_mic.wav")));
            let far = read_wav_f32(&dir.join(format!("{id}_farend.wav")));
            let tgt = read_wav_f32(&dir.join(format!("{id}_target.wav")));
            let (Some(mic), Some(far), Some(tgt)) = (mic, far, tgt) else {
                continue;
            };
            let Some(m) = model() else { return };
            let n = mic.len().min(far.len());
            let out = drive_ref(&mut Engine::new(m), &mic[..n], &far[..n]);
            // Align the Engine output to the mic (strong full-signal correlation),
            // then score against the clean near-end at that same shift.
            let shift = best_shift(&out, &mic, 1500, 1800);
            in_sdr += si_sdr(&mic, &tgt, 0);
            out_sdr += si_sdr(&out, &tgt, shift);
            clips += 1;
        }
        if clips > 0 {
            eprintln!(
                "synth SI-SDR over {clips} clips: mic {:.2} dB -> A+B {:.2} dB (+{:.2} dB)",
                in_sdr / clips as f64,
                out_sdr / clips as f64,
                (out_sdr - in_sdr) / clips as f64
            );
        }
    }

    // A pure 12 kHz tone (> the 8 kHz AEC ceiling) must survive to the output via the
    // high-band bypass; without split-band the down/up path would erase it.
    #[test]
    fn high_band_survives() {
        let Some(m) = model() else {
            eprintln!("skip");
            return;
        };
        let mut eng = Engine::new(m);
        let sr = 48000usize;
        let mic: Vec<f32> = (0..sr)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 12000.0 * i as f32 / sr as f32).sin())
            .collect();
        let out = drive(&mut eng, &mic);
        let rms = |x: &[f32]| (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let half = sr / 2;
        let (mi, oi) = (rms(&mic[half..]), rms(&out[half..]));
        assert!(
            oi > 0.5 * mi,
            "12 kHz tone lost: out RMS {oi:.4} vs in {mi:.4}"
        );
    }

    // The high-band duck must fire only on far-end echo, never on near-end voice.
    // A signal with energy in both bands and no far-end reference is pure near-end:
    // the AEC cancels nothing (g~1), so the >8 kHz band must reach the output
    // (the mitigation must not dull near-end sibilance).
    #[test]
    fn near_end_high_band_is_preserved() {
        let Some(m) = model() else {
            eprintln!("skip");
            return;
        };
        let mut eng = Engine::new(m);
        let sr = 48000usize;
        let mic: Vec<f32> = (0..2 * sr)
            .map(|i| {
                let tau = 2.0 * std::f32::consts::PI * i as f32 / sr as f32;
                0.2 * (tau * 1000.0).sin() + 0.2 * (tau * 12000.0).sin()
            })
            .collect();
        let out = drive(&mut eng, &mic);
        let half = mic.len() / 2;
        let (mlo, mhi) = band_energies(&mic[half..]);
        let (olo, ohi) = band_energies(&out[half..]);
        let keep_hi = (ohi / mhi.max(1e-12)).sqrt();
        eprintln!(
            "near-end preservation: >8kHz {keep_hi:.2}x, <8kHz {:.2}x",
            (olo / mlo.max(1e-12)).sqrt()
        );
        assert!(keep_hi > 0.8, "near-end >8 kHz was ducked: {keep_hi:.2}x");
    }
}

#[cfg(test)]
mod abi_contract_tests {
    use super::*;
    #[test]
    fn rejects_unsupported_streams_before_loading_a_model() {
        let mut info = spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32P,
            flags: 0,
            rate: 48_000,
            channels: 1,
            position: [0; 64],
        };
        unsafe {
            assert!(valid_format(&info));
            assert!(!valid_format(ptr::null()));
            info.rate = 44_100;
            assert_eq!(m_init(ptr::null_mut(), ptr::null(), &info), -22);
            info.rate = 48_000;
            info.channels = 2;
            assert!(!valid_format(&info));
            info.channels = 1;
            info.format = 0;
            assert!(!valid_format(&info));
        }
    }
}

// RT-02 gate: the split-band Engine (resamplers + FIFOs + GTCRN core) must not
// touch the heap per quantum on the PipeWire RT thread after warmup.
#[cfg(test)]
mod rt_alloc_tests {
    use super::Engine;
    use aec_gtcrn::Model;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
        static ALLOCS: Cell<usize> = const { Cell::new(0) };
        static DEALLOCS: Cell<usize> = const { Cell::new(0) };
    }
    struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if ARMED.try_with(Cell::get).unwrap_or(false) {
                let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
            }
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if ARMED.try_with(Cell::get).unwrap_or(false) {
                let _ = DEALLOCS.try_with(|c| c.set(c.get() + 1));
            }
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            if ARMED.try_with(Cell::get).unwrap_or(false) {
                let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
            }
            unsafe { System.realloc(p, l, n) }
        }
    }
    #[global_allocator]
    static GA: Counting = Counting;

    #[test]
    fn engine_run_is_alloc_free_after_warmup() {
        let Some(model) = std::env::var("AEC_GTCRN_GGUF")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
            .and_then(|p| Model::load(&p).ok())
        else {
            eprintln!("AEC_GTCRN_GGUF unset — skipping spa RT alloc gate");
            return;
        };
        let mut eng = Engine::new(model);
        let q = 480usize;
        let mic: Vec<f32> = (0..q * 300)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin())
            .collect();
        let zero = vec![0.0f32; q];
        let mut ob = vec![0.0f32; q];
        // Warm past the FIFO/pool/GCC-PHAT fill so steady-state capacities are set.
        let warm = 200;
        let mut o = 0;
        for _ in 0..warm {
            eng.run(&mic[o..o + q], &zero, &mut ob);
            o += q;
        }
        ALLOCS.with(|a| a.set(0));
        DEALLOCS.with(|a| a.set(0));
        ARMED.with(|a| a.set(true));
        for _ in 0..80 {
            eng.run(&mic[o..o + q], &zero, &mut ob);
            o += q;
        }
        ARMED.with(|a| a.set(false));
        assert_eq!(
            ALLOCS.with(Cell::get),
            0,
            "Engine::run allocated on the RT path"
        );
        assert_eq!(
            DEALLOCS.with(Cell::get),
            0,
            "Engine::run freed memory on RT"
        );
    }
    // This is an explicit release gate, NOT a passing/soft-skipped test. Model
    // absence is a failure when requested. Per-instance scratch must pass this
    // on the actual model/toolchain; static preparation is not a test result.
    #[test]
    #[ignore = "release gate: requires the distributed model; execute explicitly"]
    fn first_callback_on_fresh_thread_has_no_heap_activity() {
        let path = std::env::var("AEC_GTCRN_GGUF").expect("required model fixture");
        let mut eng = Engine::new(Model::load(&path).expect("valid model fixture"));
        let (allocs, frees) = std::thread::spawn(move || {
            let mic = [0.05; super::MAX_CHUNK];
            let render = [0.1; super::MAX_CHUNK];
            let mut out = [0.0; super::MAX_CHUNK];
            ALLOCS.with(|c| c.set(0));
            DEALLOCS.with(|c| c.set(0));
            ARMED.with(|c| c.set(true));
            eng.run(&mic, &render, &mut out);
            ARMED.with(|c| c.set(false));
            (ALLOCS.with(Cell::get), DEALLOCS.with(Cell::get))
        })
        .join()
        .expect("audio test thread");
        assert_eq!(
            (allocs, frees),
            (0, 0),
            "first callback still uses the heap"
        );
    }

    #[test]
    fn ffi_span_rules_permit_exact_inplace_but_not_partial_overlap() {
        let mut samples = [0.0f32; 16];
        let p = samples.as_mut_ptr();
        assert!(super::valid_audio_span(p, 16));
        assert!(!super::partial_overlap(p, p, 8));
        assert!(super::partial_overlap(p, p.wrapping_add(1), 8));
        assert!(!super::partial_overlap(p, p.wrapping_add(8), 8));
        assert!(!super::valid_audio_span(std::ptr::null(), 8));
    }

    #[test]
    fn print_rust_spa_layout_for_installed_header_comparison() {
        use std::mem::{offset_of, size_of};
        eprintln!(
            "rust-layout: handle={} factory={} aec={} methods={} raw={} methods.init={} methods.run={} methods.init2={}",
            size_of::<super::spa_handle>(),
            size_of::<super::spa_handle_factory>(),
            size_of::<super::spa_audio_aec>(),
            size_of::<super::spa_audio_aec_methods>(),
            size_of::<super::spa_audio_info_raw>(),
            offset_of!(super::spa_audio_aec_methods, init),
            offset_of!(super::spa_audio_aec_methods, run),
            offset_of!(super::spa_audio_aec_methods, init2)
        );
    }
}
