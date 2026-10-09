//! PipeWire SPA AEC plugin exposing `spa_audio_aec`, backed by the native Rust
//! GTCRN-AEC (`gtcrn-aec`). `module-echo-cancel` loads it via
//! `library.name = aec/libspa-aec-gtcrn`.
//!
//! The graph runs at 48 kHz, mono, planar f32. The band up to 8 kHz is
//! resampled to 16 kHz for the neural canceller, which works in 256-sample
//! hops; FIFOs decouple those hops from the host's block size. The band above
//! 8 kHz goes through a linear adaptive filter instead and is added back,
//! ducked while the neural band is strongly suppressed.
//!
//! Only what `module-echo-cancel` calls is implemented: the factory, the
//! handle's `get_interface`, and `init`/`init2`/`run`/`activate`/`deactivate`.

#![allow(non_camel_case_types)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::io::Write;
use std::ptr;

use gtcrn_aec::{Model, Streamer, hbaec::HbAec};
use ops::resample::{Down3, Up3};

// SPA ABI, mirroring spa/support/plugin.h and spa/interfaces/audio/aec.h.

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

// The installed C headers' sizes and offsets on 64-bit targets;
// gtcrn-aec/tools/spa-aec-abi-check.c checks the same numbers against them.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<spa_handle>() == 24);
    assert!(size_of::<spa_handle_factory>() == 48);
    assert!(size_of::<spa_audio_aec>() == 56);
    assert!(size_of::<spa_audio_aec_methods>() == 88);
    assert!(size_of::<spa_audio_info_raw>() == 272);
    assert!(offset_of!(spa_audio_aec_methods, init) == 16);
    assert!(offset_of!(spa_audio_aec_methods, run) == 24);
    assert!(offset_of!(spa_audio_aec_methods, init2) == 80);
};

// SAFETY: the statics of these types hold only pointers to immutable static
// data and function entry points, and nothing writes through them.
unsafe impl Sync for spa_audio_aec_methods {}
// SAFETY: as above.
unsafe impl Sync for spa_interface_info {}
// SAFETY: as above.
unsafe impl Sync for spa_handle_factory {}

const AEC_TYPE: &[u8] = b"Spa:Pointer:Interface:Audio:AEC\0";
const FACTORY_NAME: &[u8] = b"audio.aec\0";
// The block size the plugin asks the host for (one neural hop at 48 kHz),
// not its input-to-output delay. gtcrn-aec/tools/spa-aec-abi-check.c checks
// these strings through the installed C headers.
const AEC_BLOCK: &[u8] = b"768/48000\0";
const AEC_NAME: &[u8] = b"gtcrn\0";
const SPA_AUDIO_FORMAT_F32P: u32 = 0x206;
const SPA_AUDIO_CHANNEL_MONO: u32 = 2;
const DEFAULT_MODEL: &str = "/usr/share/gtcrn-aec-native/localvqe-pi-aec-v1-49k-f32.gguf";
const EINVAL: c_int = 22;
const ENOTSUP: c_int = 95;

const MAX_CHUNK: usize = 768; // one 16 kHz neural hop at 48 kHz
// Input-to-output delay of both bands, independent of the host block size:
// the pre-filled output hop, the STFT overlap-add hop, and the down- and
// up-sampling filters. 2 * 768 + 2 * 96 = 1728 samples, 36 ms.
const AEC_LATENCY_48K: usize = 2 * MAX_CHUNK + 2 * ops::resample::LPF_DELAY;
const MIC_RING: usize = 8192; // > 2·latency + LPF span + one bounded chunk

/// High-band filter block size and partition count (48 kHz): 64 partitions of
/// 256 samples ≈ 341 ms of echo-tail coverage.
const HB_BLOCK: usize = 256;
const HB_PARTS: usize = 64;

struct Engine {
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
    rs: Vec<f32>,           // reused resampler output
    fifo_gain: Vec<f32>,    // per-output-sample low-band suppression ratio, parallel to fifo_out
    gain_scratch: Vec<f32>, // reused per-emit gain buffer
    hb_gain: f32,           // smoothed high-band gain
    ref_hist: Box<[f32]>,   // raw 48 kHz reference ring
    hb: HbAec,              // high-band linear echo canceller
    hb_mic: Vec<f32>,       // accumulating high-band mic block (HB_BLOCK)
    hb_ref: Vec<f32>,       // accumulating high-band reference block
    hbout: Box<[f32]>,      // echo-cancelled high band, indexed by absolute sample count
    hb_t: usize,            // next absolute sample fed into the high-band filter
    hop_gain: [f32; 4],     // low-band output/input amplitude ratio of recent hops
    hops: usize,            // neural hops processed
    previous_mic_energy: f64,
    hb_blk: Vec<f32>, // reused per-block output
}

// `init` builds the engine on the main thread and `run` uses it on the data
// thread through a raw pointer, which the compiler cannot check.
const _: () = {
    const fn is_send<T: Send>() {}
    is_send::<Engine>()
};

impl Engine {
    fn new(model: Model) -> Result<Self, String> {
        let streamer = Streamer::new(&model)?;
        // Reserve a complete hop before emitting audio. Starting empty inserts
        // zeros at callback boundaries until enough slack accumulates, making
        // latency depend on the callback size and desynchronizing the bands.
        // Most queued before an emit: the pre-filled hop of slack, the hop the
        // block completes, and the two samples by which the down-sampler's
        // phase lets a hop complete before its last input arrives.
        let mut fifo_out = Vec::with_capacity(2 * MAX_CHUNK + 2);
        fifo_out.resize(MAX_CHUNK, 0.0);
        let mut fifo_gain = Vec::with_capacity(2 * MAX_CHUNK + 2);
        fifo_gain.resize(MAX_CHUNK, 1.0);
        Ok(Self {
            model,
            streamer,
            down_mic: Down3::new(MAX_CHUNK),
            down_ref: Down3::new(MAX_CHUNK),
            up_out: Up3::new(Streamer::HOP),
            fifo_mic: Vec::with_capacity(2 * Streamer::HOP),
            fifo_ref: Vec::with_capacity(2 * Streamer::HOP),
            fifo_out,
            lpf: ops::resample::lpf_prototype_48k(),
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
            hop_gain: [1.0; 4],
            hops: 0,
            previous_mic_energy: 0.0,
            hb_blk: vec![0.0; HB_BLOCK],
        })
    }

    /// High band (above ~8 kHz) of `hist[bi]`: the sample minus its zero-phase
    /// low-pass. Reads forward to `bi + LPF_DELAY`, so the caller must have
    /// that much history.
    fn band_hi(&self, hist: &[f32], bi: i64) -> f32 {
        let d = ops::resample::LPF_DELAY as i64;
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

    /// Run the high-band canceller over every sample whose low-pass window is
    /// complete, filling `hbout`. The filter adapts only while the low band is
    /// strongly suppressed, a sign that echo dominates (see `hbaec`).
    ///
    /// Each block reads the ratio of the newest hop whose input it has
    /// entirely seen, a choice fixed by sample counts alone, so the output
    /// does not depend on how the host splits the stream.
    fn feed_high_band(&mut self) {
        let d = ops::resample::LPF_DELAY;
        while (self.hb_t + d) < self.in_count {
            let t = self.hb_t as i64;
            self.hb_mic.push(self.band_hi(&self.mic_hist, t));
            self.hb_ref.push(self.band_hi(&self.ref_hist, t));
            self.hb_t += 1;
            if self.hb_mic.len() == HB_BLOCK {
                // The block needed `hb_t + d` input samples; hop k is complete
                // once `(k + 1) * MAX_CHUNK` have arrived.
                let g = match ((self.hb_t + d) / MAX_CHUNK).checked_sub(1) {
                    Some(k) => {
                        debug_assert!(k < self.hops && k + self.hop_gain.len() >= self.hops);
                        self.hop_gain[k % self.hop_gain.len()]
                    }
                    None => 1.0,
                };
                let adapt = g < 0.5;
                self.hb
                    .process_block(&self.hb_mic, &self.hb_ref, adapt, &mut self.hb_blk);
                let start = self.hb_t - HB_BLOCK;
                for (i, &v) in self.hb_blk.iter().enumerate() {
                    self.hbout[(start + i) % MIC_RING] = v;
                }
                self.hb_mic.clear();
                self.hb_ref.clear();
            }
        }
    }

    /// Process one host block of at most `MAX_CHUNK` samples; the bound keeps
    /// the history rings from wrapping before the high band has consumed them.
    fn run(&mut self, rec: &[f32], play: &[f32], out: &mut [f32]) {
        assert!(rec.len() <= MAX_CHUNK && play.len() == rec.len() && out.len() == rec.len());
        for (&m, &r) in rec.iter().zip(play.iter()) {
            self.mic_hist[self.in_count % MIC_RING] = m;
            self.ref_hist[self.in_count % MIC_RING] = r;
            self.in_count += 1;
        }
        self.down_mic.process(rec, &mut self.rs);
        self.fifo_mic.extend_from_slice(&self.rs);
        self.down_ref.process(play, &mut self.rs);
        self.fifo_ref.extend_from_slice(&self.rs);
        let mut consumed = 0;
        while let (Some(m), Some(r)) = (
            self.fifo_mic[consumed..].first_chunk(),
            self.fifo_ref[consumed..].first_chunk(),
        ) {
            let mut o16 = self.streamer.process_hop(&self.model, m, r);
            // Suppression ratio of this hop, which drives the high-band duck. The
            // output frame need not describe the same instant as the input.
            let energy =
                |samples: &[f32]| samples.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            let me = energy(m);
            let oe = energy(&o16);
            // The STFT output includes the previous microphone hop. Keep that
            // hop's energy in the ceiling so a speech offset is not cut short.
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
            self.hop_gain[self.hops % self.hop_gain.len()] = g;
            self.hops += 1;
            self.up_out.process(&o16, &mut self.rs);
            self.fifo_out.extend_from_slice(&self.rs);
            self.fifo_gain.resize(self.fifo_out.len(), g);
            consumed += Streamer::HOP;
        }
        if consumed > 0 {
            self.fifo_mic.drain(..consumed);
            self.fifo_ref.drain(..consumed);
        }
        self.feed_high_band();
        // emit out.len() samples; zero-fill during the startup fill
        let n = out.len();
        let avail = self.fifo_out.len().min(n);
        out[..avail].copy_from_slice(&self.fifo_out[..avail]);
        out[avail..].fill(0.0);
        // Gains aligned with the emitted samples (passthrough where zero-filled).
        self.gain_scratch.clear();
        self.gain_scratch.resize(n, 1.0);
        self.gain_scratch[..avail].copy_from_slice(&self.fifo_gain[..avail]);
        self.fifo_out.drain(..avail);
        self.fifo_gain.drain(..avail);
        // Add the high band delayed by the same latency as the low band (output
        // sample p aligns with mic[p - latency]). It is ducked while the low
        // band is strongly suppressed, which means far-end echo dominates, and
        // passes while the low band is kept (near-end speech, double talk).
        // The one-pole smoothing avoids zipper noise on gain changes.
        const G_LO: f32 = 0.15; // full duck at/below this low-band ratio
        const G_HI: f32 = 0.45; // full passthrough at/above
        const SMOOTH: f32 = 0.0021; // ~10 ms one-pole at 48 kHz
        let mut hb = self.hb_gain;
        for (j, o) in out.iter_mut().enumerate() {
            let target = ((self.gain_scratch[j] - G_LO) / (G_HI - G_LO)).clamp(0.0, 1.0);
            hb += (target - hb) * SMOOTH;
            // The latency exceeds the high-band filter's lag (LPF_DELAY plus
            // one block), so the sample is always ready; before it, silence.
            let hi = match (self.out_count + j).checked_sub(AEC_LATENCY_48K) {
                Some(idx) => {
                    debug_assert!(idx + HB_BLOCK + ops::resample::LPF_DELAY < self.in_count);
                    self.hbout[idx % MIC_RING]
                }
                None => 0.0,
            };
            *o += hb * hi;
        }
        self.hb_gain = hb;
        self.out_count += n;
    }
}

#[repr(C)]
struct Handle {
    handle: spa_handle,
    aec: spa_audio_aec,
    engine: *mut Engine, // null until init/init2
}

/// The value of `key` in `args`.
///
/// # Safety
/// `args` is null or points to a valid `spa_dict` whose `n_items` items, and
/// their non-null key and value strings, stay valid for the call.
unsafe fn dict_get(args: *const spa_dict, key: &str) -> Option<String> {
    // SAFETY: the caller guarantees `args` is null or valid.
    let d = unsafe { args.as_ref() }?;
    if d.items.is_null() {
        return None;
    }
    // SAFETY: a non-null `items` holds `n_items` entries.
    let items = unsafe { std::slice::from_raw_parts(d.items, d.n_items as usize) };
    items
        .iter()
        .filter(|it| !it.key.is_null() && !it.value.is_null())
        // SAFETY: non-null keys and values are NUL-terminated strings.
        .find(|it| unsafe { CStr::from_ptr(it.key) }.to_bytes() == key.as_bytes())
        .map(|it| {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(it.value) }
                .to_string_lossy()
                .into_owned()
        })
}

/// The engine of the handle `object`, once `init` has built it.
///
/// # Safety
/// `object` is null or the `Handle` that `f_init` initialized.
unsafe fn engine_from(object: *mut c_void) -> Option<*mut Engine> {
    // SAFETY: the caller guarantees `object` is null or an initialized Handle.
    let h = unsafe { object.cast::<Handle>().as_ref() }?;
    (!h.engine.is_null()).then_some(h.engine)
}

/// # Safety
/// `object` is null or an initialized `Handle` not used by another thread
/// during the call; `args` as for [`dict_get`].
unsafe fn do_init(object: *mut c_void, args: *const spa_dict) -> c_int {
    // SAFETY: the caller guarantees `object` is null or an exclusive Handle.
    let Some(h) = (unsafe { object.cast::<Handle>().as_mut() }) else {
        return -EINVAL;
    };
    if !h.engine.is_null() {
        return 0;
    }
    // SAFETY: forwarded from the caller.
    let path = unsafe { dict_get(args, "gtcrn.model") }.unwrap_or_else(|| DEFAULT_MODEL.into());
    match Model::load(&path).and_then(Engine::new) {
        Ok(engine) => {
            h.engine = Box::into_raw(Box::new(engine));
            0
        }
        Err(error) => {
            // init runs on the main thread, so stderr (the PipeWire log) is
            // usable; a failed write must not abort the host.
            let _ = writeln!(
                std::io::stderr(),
                "spa-aec-gtcrn: cannot use model {path}: {error}"
            );
            // A file that cannot be opened reports why; one that opens but is
            // not a usable model is invalid.
            -std::fs::File::open(&path)
                .err()
                .and_then(|e| e.raw_os_error())
                .unwrap_or(EINVAL)
        }
    }
}

fn valid_format(info: Option<&spa_audio_info_raw>) -> bool {
    info.is_some_and(|i| i.format == SPA_AUDIO_FORMAT_F32P && i.rate == 48_000 && i.channels == 1)
}

/// Asks for a mono stream: `module-echo-cancel` adopts the channel layout
/// `init2` writes back, and its default is stereo.
fn negotiate_mono(info: Option<&mut spa_audio_info_raw>) -> bool {
    let Some(i) = info else { return false };
    if i.channels != 1 {
        i.channels = 1;
        i.position[0] = SPA_AUDIO_CHANNEL_MONO;
    }
    valid_format(Some(i))
}

unsafe extern "C" fn m_init(
    object: *mut c_void,
    args: *const spa_dict,
    info: *const spa_audio_info_raw,
) -> c_int {
    // SAFETY: the host passes null or a valid info, the handle it got from
    // `get_interface` and a null or valid dict, on its main thread.
    unsafe {
        if !valid_format(info.as_ref()) {
            return -EINVAL;
        }
        do_init(object, args)
    }
}
unsafe extern "C" fn m_init2(
    object: *mut c_void,
    args: *const spa_dict,
    first: *mut spa_audio_info_raw,
    second: *mut spa_audio_info_raw,
    third: *mut spa_audio_info_raw,
) -> c_int {
    // aec.h and module-echo-cancel name the three streams differently. All
    // three share one contract, so treat each alike without relying on order.
    // SAFETY: as in `m_init`; the infos are distinct writable structs.
    unsafe {
        let negotiated = [first, second, third].map(|i| negotiate_mono(i.as_mut()));
        if negotiated.contains(&false) {
            return -EINVAL;
        }
        do_init(object, args)
    }
}
// module-echo-cancel calls these from stream state changes on the main thread
// while the data thread may be inside `run`, with no lock between them, so
// they cannot touch the engine. After a pause the engine continues from its
// buffered and adapted state.
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
    // SAFETY: `object` is the handle the host got from `get_interface`.
    let Some(engine) = (unsafe { engine_from(object) }) else {
        return -EINVAL;
    };
    // No buffers are accessed for a zero-size callback.
    if n == 0 {
        return 0;
    }
    if rec.is_null() || play.is_null() || out.is_null() {
        return -EINVAL;
    }
    // SAFETY: each non-null array holds the one channel pointer of a mono stream.
    let (rec0, play0, out0) = unsafe { (*rec, *play, *out) };
    let n = n as usize;
    if !valid_audio_span(rec0, n)
        || !valid_audio_span(play0, n)
        || !valid_audio_span(out0, n)
        || partial_overlap(rec0, out0, n)
        || partial_overlap(play0, out0, n)
    {
        return -EINVAL;
    }
    // Staging through local chunks allows exact in-place buffers; partial overlap
    // is rejected above because a write could clobber input not yet read.
    // SAFETY: the host serializes `run` and calls nothing else on the engine
    // meanwhile (`activate`/`deactivate` leave it alone).
    let eng = unsafe { &mut *engine };
    let _denormals = ops::DenormalGuard::new();
    let mut mic = [0.0f32; MAX_CHUNK];
    let mut render = [0.0f32; MAX_CHUNK];
    let mut result = [0.0f32; MAX_CHUNK];
    let mut offset = 0;
    while offset < n {
        let count = (n - offset).min(MAX_CHUNK);
        for i in 0..count {
            // SAFETY: the spans hold `n` samples (checked above as far as a
            // pointer can be); reads finish before the writes below.
            let (m, r) = unsafe { (rec0.add(offset + i).read(), play0.add(offset + i).read()) };
            mic[i] = if m.is_finite() { m } else { 0.0 };
            render[i] = if r.is_finite() { r } else { 0.0 };
        }
        eng.run(&mic[..count], &render[..count], &mut result[..count]);
        for (i, &sample) in result[..count].iter().enumerate() {
            // SAFETY: as above; an exact in-place output only overwrites input
            // already read.
            unsafe { out0.add(offset + i).write(sample) };
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

static METHODS: spa_audio_aec_methods = spa_audio_aec_methods {
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
};

unsafe extern "C" fn h_get_interface(
    handle: *mut spa_handle,
    type_: *const c_char,
    iface: *mut *mut c_void,
) -> c_int {
    if handle.is_null() || type_.is_null() || iface.is_null() {
        return -EINVAL;
    }
    // SAFETY: the loader passes the handle `f_init` initialized (a `Handle`,
    // whose first field is the `spa_handle`), a NUL-terminated type and a
    // writable result pointer.
    unsafe {
        *iface = ptr::null_mut();
        if CStr::from_ptr(type_).to_bytes_with_nul() != AEC_TYPE {
            return -ENOTSUP;
        }
        *iface = (&raw mut (*handle.cast::<Handle>()).aec).cast();
    }
    0
}

unsafe extern "C" fn h_clear(handle: *mut spa_handle) -> c_int {
    // SAFETY: the loader passes null or the handle `f_init` initialized, once,
    // after the last call into it.
    if let Some(h) = unsafe { handle.cast::<Handle>().as_mut() }
        && !h.engine.is_null()
    {
        // SAFETY: `engine` came from `Box::into_raw` in `do_init`.
        drop(unsafe { Box::from_raw(h.engine) });
        h.engine = ptr::null_mut();
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
        return -EINVAL;
    }
    let h = handle.cast::<Handle>();
    // SAFETY: the loader allocates `f_get_size()` bytes, suitably aligned, for
    // the handle.
    unsafe {
        h.write(Handle {
            handle: spa_handle {
                version: 0,
                get_interface: Some(h_get_interface),
                clear: Some(h_clear),
            },
            aec: spa_audio_aec {
                iface: spa_interface {
                    type_: AEC_TYPE.as_ptr().cast(),
                    version: 1,
                    cb: spa_callbacks {
                        funcs: (&raw const METHODS).cast(),
                        data: h.cast(), // methods receive this as `object`
                    },
                },
                name: AEC_NAME.as_ptr().cast(),
                info: ptr::null(),
                latency: AEC_BLOCK.as_ptr().cast(),
            },
            engine: ptr::null_mut(),
        });
    }
    0
}

static AEC_IFACE_INFO: spa_interface_info = spa_interface_info {
    type_: AEC_TYPE.as_ptr().cast(),
};

unsafe extern "C" fn f_enum_interface_info(
    _f: *const spa_handle_factory,
    info: *mut *const spa_interface_info,
    index: *mut u32,
) -> c_int {
    if info.is_null() || index.is_null() {
        return -EINVAL;
    }
    // SAFETY: both are non-null pointers the loader owns for the call.
    unsafe {
        if *index == 0 {
            *info = &AEC_IFACE_INFO;
            *index += 1;
            1
        } else {
            0
        }
    }
}

static FACTORY: spa_handle_factory = spa_handle_factory {
    version: 1,
    name: FACTORY_NAME.as_ptr().cast(),
    info: ptr::null(),
    get_size: Some(f_get_size),
    init: Some(f_init),
    enum_interface_info: Some(f_enum_interface_info),
};

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
        return -EINVAL;
    }
    // SAFETY: both are non-null pointers the loader owns for the call.
    unsafe {
        if *index == 0 {
            *factory = &FACTORY;
            *index += 1;
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
const TEST_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model/localvqe-pi-aec-v1-49k-f32.gguf"
);

#[cfg(test)]
fn test_engine() -> Engine {
    let path = std::env::var("AEC_GTCRN_GGUF").unwrap_or_else(|_| TEST_MODEL.into());
    let model = Model::load(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    Engine::new(model).unwrap()
}

#[cfg(test)]
mod split_band_tests {
    use super::{Engine, test_engine};

    /// Run `mic`/`far` through the engine in 480-sample host blocks.
    fn drive(eng: &mut Engine, mic: &[f32], far: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; mic.len()];
        let blocks = mic
            .as_chunks::<480>()
            .0
            .iter()
            .zip(far.as_chunks::<480>().0);
        for ((m, r), o) in blocks.zip(out.as_chunks_mut::<480>().0) {
            eng.run(m, r, o);
        }
        out
    }

    fn sine(freq: f32, amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin())
            .collect()
    }

    // Split `x` into (low, high) band energy with the plugin's own low-pass.
    fn band_energies(x: &[f32]) -> (f64, f64) {
        let lpf = ops::resample::lpf_prototype_48k();
        let d = ops::resample::LPF_DELAY;
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

    #[test]
    fn output_does_not_depend_on_host_block_size() {
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
        let expected = render(test_engine(), 768);
        for quantum in [1, 128, 256, 480, 767] {
            let actual = render(test_engine(), quantum);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 1e-6, "quantum {quantum}: max sample error {error}");
        }
    }

    #[test]
    fn both_bands_have_the_documented_latency() {
        // One click with a quiet bed: the output's strongest sample is the
        // click, delayed by the plugin latency.
        let mut mic = sine(300.0, 1e-3, 3 * 48_000);
        let click = 2 * 48_000;
        mic[click] = 0.5;
        let out = drive(&mut test_engine(), &mic, &vec![0.0; mic.len()]);
        let peak = (0..out.len())
            .max_by(|&a, &b| out[a].abs().total_cmp(&out[b].abs()))
            .unwrap();
        assert_eq!(peak - click, super::AEC_LATENCY_48K);
    }

    #[test]
    fn silent_microphone_does_not_emit_the_predicted_echo() {
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
        let out = drive(&mut test_engine(), &mic, &reference);
        let peak = out[2 * 48_000..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        assert!(peak < 1e-7, "silent microphone emitted echo: peak {peak}");
    }

    // The high band must be the mic's complement above 8 kHz: ~0 for an
    // in-band tone, ~full for a tone above it. Isolates the split filter.
    fn hiband_rms(freq: f32) -> f32 {
        let mut eng = test_engine();
        let mic = sine(freq, 0.2, 48_000);
        let (s, e) = (24_000, 43_200);
        let mut acc = 0.0f32;
        let delay = ops::resample::LPF_DELAY;
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
        let inrms = 0.2 / 2f32.sqrt();
        let lo = hiband_rms(1000.0);
        let hi = hiband_rms(12000.0);
        assert!(lo < 0.05 * inrms, "1 kHz leaked into hi band: {lo:.4}");
        assert!(hi > 0.9 * inrms, "12 kHz missing from hi band: {hi:.4}");
    }

    // The network cancels only up to 8 kHz; on far-end-only echo the high band
    // must still be removed, by the linear filter and the duck.
    #[test]
    fn far_end_echo_is_cancelled_in_both_bands() {
        let n = 4 * 48_000;
        let mut seed = 99u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / 8_388_608.0 - 1.0
        };
        let far: Vec<f32> = (0..n)
            .map(|i| {
                let tau = 2.0 * std::f32::consts::PI * i as f32 / 48_000.0;
                0.12 * ((tau * 500.0).sin()
                    + (tau * 3000.0).sin()
                    + (tau * 11000.0).sin()
                    + (tau * 14000.0).sin())
                    + 0.02 * rng()
            })
            .collect();
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
        let out = drive(&mut test_engine(), &mic, &far);
        // Measure the converged tail only.
        let tail = 2 * 48_000;
        let (mic_lo, mic_hi) = band_energies(&mic[tail..]);
        let (out_lo, out_hi) = band_energies(&out[tail..]);
        let erle = |i: f64, o: f64| 10.0 * (i / o.max(1e-12)).log10();
        let (erle_lo, erle_hi) = (erle(mic_lo, out_lo), erle(mic_hi, out_hi));
        eprintln!("per-band ERLE: <8kHz = {erle_lo:.1} dB, >8kHz = {erle_hi:.1} dB");
        assert!(erle_lo > 12.0, "low band not cancelled: {erle_lo:.1} dB");
        assert!(erle_hi > 12.0, "high band not cancelled: {erle_hi:.1} dB");
    }

    // A 12 kHz tone, above the network's band, must reach the output.
    #[test]
    fn high_band_survives() {
        let mic = sine(12000.0, 0.2, 48_000);
        let out = drive(&mut test_engine(), &mic, &vec![0.0; mic.len()]);
        let rms = |x: &[f32]| (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let (mi, oi) = (rms(&mic[24_000..]), rms(&out[24_000..]));
        assert!(
            oi > 0.5 * mi,
            "12 kHz tone lost: out RMS {oi:.4} vs in {mi:.4}"
        );
    }

    // Without a far-end reference the low band is kept (ratio ~1), so the duck
    // must leave the near-end high band alone.
    #[test]
    fn near_end_high_band_is_preserved() {
        let low = sine(1000.0, 0.2, 2 * 48_000);
        let mic: Vec<f32> = sine(12000.0, 0.2, low.len())
            .iter()
            .zip(&low)
            .map(|(h, l)| h + l)
            .collect();
        let out = drive(&mut test_engine(), &mic, &vec![0.0; mic.len()]);
        let half = mic.len() / 2;
        let (_, mhi) = band_energies(&mic[half..]);
        let (_, ohi) = band_energies(&out[half..]);
        let keep_hi = (ohi / mhi.max(1e-12)).sqrt();
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
        assert!(valid_format(Some(&info)));
        assert!(!valid_format(None));
        info.rate = 44_100;
        assert_eq!(
            // SAFETY: m_init rejects the rate before it touches the null object.
            unsafe { m_init(ptr::null_mut(), ptr::null(), &info) },
            -EINVAL
        );
        info.rate = 48_000;
        info.channels = 2;
        assert!(!valid_format(Some(&info)));
        info.channels = 1;
        info.format = 0;
        assert!(!valid_format(Some(&info)));
    }

    #[test]
    fn init2_turns_the_default_stereo_streams_mono() {
        let model = std::ffi::CString::new(TEST_MODEL).unwrap();
        let items = [spa_dict_item {
            key: c"gtcrn.model".as_ptr(),
            value: model.as_ptr(),
        }];
        let dict = spa_dict {
            flags: 0,
            n_items: 1,
            items: items.as_ptr(),
        };
        let stereo = || {
            let mut position = [0; 64];
            position[..2].copy_from_slice(&[3, 4]); // FL, FR
            spa_audio_info_raw {
                format: SPA_AUDIO_FORMAT_F32P,
                flags: 0,
                rate: 48_000,
                channels: 2,
                position,
            }
        };
        let mut infos = [stereo(), stereo(), stereo()];
        let mut handle = std::mem::MaybeUninit::<Handle>::uninit();
        let [a, b, c] = &mut infos;
        // SAFETY: `handle` has room for a Handle, the infos outlive the calls,
        // and the handle is cleared once.
        unsafe {
            let h = handle.as_mut_ptr();
            assert_eq!(
                f_init(ptr::null(), h.cast(), ptr::null(), ptr::null(), 0),
                0
            );
            assert_eq!(m_init2(h.cast(), &dict, a, b, c), 0);
            h_clear(h.cast());
        }
        for info in infos {
            assert_eq!(
                (info.channels, info.position[0]),
                (1, SPA_AUDIO_CHANNEL_MONO)
            );
        }
        let mut wrong_rate = stereo();
        wrong_rate.rate = 44_100;
        assert!(!negotiate_mono(Some(&mut wrong_rate)));
    }

    #[test]
    fn dict_get_skips_null_entries() {
        let items = [
            spa_dict_item {
                key: ptr::null(),
                value: c"x".as_ptr(),
            },
            spa_dict_item {
                key: c"gtcrn.model".as_ptr(),
                value: ptr::null(),
            },
            spa_dict_item {
                key: c"gtcrn.model".as_ptr(),
                value: c"/m.gguf".as_ptr(),
            },
        ];
        let dict = spa_dict {
            flags: 0,
            n_items: 3,
            items: items.as_ptr(),
        };
        let no_items = spa_dict {
            flags: 0,
            n_items: 2,
            items: ptr::null(),
        };
        // SAFETY: `dict` points to live NUL-terminated strings; `no_items` and the
        // null dict exercise the null checks.
        unsafe {
            assert_eq!(dict_get(&dict, "gtcrn.model").as_deref(), Some("/m.gguf"));
            assert_eq!(dict_get(&no_items, "gtcrn.model"), None);
            assert_eq!(dict_get(ptr::null(), "gtcrn.model"), None);
        }
    }

    /// `init` with `gtcrn.model` set to `model`, on a fresh handle.
    fn init_with_model(model: &str) -> c_int {
        let model = std::ffi::CString::new(model).unwrap();
        let items = [spa_dict_item {
            key: c"gtcrn.model".as_ptr(),
            value: model.as_ptr(),
        }];
        let dict = spa_dict {
            flags: 0,
            n_items: 1,
            items: items.as_ptr(),
        };
        let info = spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32P,
            flags: 0,
            rate: 48_000,
            channels: 1,
            position: [0; 64],
        };
        let mut handle = std::mem::MaybeUninit::<Handle>::uninit();
        // SAFETY: `handle` has room for a Handle, the buffers outlive the calls,
        // and the handle is cleared once.
        unsafe {
            let h = handle.as_mut_ptr();
            assert_eq!(
                f_init(ptr::null(), h.cast(), ptr::null(), ptr::null(), 0),
                0
            );
            let result = m_init(h.cast(), &dict, &info);
            h_clear(h.cast());
            result
        }
    }

    /// Echo of a noisy far end (50 ms delay) plus a near-end burst, 48 kHz.
    fn echo_and_near_end(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut seed = 5u32;
        let far: Vec<f32> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let env = 0.6 + 0.4 * (i as f32 * 4e-4).sin();
                env * 0.2 * ((seed >> 9) as f32 / 8_388_608.0 - 1.0)
            })
            .collect();
        let mic = (0..n)
            .map(|i| {
                let echo = if i >= 2400 { 0.4 * far[i - 2400] } else { 0.0 };
                let tau = 2.0 * std::f32::consts::PI * i as f32 / 48_000.0;
                let near = if (120_000..168_000).contains(&i) {
                    0.2 * (tau * 300.0).sin() + 0.05 * (tau * 12_000.0).sin()
                } else {
                    0.0
                };
                echo + near
            })
            .collect();
        (mic, far)
    }

    /// Run the signals through the C ABI in host blocks of `block` samples,
    /// writing the output over the microphone buffer when `in_place`.
    fn run_through_abi(mic: &[f32], far: &[f32], block: usize, in_place: bool) -> Vec<f32> {
        let model = std::ffi::CString::new(TEST_MODEL).unwrap();
        let items = [spa_dict_item {
            key: c"gtcrn.model".as_ptr(),
            value: model.as_ptr(),
        }];
        let dict = spa_dict {
            flags: 0,
            n_items: 1,
            items: items.as_ptr(),
        };
        let info = spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32P,
            flags: 0,
            rate: 48_000,
            channels: 1,
            position: [0; 64],
        };
        let mut rec = mic.to_vec();
        let mut out = vec![0.0; mic.len()];
        let mut handle = std::mem::MaybeUninit::<Handle>::uninit();
        // SAFETY: `handle` has room for a Handle, the buffers outlive the calls,
        // and the handle is cleared once.
        unsafe {
            let h = handle.as_mut_ptr();
            assert_eq!(
                f_init(ptr::null(), h.cast(), ptr::null(), ptr::null(), 0),
                0
            );
            assert_eq!(m_init(h.cast(), &dict, &info), 0);
            let mut offset = 0;
            while offset < mic.len() {
                let n = block.min(mic.len() - offset);
                let r = [rec[offset..].as_ptr()];
                let p = [far[offset..].as_ptr()];
                let mut o = [if in_place {
                    rec[offset..].as_mut_ptr()
                } else {
                    out[offset..].as_mut_ptr()
                }];
                assert_eq!(
                    m_run(h.cast(), r.as_ptr(), p.as_ptr(), o.as_mut_ptr(), n as u32),
                    0
                );
                offset += n;
            }
            h_clear(h.cast());
        }
        if in_place { rec } else { out }
    }

    #[test]
    fn output_with_an_active_reference_does_not_depend_on_host_block_size() {
        let (mic, far) = echo_and_near_end(4 * 48_000);
        let expected = run_through_abi(&mic, &far, 480, false);
        assert_eq!(run_through_abi(&mic, &far, 480, true), expected, "in place");
        for block in [1, 128, 1024, 8192] {
            let actual = run_through_abi(&mic, &far, block, false);
            let first = actual.iter().zip(&expected).position(|(a, b)| a != b);
            assert_eq!(first, None, "block {block} differs from block 480");
        }
    }

    #[test]
    fn init_reports_why_a_model_is_unusable() {
        assert_eq!(init_with_model("/nonexistent/model.gguf"), -2); // ENOENT
        assert_eq!(init_with_model(env!("CARGO_MANIFEST_PATH")), -EINVAL);
        assert_eq!(init_with_model(TEST_MODEL), 0);
    }

    #[test]
    fn ffi_span_rules_permit_exact_inplace_but_not_partial_overlap() {
        let mut samples = [0.0f32; 16];
        let p = samples.as_mut_ptr();
        assert!(valid_audio_span(p, 16));
        assert!(!partial_overlap(p, p, 8));
        assert!(partial_overlap(p, p.wrapping_add(1), 8));
        assert!(!partial_overlap(p, p.wrapping_add(8), 8));
        assert!(!valid_audio_span(std::ptr::null(), 8));
    }
}

#[cfg(test)]
#[path = "../../../testdata/heap_calls.rs"]
mod heap_calls;

// The engine runs on PipeWire's data thread: after warm-up, and on the first
// callback from a thread other than the one that built it, it must not touch
// the heap.
#[cfg(test)]
mod rt_alloc_tests {
    use super::{MAX_CHUNK, test_engine};
    use crate::heap_calls::heap_calls;

    #[test]
    fn engine_run_is_alloc_free_after_warmup() {
        let mut eng = test_engine();
        let q = 480;
        let mic: Vec<f32> = (0..q * 280)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin())
            .collect();
        let zero = vec![0.0f32; q];
        let mut out = vec![0.0f32; q];
        let mut blocks = mic.chunks_exact(q);
        // Warm past the FIFO fill and the first GCC-PHAT update.
        for block in blocks.by_ref().take(200) {
            eng.run(block, &zero, &mut out);
        }
        let calls = heap_calls(|| {
            for block in blocks {
                eng.run(block, &zero, &mut out);
            }
        });
        assert_eq!(calls, 0, "Engine::run used the heap");
    }

    // A host block that ends one or two samples before a hop boundary,
    // followed by a full block, leaves the most output queued.
    #[test]
    fn blocks_ending_just_before_a_hop_are_alloc_free() {
        let mut eng = test_engine();
        let (mic, render) = ([0.05; MAX_CHUNK], [0.1; MAX_CHUNK]);
        let mut out = [0.0; MAX_CHUNK];
        eng.run(
            &mic[..MAX_CHUNK - 2],
            &render[..MAX_CHUNK - 2],
            &mut out[..MAX_CHUNK - 2],
        );
        let calls = heap_calls(|| eng.run(&mic, &render, &mut out));
        assert_eq!(calls, 0, "a full block after a short one used the heap");
    }

    #[test]
    fn first_callback_on_another_thread_is_alloc_free() {
        let mut eng = test_engine();
        let calls = std::thread::spawn(move || {
            let (mic, render) = ([0.05; MAX_CHUNK], [0.1; MAX_CHUNK]);
            let mut out = [0.0; MAX_CHUNK];
            heap_calls(|| eng.run(&mic, &render, &mut out))
        })
        .join()
        .unwrap();
        assert_eq!(calls, 0, "the first callback used the heap");
    }
}
