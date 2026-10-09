//! Whole-recording enhancement on several threads, for converters rather than
//! live audio.
//!
//! Every channel runs through the network as a pipeline: the stages of
//! `Model::into_stages` are spread over threads,
//! and while a later stage works on hop t an earlier one already works on hop
//! t+1. Each stage keeps only its own state and does the same arithmetic in the
//! same order, so the output is bit-identical to the LADSPA plugin for any
//! thread count. The recording is preceded by its opening half second played
//! backwards, the lead-in the converters put in front of the plugin.
use crate::{
    AudioProcessor, Bundle, HOP, Result,
    audio::{Analysis, LATENCY, Synthesis, bounded_input},
    model::{Frame, Stage},
};
use ops::DenormalGuard;
use std::{
    collections::{BTreeMap, VecDeque},
    io::{ErrorKind, Read, Write},
    sync::{
        Arc,
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, Scope},
};

/// Opening samples played backwards in front of the recording.
pub const LEAD_IN: usize = 24_000;
/// Samples per message between the reader, the channels and the writer.
const BLOCK: usize = 50 * HOP;

enum Hop {
    Frame(Box<Frame>),
    /// End of the recording, with its length in samples.
    End(usize),
}

enum Output {
    Samples(usize, Vec<f32>),
    End(usize),
}

type Sink = SyncSender<Result<Output>>;

/// Reads interleaved f32 little-endian 48 kHz frames until end of input and
/// writes the enhanced frames, same length and channel count, without the
/// plugin's latency. Each channel gets `threads / channels` stage threads.
pub fn enhance(
    bundle: &Arc<Bundle>,
    input: impl Read + Send,
    mut output: impl Write,
    channels: usize,
    threads: usize,
    attenuation_db: f32,
) -> Result<()> {
    if channels == 0 || threads == 0 {
        return Err("channels and threads must be at least 1".into());
    }
    let groups = (threads / channels).max(1);
    let cpus = stage_cpus().0;
    thread::scope(|scope| {
        let (out_tx, out_rx) = mpsc::sync_channel(4 * channels);
        let mut feeds = Vec::with_capacity(channels);
        for channel in 0..channels {
            let (feed_tx, feed_rx) = mpsc::sync_channel(4);
            feeds.push(feed_tx);
            // Stage threads of every channel take the best CPUs in turn.
            let pins: Vec<Option<usize>> = (0..groups)
                .map(|g| match cpus.len() {
                    0 => None,
                    n => Some(cpus[(channel * groups + g) % n]),
                })
                .collect();
            start_channel(
                scope,
                bundle,
                channel,
                &pins,
                attenuation_db,
                feed_rx,
                &out_tx,
            )?;
        }
        let errors = out_tx.clone();
        drop(out_tx);
        scope.spawn(move || {
            if let Err(e) = read(input, &feeds) {
                let _ = errors.send(Err(e));
            }
        });
        write(&mut output, channels, &out_rx)
    })
}

fn start_channel<'scope>(
    scope: &'scope Scope<'scope, '_>,
    bundle: &Arc<Bundle>,
    channel: usize,
    pins: &[Option<usize>],
    attenuation_db: f32,
    feed: Receiver<Option<Vec<f32>>>,
    out: &Sink,
) -> Result<()> {
    // Activate, then the control of the first run, as the plugin sees them.
    let mut processor = AudioProcessor::new(bundle.clone())?;
    processor.reset();
    processor.set_attenuation_db(attenuation_db);
    let (analysis, model, synthesis, dry_mix) = processor.into_parts();
    let stages = group(model.into_stages(), pins.len());
    let (free_tx, free_rx) = mpsc::sync_channel(stages.len() + 3);
    let (first_tx, mut rx) = mpsc::sync_channel(1);
    let errors = out.clone();
    let frames = stages.len() + 3;
    scope.spawn(move || {
        let result = source(feed, analysis, dry_mix, frames, &free_rx, &first_tx);
        report(result, &errors);
    });
    for (i, mut stages) in stages.into_iter().enumerate() {
        let (tx, next_rx) = mpsc::sync_channel(1);
        let input = std::mem::replace(&mut rx, next_rx);
        let cpu = pins.get(i).copied().flatten();
        scope.spawn(move || {
            if let Some(cpu) = cpu {
                pin(cpu);
            }
            let _denormals = DenormalGuard::new();
            while let Ok(hop) = input.recv() {
                if let Hop::Frame(mut frame) = hop {
                    for stage in &mut stages {
                        stage.run(&mut frame);
                    }
                    if tx.send(Hop::Frame(frame)).is_err() {
                        return;
                    }
                } else if tx.send(hop).is_err() {
                    return;
                }
            }
        });
    }
    let out = out.clone();
    scope.spawn(move || {
        let result = sink(channel, &rx, synthesis, &free_tx, &out);
        report(result, &out);
    });
    Ok(())
}

/// Stage threads to use by default: one per physical core of the fastest kind,
/// at most 256.
pub fn default_threads() -> usize {
    let n = match stage_cpus().1 {
        0 => thread::available_parallelism().map_or(1, usize::from),
        n => n,
    };
    n.min(256)
}

/// CPUs for the stage threads, best first: one per physical core, fastest
/// cores first, then the other hardware threads; and how many cores run within
/// 10% of the top frequency, which also counts cores that firmware rates a step
/// apart (favoured or preferred cores). Two stages on the hyperthreads of one
/// core run the SIMD kernels at about half speed, and the scheduler, waking a
/// stage near the stage that woke it, puts them there; so every stage thread is
/// pinned.
fn stage_cpus() -> (Vec<usize>, usize) {
    let mut cores: BTreeMap<String, (u64, Vec<usize>)> = BTreeMap::new();
    for cpu in allowed_cpus() {
        let dir = format!("/sys/devices/system/cpu/cpu{cpu}");
        let Ok(core) = std::fs::read_to_string(format!("{dir}/topology/core_cpus_list")) else {
            continue;
        };
        let freq = std::fs::read_to_string(format!("{dir}/cpufreq/cpuinfo_max_freq"))
            .ok()
            .and_then(|f| f.trim().parse().ok())
            .unwrap_or(0);
        cores.entry(core).or_insert((freq, Vec::new())).1.push(cpu);
    }
    let mut cores: Vec<(u64, Vec<usize>)> = cores.into_values().collect();
    cores.sort_by_key(|(freq, cpus)| (std::cmp::Reverse(*freq), cpus[0]));
    let top = cores.first().map_or(0, |c| c.0);
    let fast = cores
        .iter()
        .filter(|c| c.0.saturating_mul(10) >= top.saturating_mul(9))
        .count();
    let mut order: Vec<usize> = cores.iter().map(|c| c.1[0]).collect();
    order.extend(cores.iter().flat_map(|c| c.1[1..].iter().copied()));
    (order, fast)
}

/// CPUs this process may run on.
fn allowed_cpus() -> Vec<usize> {
    // SAFETY: cpu_set_t is a plain bit array, all zeroes is the empty set, the
    // kernel writes at most the size passed, and every index is below CPU_SETSIZE.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
    }
}

/// Keeps the calling thread on one CPU; on failure it stays where it was.
fn pin(cpu: usize) {
    // SAFETY: as in `allowed_cpus`; `cpu` came from the allowed set.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn report(result: Result<()>, out: &Sink) {
    if let Err(e) = result {
        let _ = out.send(Err(e));
    }
}

/// Splits the stages into at most `groups` runs of neighbours, keeping the
/// costliest run as cheap as possible.
fn group(stages: Vec<Stage>, groups: usize) -> Vec<Vec<Stage>> {
    let costs: Vec<u32> = stages.iter().map(Stage::cost).collect();
    let runs = |cap: u32| {
        let mut runs = 1;
        let mut sum = 0;
        for &c in &costs {
            if sum + c > cap {
                runs += 1;
                sum = 0;
            }
            sum += c;
        }
        runs
    };
    let mut cap = costs.iter().copied().max().unwrap_or(0);
    while runs(cap) > groups {
        cap += 1;
    }
    let mut out = vec![Vec::new()];
    let mut sum = 0;
    for (stage, c) in stages.into_iter().zip(costs) {
        if sum + c > cap {
            out.push(Vec::new());
            sum = 0;
        }
        sum += c;
        if let Some(run) = out.last_mut() {
            run.push(stage);
        }
    }
    out
}

/// Lead-in, recording and latency padding, hop by hop through the analysis.
fn source(
    feed: Receiver<Option<Vec<f32>>>,
    mut analysis: Analysis,
    dry_mix: f32,
    frames: usize,
    free: &Receiver<Box<Frame>>,
    first: &SyncSender<Hop>,
) -> Result<()> {
    let _denormals = DenormalGuard::new();
    let mut spare: Vec<Box<Frame>> = (0..frames).map(|_| Box::default()).collect();
    let mut pending: Vec<f32> = Vec::with_capacity(BLOCK + LEAD_IN + HOP);
    let mut head: Option<Vec<f32>> = Some(Vec::with_capacity(LEAD_IN));
    let mut length = 0;
    let mut push = |samples: &[f32], pending: &mut Vec<f32>| -> Result<bool> {
        pending.extend(samples.iter().map(|&x| bounded_input(x)));
        let mut used = 0;
        while pending.len() - used >= HOP {
            let Some(mut frame) = spare.pop().or_else(|| free.recv().ok()) else {
                return Ok(false);
            };
            let spec = analysis
                .run(&pending[used..used + HOP])
                .ok_or("the FFT failed")?;
            frame.spec.copy_from_slice(spec);
            frame.dry_mix = dry_mix;
            if first.send(Hop::Frame(frame)).is_err() {
                return Ok(false);
            }
            used += HOP;
        }
        pending.drain(..used);
        Ok(true)
    };
    loop {
        // A closed feed means the reader failed; it reports why.
        let Ok(block) = feed.recv() else {
            return Ok(());
        };
        let ended = block.is_none();
        let block = block.unwrap_or_default();
        length += block.len();
        if let Some(h) = &mut head {
            h.extend_from_slice(&block);
            if h.len() < LEAD_IN && !ended {
                continue;
            }
            let mut lead: Vec<f32> = h[..LEAD_IN.min(h.len())].to_vec();
            lead.resize(LEAD_IN, 0.0);
            lead.reverse();
            if !push(&lead, &mut pending)? || !push(h, &mut pending)? {
                return Ok(());
            }
            head = None;
        } else if !push(&block, &mut pending)? {
            return Ok(());
        }
        if ended {
            if !push(&[0.0; LATENCY], &mut pending)? {
                return Ok(());
            }
            let _ = first.send(Hop::End(length));
            return Ok(());
        }
    }
}

/// Synthesis and the plugin's one-hop output delay. Drops the lead-in and the
/// latency, and holds back the last hop until the length is known.
fn sink(
    channel: usize,
    hops: &Receiver<Hop>,
    mut synthesis: Synthesis,
    free: &SyncSender<Box<Frame>>,
    out: &Sink,
) -> Result<()> {
    let _denormals = DenormalGuard::new();
    let mut skip = LEAD_IN + LATENCY;
    let mut sent = 0;
    let mut block = Vec::with_capacity(BLOCK + 2 * HOP);
    let mut hop = [0.0f32; HOP];
    let mut emit = |samples: &[f32], block: &mut Vec<f32>| {
        let dropped = skip.min(samples.len());
        skip -= dropped;
        block.extend_from_slice(&samples[dropped..]);
    };
    // The plugin emits a hop's result while it reads the next one.
    emit(&hop, &mut block);
    while let Ok(next) = hops.recv() {
        match next {
            Hop::Frame(frame) => {
                if !synthesis.run(&frame.spec, &mut hop) {
                    return Err(format!("processing fault in channel {channel}"));
                }
                let _ = free.send(frame);
                emit(&hop, &mut block);
                if block.len() >= BLOCK + HOP {
                    let ready: Vec<f32> = block.drain(..block.len() - HOP).collect();
                    sent += ready.len();
                    if out.send(Ok(Output::Samples(channel, ready))).is_err() {
                        return Ok(());
                    }
                }
            }
            Hop::End(length) => {
                block.truncate(length.saturating_sub(sent));
                let _ = out.send(Ok(Output::Samples(channel, block)));
                let _ = out.send(Ok(Output::End(length)));
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Reads until `buf` is full or the input ends; returns the bytes read.
fn fill(input: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match input.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(format!("reading the input: {e}")),
        }
    }
    Ok(filled)
}

fn read(mut input: impl Read, feeds: &[SyncSender<Option<Vec<f32>>>]) -> Result<()> {
    let frame = 4 * feeds.len();
    let mut bytes = vec![0u8; frame * BLOCK];
    loop {
        let n = fill(&mut input, &mut bytes)?;
        if n % frame != 0 {
            return Err("the input ends inside a frame".into());
        }
        for (channel, feed) in feeds.iter().enumerate() {
            let samples = bytes[..n]
                .chunks_exact(frame)
                .map(|f| {
                    let s = &f[4 * channel..4 * channel + 4];
                    f32::from_le_bytes([s[0], s[1], s[2], s[3]])
                })
                .collect();
            if feed.send(Some(samples)).is_err() {
                return Ok(());
            }
        }
        if n < bytes.len() {
            for feed in feeds {
                let _ = feed.send(None);
            }
            return Ok(());
        }
    }
}

fn write(
    output: &mut impl Write,
    channels: usize,
    results: &Receiver<Result<Output>>,
) -> Result<()> {
    let mut queues = vec![VecDeque::<f32>::new(); channels];
    let mut length = None;
    let mut ended = 0;
    let mut written = 0;
    let mut bytes = Vec::new();
    while ended < channels {
        match results
            .recv()
            .map_err(|_| "the enhancement threads stopped unexpectedly")??
        {
            Output::Samples(channel, samples) => queues[channel].extend(samples),
            Output::End(n) => {
                length = Some(n);
                ended += 1;
            }
        }
        let take = queues.iter().map(VecDeque::len).min().unwrap_or(0);
        bytes.clear();
        for _ in 0..take {
            for queue in &mut queues {
                let x = queue.pop_front().unwrap_or_default();
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        output
            .write_all(&bytes)
            .map_err(|e| format!("writing the output: {e}"))?;
        written += take;
    }
    if length != Some(written) {
        return Err("the enhancement ended early".into());
    }
    output
        .flush()
        .map_err(|e| format!("writing the output: {e}"))
}
