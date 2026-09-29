//! Calibration probe: feed a mono-48k f32 file through the DFN3 engine and print
//! the per-ERB `band_db` (processed-spectrum level the silence-floor gate uses),
//! averaged over the file and over a chosen [start,end] second window. Compare
//! against the GUI's per-band dB on the same processed output to align the curve.
//!
//! Usage: band_probe WEIGHTS.bin INPUT.f32 [START_S END_S]
use dfn3_ladspa::Dfn3;
use std::io::Read;

const HOP: usize = 480;
const SR: usize = 48000;
const NB_ERB: usize = 32;
// Same ERB widths as the engine, to label band centre frequencies.
const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37,
    42, 50, 56, 67,
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: band_probe WEIGHTS.bin INPUT.f32 [START_S END_S]");
        std::process::exit(2);
    }
    let mut wbytes = Vec::new();
    std::fs::File::open(&a[1])
        .unwrap()
        .read_to_end(&mut wbytes)
        .unwrap();
    let mut ibytes = Vec::new();
    std::fs::File::open(&a[2])
        .unwrap()
        .read_to_end(&mut ibytes)
        .unwrap();
    let samples: Vec<f32> = ibytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let (start_s, end_s) = if a.len() >= 5 {
        (a[3].parse::<f32>().unwrap(), a[4].parse::<f32>().unwrap())
    } else {
        (0.0, samples.len() as f32 / SR as f32)
    };

    let mut eng = Dfn3::new(&wbytes);
    let mut out = vec![0.0f32; HOP];
    let mut sum = [0.0f64; NB_ERB];
    let mut sum_win = [0.0f64; NB_ERB];
    let mut n = 0u64;
    let mut n_win = 0u64;
    for (hop_idx, chunk) in samples.chunks(HOP).enumerate() {
        if chunk.len() < HOP {
            break;
        }
        eng.process(chunk, &mut out);
        let t = (hop_idx * HOP) as f32 / SR as f32;
        for (total, &db) in sum.iter_mut().zip(&eng.band_db) {
            *total += f64::from(db);
        }
        n += 1;
        if t >= start_s && t < end_s {
            for (total, &db) in sum_win.iter_mut().zip(&eng.band_db) {
                *total += f64::from(db);
            }
            n_win += 1;
        }
    }
    let bin_hz = SR as f32 / 960.0;
    let mut off = 0usize;
    println!("band  centreHz   avg_all_dB   avg_[{start_s},{end_s})_dB");
    for b in 0..NB_ERB {
        let w = ERB_WIDTHS[b];
        let centre = (off as f32 + w as f32 * 0.5) * bin_hz;
        off += w;
        let all = if n > 0 { sum[b] / n as f64 } else { 0.0 };
        let win = if n_win > 0 {
            sum_win[b] / n_win as f64
        } else {
            0.0
        };
        println!("{b:3}  {centre:8.0}   {all:9.1}   {win:9.1}");
    }
    eprintln!("hops total {n}, window hops {n_win}");
}
