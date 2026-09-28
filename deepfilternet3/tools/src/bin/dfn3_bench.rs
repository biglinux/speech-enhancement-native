#![allow(clippy::needless_range_loop)]
use dfn3_ladspa::{Dfn3, HOP};
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let wb = std::fs::read(&a[1]).unwrap();
    let mut eng = Dfn3::new(&wb);
    let n = 3000;
    let mut frames = vec![[0.0f32; HOP]; n];
    for f in 0..n {
        for j in 0..HOP {
            frames[f][j] = 0.05 * ((f * HOP + j) as f32 * 0.02).sin()
                + 0.01 * ((((f * 7 + j) % 997) as f32 / 498.0) - 1.0);
        }
    }
    let mut ob = [0.0f32; HOP];
    for f in 0..100 {
        eng.process(&frames[f % n], &mut ob);
    }
    let mut ts = vec![0.0f64; n];
    for f in 0..n {
        let t = Instant::now();
        eng.process(&frames[f], &mut ob);
        ts[f] = t.elapsed().as_secs_f64() * 1e3;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = ts.iter().sum::<f64>() / n as f64;
    println!(
        "DFN3-rust: mean={:.3} p50={:.3} p99={:.3} ms/frame  RTF={:.3}",
        mean,
        ts[n / 2],
        ts[(n as f64 * 0.99) as usize],
        mean / 10.0
    );
}
