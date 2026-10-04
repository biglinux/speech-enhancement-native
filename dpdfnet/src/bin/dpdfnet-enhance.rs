//! Enhances a whole recording on several threads: raw f32 little-endian 48 kHz
//! interleaved frames on stdin, the same frames enhanced and aligned on stdout.
//!
//! ffmpeg -i in.m4a -ar 48000 -f f32le - | dpdfnet-enhance --channels 2 |
//!     ffmpeg -f f32le -ar 48000 -ac 2 -i - out.flac
use dpdfnet_native::{offline, Bundle};

const USAGE: &str =
    "usage: dpdfnet-enhance [--channels N] [--threads N] [--attenuation DB] < f32le > f32le";

fn value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T, String> {
    args.next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{name} needs a number\n{USAGE}"))
}

fn run() -> Result<(), String> {
    let mut channels = 1usize;
    let mut threads = offline::default_threads();
    let mut attenuation = 100.0f32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--channels" => channels = value(&mut args, "--channels")?,
            "--threads" => threads = value(&mut args, "--threads")?,
            "--attenuation" => attenuation = value(&mut args, "--attenuation")?,
            _ => return Err(USAGE.into()),
        }
    }
    if !(1..=64).contains(&channels) || !(1..=256).contains(&threads) {
        return Err("--channels must be 1-64 and --threads 1-256".into());
    }
    if !(0.0..=100.0).contains(&attenuation) {
        return Err("--attenuation must be 0-100 dB".into());
    }
    let bundle = Bundle::open(Bundle::default_dir())?;
    offline::enhance(
        &bundle,
        std::io::stdin(),
        std::io::stdout().lock(),
        channels,
        threads,
        attenuation,
    )
}

fn main() {
    if let Err(e) = run() {
        eprintln!("dpdfnet-enhance: {e}");
        std::process::exit(1);
    }
}
