//! Enhances a whole recording on several threads: raw f32 little-endian 48 kHz
//! interleaved frames on stdin, the same frames enhanced and aligned on stdout.
//!
//! ffmpeg -i in.m4a -ar 48000 -f f32le - | dpdfnet-enhance --channels 2 |
//!     ffmpeg -f f32le -ar 48000 -ac 2 -i - out.flac
use dpdfnet_native::{offline, Bundle};
use std::{io::IsTerminal, process::exit};

const USAGE: &str = "\
usage: dpdfnet-enhance [--channels N] [--threads N] [--attenuation DB] < f32le > f32le

  --channels N      interleaved channels, 1-64 (default 1)
  --threads N       stage threads, 1-256 (default: cores of the fastest kind)
  --attenuation DB  0 keeps the input, 100 removes all noise (default 100)

The model is read from $DPDFNET_NATIVE_MODEL, else the installed bundle.";

/// Exits with status 2, the conventional status for a usage error.
fn usage_error(message: &str) -> ! {
    eprintln!("dpdfnet-enhance: {message}\nTry 'dpdfnet-enhance --help'.");
    exit(2);
}

fn value<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> T {
    let Some(v) = args.next() else {
        usage_error(&format!("{name} needs a value"));
    };
    v.parse()
        .unwrap_or_else(|_| usage_error(&format!("{name}: not a number: {v:?}")))
}

fn main() {
    let mut channels = 1usize;
    let mut threads = None;
    let mut attenuation = 100.0f32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            "--channels" => channels = value(&mut args, "--channels"),
            "--threads" => threads = Some(value(&mut args, "--threads")),
            "--attenuation" => attenuation = value(&mut args, "--attenuation"),
            _ => usage_error(&format!("unknown argument {arg:?}")),
        }
    }
    let threads = threads.unwrap_or_else(offline::default_threads);
    if !(1..=64).contains(&channels) {
        usage_error("--channels must be 1-64");
    }
    if !(1..=256).contains(&threads) {
        usage_error("--threads must be 1-256");
    }
    if !(0.0..=100.0).contains(&attenuation) {
        usage_error("--attenuation must be 0-100 dB");
    }
    let stdout = std::io::stdout().lock();
    if stdout.is_terminal() {
        usage_error("refusing to write raw f32 samples to a terminal; redirect stdout");
    }
    let result = Bundle::open(Bundle::default_dir()).and_then(|bundle| {
        offline::enhance(
            &bundle,
            std::io::stdin(),
            stdout,
            channels,
            threads,
            attenuation,
        )
    });
    if let Err(e) = result {
        eprintln!("dpdfnet-enhance: {e}");
        exit(1);
    }
}
