//! Enhances a whole recording on several threads: raw f32 little-endian 48 kHz
//! interleaved frames on stdin, the same frames enhanced and aligned on stdout.
//!
//! ffmpeg -i in.m4a -ar 48000 -f f32le - | dpdfnet-enhance --channels 2 |
//!     ffmpeg -f f32le -ar 48000 -ac 2 -i - out.flac
use dpdfnet_native::{Bundle, offline};
use std::{io::IsTerminal, process::ExitCode};

const USAGE: &str = "\
usage: dpdfnet-enhance [--channels N] [--threads N] [--attenuation DB] < f32le > f32le

  --channels N      interleaved channels, 1-64 (default 1)
  --threads N       stage threads, 1-256 (default: cores of the fastest kind)
  --attenuation DB  0 keeps the input, 100 removes all noise (default 100)

The model is read from $DPDFNET_NATIVE_MODEL, else the installed bundle.";

struct Options {
    channels: usize,
    threads: usize,
    attenuation: f32,
}

fn value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T, String> {
    let v = args.next().ok_or_else(|| format!("{name} needs a value"))?;
    v.parse()
        .map_err(|_| format!("{name}: not a number: {v:?}"))
}

/// The options, `None` for `--help`, or a usage error.
fn parse(mut args: impl Iterator<Item = String>) -> Result<Option<Options>, String> {
    let mut channels = 1usize;
    let mut threads = None;
    let mut attenuation = 100.0f32;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--channels" => channels = value(&mut args, "--channels")?,
            "--threads" => threads = Some(value(&mut args, "--threads")?),
            "--attenuation" => attenuation = value(&mut args, "--attenuation")?,
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let threads = threads.unwrap_or_else(offline::default_threads);
    if !(1..=64).contains(&channels) {
        return Err("--channels must be 1-64".into());
    }
    if !(1..=256).contains(&threads) {
        return Err("--threads must be 1-256".into());
    }
    if !(0.0..=100.0).contains(&attenuation) {
        return Err("--attenuation must be 0-100 dB".into());
    }
    Ok(Some(Options {
        channels,
        threads,
        attenuation,
    }))
}

fn main() -> ExitCode {
    let options = match parse(std::env::args().skip(1)) {
        Ok(Some(options)) => options,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("dpdfnet-enhance: {message}\nTry 'dpdfnet-enhance --help'.");
            // The conventional status for a usage error.
            return ExitCode::from(2);
        }
    };
    let stdout = std::io::stdout().lock();
    if stdout.is_terminal() {
        eprintln!(
            "dpdfnet-enhance: refusing to write raw f32 samples to a terminal; redirect stdout"
        );
        return ExitCode::from(2);
    }
    let result = Bundle::open(Bundle::default_dir()).and_then(|bundle| {
        offline::enhance(
            &bundle,
            std::io::stdin(),
            stdout,
            options.channels,
            options.threads,
            options.attenuation,
        )
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dpdfnet-enhance: {e}");
            ExitCode::FAILURE
        }
    }
}
