# FireRedVAD

A native port of FireRedVAD (FireRedTeam, Apache-2.0) in `firered-vad/`: two of its
models, rebuilt from the release's checkpoints by `tools/export_weights.py`.

| Model | Outputs | Lookahead | Used by |
|---|---|---|---|
| AED | speech, singing, music | 1.6 s | `dpdfnet-enhance --speech-only`, through `SpeechOnly` ([how](dpdfnet.md#whole-recordings)) |
| Stream-VAD | speech | none | the DeepFilterNet3 voice gate, when chosen ([how](deepfilternet3.md#voice-gate)) |

## Features and network

Kaldi log-mel fbank at 16 kHz: a 25 ms Povey window every 10 ms, DC removal,
pre-emphasis 0.97, 512-point power spectrum, 80 triangular mel bands from 20 Hz to
Nyquist, natural log floored at the f32 epsilon, frames that end before a partial
one. Samples are scaled to 16-bit values first, as in training. Each band is then
normalized by the release's global CMVN.

The network is a DFSMN: an affine layer to 256 with ReLU, a projection to 128 with
ReLU, then eight memory layers, a 256-wide layer with ReLU and a sigmoid output.
Each memory layer adds to its projection a per-channel filter over the 20 frames up
to it and, in the AED, the 20 frames after it; every layer but the first also adds
its input. The AED therefore answers a frame 160 frames after hearing it. Frames
before the stream and after its end count as zero.

The network has no recurrent state: its answer depends on 1.6 s of frames, so it
cannot drift over hours of music or noise as Silero does.

## Accuracy and cost

`tests/parity.rs` compares both models with the official package over 20 s of the
shared test speech, silence and noise: the probabilities differ by less than 1e-4.

Measured on one core of the i5-13400, ten minutes of speech: the AED takes 2.8 s
and the Stream-VAD 2.6 s, about 17 and 16 s of CPU per hour of audio. Both process
one frame at a time.

Reproduce the weights and the parity fixtures (Python with `torch`, `kaldiio` and,
for the fixtures, the `fireredvad` package; none of this is needed at runtime):

```sh
python firered-vad/tools/export_weights.py FireRedVAD/AED firered-vad/model/firered_aed.bin
python firered-vad/tools/export_weights.py FireRedVAD/Stream-VAD firered-vad/model/firered_stream_vad.bin
python firered-vad/tools/reference.py FireRedVAD firered-vad/tests/fixtures
```

The checkpoints are read with `weights_only=True`; the official package loads them
with `weights_only=False`, which runs any code a tampered file carries.
