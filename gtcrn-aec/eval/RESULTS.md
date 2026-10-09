# AEC candidates on a synthetic 16 kHz test set

Measured on 2026-09-28 with the code of the commit that added GTCRN-AEC; the CPU
and the source clips were not recorded. The `localvqe_*` rows are the
LocalVQE ggml CLI running each GGUF over whole files at 16 kHz. They measure
the network the plugin ports, not the Rust port itself and not the shipped
plugin, which runs at 48 kHz and adds a separate filter above 8 kHz
(`docs/gtcrn-aec.md`).

<!-- aggregate:start -->
```
         cand | FE-ERLE | DC-ERLE | NE-PESQ | NE-STOI | DT-PESQ | DT-STOI |    RTf | lat_ms | wt_KB
--------------+---------+---------+---------+---------+---------+---------+--------+--------+------
  passthrough |    -0.0 |    -0.0 |    4.21 |   0.967 |    1.07 |   0.383 |      0 |      0 |     0
       webrtc |   18.71 |    13.6 |    4.02 |   0.963 |    1.05 |   0.297 | 0.0083 |     10 |     0
     speexdsp |    3.91 |    3.09 |    4.09 |   0.966 |    1.12 |     0.4 | 0.0081 |     16 |     0
     dtln_128 |   30.38 |   28.31 |    2.35 |   0.863 |    1.13 |   0.465 |  0.034 |     32 |  7103
     dtln_256 |   30.76 |   29.13 |    2.17 |   0.863 |    1.11 |   0.495 | 0.1022 |     32 | 15180
     dtln_512 |   30.85 |    31.8 |    1.91 |   0.852 |    1.14 |   0.501 | 0.3873 |     32 | 40549
localvqe_2.7k |    1.56 |    0.78 |    4.21 |   0.967 |     1.1 |   0.387 | 0.0358 |     16 |    16
 localvqe_49k |   11.48 |    6.41 |    4.25 |   0.961 |    1.11 |   0.443 | 0.0801 |     16 |  2271
localvqe_200k |    9.32 |   10.68 |     4.2 |   0.967 |    1.04 |   0.368 | 0.3178 |     16 |  2856
```
<!-- aggregate:end -->

- FE-ERLE, DC-ERLE: echo removed, in dB, on the far-end-only and the
  delay-change scenarios.
- NE-PESQ, NE-STOI: near-end speech with no echo, against the clean near end
  (PESQ 1 to 4.5, STOI 0 to 1).
- DT-PESQ, DT-STOI: the same during double talk.
- RTf: processing time over audio time, one thread.
- lat_ms: the frame or hop length each runner reports, not a measured delay.
- wt_KB: size of the weight files.

## Test set

`gen_testset.py` builds four 10 s scenarios per rate (16 and 48 kHz) from three
source clips that the repository does not distribute: far-end speech
(`--far`), near-end speech from a second speaker (`--near`) and recorded room
noise (`--noise`), in any format and rate `soundfile` reads. Each clip is tiled
or trimmed to 10 s and set to -16, -20 and -48 dBFS RMS. The echo is the far
end through a synthetic exponentially decaying room response (T60 0.35 s), a
`tanh` soft clip and a 45 ms delay, at -15 dBFS; in `delaychange` the delay
moves to 95 ms halfway. The seed is fixed, so the same clips give the same set.

Every candidate's output is aligned in delay and gain to its target before
`score.py` computes ERLE, PESQ and STOI. WebRTC is the PipeWire plugin
(`libspa-aec-webrtc.so`) driven through its SPA interface by `run_webrtc.c`;
SpeexDSP runs without its residual echo suppressor; DTLN-aec runs its
published LiteRT models.

## Reproduce

Requires Python with numpy, scipy, soundfile, pesq and pystoi, ffmpeg,
libspeexdsp, the tflite runtime for DTLN, the DTLN-aec pretrained models and a
LocalVQE CLI build with its GGUFs.

```sh
T=~/.cache/aec-eval/testset C=~/.cache/aec-eval/cand
python3 gen_testset.py --far far.wav --near near.wav --noise noise.wav --out $T
cc -O2 $(pkg-config --cflags libspa-0.2) run_webrtc.c -ldl -o run_webrtc
mkdir -p $C/webrtc $C/speexdsp $C/dtln_128 $C/localvqe_49k
python3 run_webrtc.py --bin ./run_webrtc --testset $T --out $C/webrtc > $C/webrtc/res.json
python3 run_speex.py --testset $T --out $C/speexdsp > $C/speexdsp/res.json
python3 run_dtln.py --size 128 --models <DTLN-aec>/pretrained_models \
    --testset $T --out $C/dtln_128 > $C/dtln_128/res.json
python3 run_localvqe.py --bin <localvqe> --model <localvqe-49k.gguf> \
    --name localvqe_49k --testset $T --out $C/localvqe_49k > $C/localvqe_49k/res.json
for c in webrtc speexdsp dtln_128 localvqe_49k; do
    python3 score.py --testset $T --cand $C/$c --name $c --rates 16000 > $C/$c/score.json
done
python3 aggregate.py --cand $C --out RESULTS.md
```

The other DTLN sizes and LocalVQE models follow the same pattern. The
`passthrough` row scores the microphone signal as if it were the output; the
repository has no runner for it.

## Reading

- Echo removal on far end only: DTLN 30 to 31 dB, WebRTC 18.7, localvqe_49k
  11.5, localvqe_200k 9.3, SpeexDSP 3.9, localvqe_2.7k 1.6.
- Near end only: every candidate except DTLN stays within 0.2 PESQ of the
  untouched microphone (4.21); DTLN drops to 1.9 to 2.35.
- Double talk: PESQ is 1.04 to 1.14 for every candidate, the passthrough
  included, so it does not separate them. STOI ranges from 0.30 (WebRTC) to
  0.50 (DTLN 512), with localvqe_49k at 0.44 and the passthrough at 0.38; one
  10 s clip per scenario cannot resolve differences of a few hundredths.

The plugin uses localvqe_49k: it removes 11.5 dB of echo while scoring like
the untouched microphone when only the near end speaks. DTLN removes more
echo but degrades the near-end voice; WebRTC removes more echo (18.7 dB).
