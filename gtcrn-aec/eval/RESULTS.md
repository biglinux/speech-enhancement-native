# AEC candidates — quality + resource (16 kHz test set)

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

FE-ERLE/DC-ERLE: echo removed (dB, higher better) on far-end-only / delay-change.
NE-*: near-end preservation with no echo (PESQ 1-4.5, STOI 0-1, higher better).
DT-*: near-end quality during double-talk. RTf: process/audio (<1 = real-time).
lat_ms: algorithmic frame latency. wt_KB: model weight size.

## Method
Test set built from recorded speech (two speakers) + recorded room noise:
far-end (loudspeaker) convolved with a synthetic RIR + bulk delay + soft-clip
loudspeaker nonlinearity → echo; near-end voice + noise. 4 scenarios × 10 s, seeded.
WebRTC is the *shipped* plugin driven offline through its real SPA `spa_audio_aec`
interface. SpeexDSP = classic MDF AEC, no residual preprocessor. DTLN-aec and
LocalVQE run their reference inference (LiteRT / ggml). All candidates scored by the
same aligned (delay+gain-compensated) PESQ/STOI/ERLE pipeline. Resource numbers are
16 kHz single-thread; RSS is indicative only (interpreter overhead differs per runner).

## Reading

- **Echo removal:** DTLN dominates (30 dB) ≫ WebRTC (18) > localvqe_49k (11) /
  localvqe_200k (9) > speex (4) > localvqe_2.7k (~0, it is a front-end filter only).
- **Your-voice preservation (no echo):** localvqe (all sizes) and passthrough/webrtc/
  speex are pristine (PESQ ~4.0-4.25). **DTLN colors/attenuates the near-end badly
  (PESQ 2.35→1.91)** — it is an aggressive suppressor.
- **Double-talk (both talking):** DTLN best (STOI 0.47-0.50) > localvqe_49k (0.44) >
  speex (0.40) > passthrough (0.38) > localvqe_200k (0.37) > **webrtc worst (0.30)**.
- **Cost/latency/size:** webrtc/speex cheapest + lowest latency (10-16 ms).
  localvqe_49k: RTf 0.08, 16 ms, 2.3 MB. DTLN_128: RTf 0.03 but 32 ms + 7 MB.

## No single dominant winner — it is a trade-off

- **DTLN_128** — maximum echo kill + best double-talk, but **degrades the user's own
  voice** (the opposite of this project's goal) and has the highest latency (32 ms) and
  weight (7 MB). MIT.
- **localvqe_49k (GTCRN-AEC)** — **best voice preservation** (PESQ 4.25, does not touch
  the near-end), beats WebRTC on double-talk, tiny (49 K params → ~sub-MB as W8A16),
  Apache-2.0, GTCRN = ultralow-compute (fits the i3/i5 goal). Weakness: 11 dB echo
  removal, below WebRTC's 18 dB.
- **WebRTC (incumbent)** — solid echo (18) + pristine near-end, but worst double-talk.

## Recommendation

For a *microphone* AEC where preserving the user's own voice is paramount and the
target is low-resource on i3/i5, **localvqe_49k (GTCRN-AEC)** is the best Rust-port
candidate: voice-safe, tiny, Apache-2.0, better double-talk than the shipped WebRTC.
Its lower raw ERLE is acceptable because the product already gates the AEC off when no
far-end plays, so the near-end-only quality (where localvqe excels) is what a user
hears most. If raw echo suppression is the priority instead, DTLN is far stronger but
sacrifices near-end voice quality. SpeexDSP and localvqe_2.7k are not competitive here.
