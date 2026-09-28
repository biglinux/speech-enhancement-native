# Test fixtures

`speech.pcm` and `continuous-speech.pcm` are synthetic speech generated with
eSpeak NG; no person was recorded. Both are mono signed 16-bit little-endian
PCM at 48 kHz. These commands reproduce them byte for byte (eSpeak NG 1.52,
FFmpeg 9):

```sh
espeak-ng -v en-us -s 150 --stdout 'Hello. This is a microphone test.' > speech.wav
ffmpeg -nostdin -v error -i speech.wav -ar 48000 -ac 1 -f s16le speech.pcm

espeak-ng -v en-us -s 130 -w continuous-speech.wav \
    'We are evaluating the microphone recording quality continuously during this conversation.'
ffmpeg -nostdin -v error -i continuous-speech.wav -ar 48000 -ac 1 -t 3 -f s16le continuous-speech.pcm
```

`golden.bin` is the engine's own output, regenerated with `BLESS=1` (see
`tests/golden.rs`).
