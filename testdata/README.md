# Shared test audio

Synthetic speech generated with eSpeak NG; no person was recorded. Both files are
mono signed 16-bit little-endian PCM at 48 kHz. The DeepFilterNet3 plugin tests and
the Silero VAD parity test read them. These commands reproduce them byte for byte
(eSpeak NG 1.52, FFmpeg 9):

```sh
espeak-ng -v en-us -s 150 --stdout 'Hello. This is a microphone test.' > speech.wav
ffmpeg -nostdin -v error -i speech.wav -ar 48000 -ac 1 -f s16le speech.pcm

espeak-ng -v en-us -s 130 -w continuous-speech.wav \
    'We are evaluating the microphone recording quality continuously during this conversation.'
ffmpeg -nostdin -v error -i continuous-speech.wav -ar 48000 -ac 1 -t 3 -f s16le continuous-speech.pcm
```

`heap_calls.rs` is the counting global allocator that the real-time tests and the
DPDFNet benchmark include with `#[path]` to prove a processing path never
touches the heap.
