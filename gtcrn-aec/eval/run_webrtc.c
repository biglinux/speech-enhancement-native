// Offline driver for the shipped WebRTC AEC SPA plugin (libspa-aec-webrtc.so).
// Loads the same plugin the product loads, runs it over raw mono f32 files,
// 10 ms frames. Args mirror echo_cancel.rs (NS/AGC off, HPF/VAD on).
//   run_webrtc <plugin.so> <rate> <mic.f32> <ref.f32> <out.f32>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <spa/support/plugin.h>
#include <spa/utils/dict.h>
#include <spa/param/audio/raw.h>
#include <spa/interfaces/audio/aec.h>

typedef int (*enum_fn)(const struct spa_handle_factory **, uint32_t *);

int main(int argc, char **argv)
{
	if (argc != 6) { fprintf(stderr, "usage: %s so rate mic ref out\n", argv[0]); return 2; }
	const char *so = argv[1];
	uint32_t rate = (uint32_t)atoi(argv[2]);

	void *dl = dlopen(so, RTLD_NOW);
	if (!dl) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
	enum_fn efn = (enum_fn)dlsym(dl, "spa_handle_factory_enum");
	if (!efn) { fprintf(stderr, "no spa_handle_factory_enum\n"); return 1; }

	const struct spa_handle_factory *factory = NULL, *f;
	uint32_t idx = 0;
	while (efn(&f, &idx) == 1) {
		if (f && f->name && strcmp(f->name, "audio.aec") == 0) { factory = f; break; }
	}
	if (!factory) { fprintf(stderr, "audio.aec factory not found\n"); return 1; }

	size_t sz = spa_handle_factory_get_size(factory, NULL);
	struct spa_handle *handle = calloc(1, sz);
	if (spa_handle_factory_init(factory, handle, NULL, NULL, 0) < 0) {
		fprintf(stderr, "handle init failed\n"); return 1;
	}
	struct spa_audio_aec *aec = NULL;
	if (spa_handle_get_interface(handle, SPA_TYPE_INTERFACE_AUDIO_AEC, (void **)&aec) < 0 || !aec) {
		fprintf(stderr, "get_interface AEC failed\n"); return 1;
	}

	struct spa_dict_item items[] = {
		{ "webrtc.gain_control", "false" },
		{ "webrtc.noise_suppression", "false" },
		{ "webrtc.high_pass_filter", "true" },
		{ "webrtc.voice_detection", "true" },
	};
	struct spa_dict args = SPA_DICT_INIT(items, 4);
	struct spa_audio_info_raw info = { 0 };
	info.format = SPA_AUDIO_FORMAT_F32;
	info.rate = rate;
	info.channels = 1;

	int r = spa_audio_aec_init2(aec, &args, &info, &info, &info);
	if (r < 0) r = spa_audio_aec_init(aec, &args, &info);
	if (r < 0) { fprintf(stderr, "aec init failed: %d\n", r); return 1; }
	spa_audio_aec_activate(aec);

	FILE *fm = fopen(argv[3], "rb"), *fr = fopen(argv[4], "rb"), *fo = fopen(argv[5], "wb");
	if (!fm || !fr || !fo) { fprintf(stderr, "file open failed\n"); return 1; }

	uint32_t n = rate / 100; // 10 ms
	float *mic = malloc(n * sizeof(float)), *ref = malloc(n * sizeof(float)), *out = malloc(n * sizeof(float));
	const float *rec_ch[1], *play_ch[1];
	float *out_ch[1];
	rec_ch[0] = mic; play_ch[0] = ref; out_ch[0] = out;
	for (;;) {
		size_t gm = fread(mic, sizeof(float), n, fm);
		size_t gr = fread(ref, sizeof(float), n, fr);
		size_t g = gm < gr ? gm : gr;
		if (g == 0) break;
		if (g < n) { memset(mic + g, 0, (n - g) * sizeof(float)); memset(ref + g, 0, (n - g) * sizeof(float)); }
		if (spa_audio_aec_run(aec, rec_ch, play_ch, out_ch, n) < 0) { fprintf(stderr, "run failed\n"); return 1; }
		fwrite(out, sizeof(float), g, fo);
	}
	fclose(fm); fclose(fr); fclose(fo);
	spa_audio_aec_deactivate(aec);
	spa_handle_clear(handle);
	return 0;
}
