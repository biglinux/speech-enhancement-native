/* Check the built Rust plugin through the installed SPA C headers.
 * No model or sound device is needed.
 * cc -std=c11 -Wall -Wextra -Werror $(pkg-config --cflags libspa-0.2) \
 *   gtcrn-aec/tools/spa-aec-abi-check.c -ldl -o spa-aec-abi-check
 * ./spa-aec-abi-check target/release/libspa_aec_gtcrn.so
 */
/* spa/utils/string.h uses locale_t (newlocale) — requires _GNU_SOURCE. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spa/support/plugin.h>
#include <spa/interfaces/audio/aec.h>

/* Unlike assert(), this evaluates calls even in a -DNDEBUG release build. */
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    return 1; \
} } while (0)

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s plugin.so\n", argv[0]); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    typedef int (*enum_fn)(const struct spa_handle_factory **, uint32_t *);
    enum_fn enumerate = (enum_fn)dlsym(lib, "spa_handle_factory_enum");
    CHECK(enumerate);
    const struct spa_handle_factory *factory = NULL;
    uint32_t index = 0;
    CHECK(enumerate(&factory, &index) == 1 && factory);
    CHECK(factory->version == SPA_VERSION_HANDLE_FACTORY);
    CHECK(factory->name && strcmp(factory->name, "audio.aec") == 0);
    CHECK(factory->get_size && factory->init && factory->enum_interface_info);
    CHECK(enumerate(&factory, &index) == 0);
    const struct spa_interface_info *interface_info = NULL;
    uint32_t interface_index = 0;
    CHECK(factory->enum_interface_info(factory, &interface_info, &interface_index) == 1);
    CHECK(interface_info && strcmp(interface_info->type, SPA_TYPE_INTERFACE_AUDIO_AEC) == 0);
    CHECK(factory->enum_interface_info(factory, &interface_info, &interface_index) == 0);
    struct spa_handle *handle = calloc(1, factory->get_size(factory, NULL));
    CHECK(handle);
    CHECK(factory->init(factory, handle, NULL, NULL, 0) == 0);
    struct spa_audio_aec *aec = NULL;
    CHECK(handle->get_interface(handle, SPA_TYPE_INTERFACE_AUDIO_AEC, (void **)&aec) == 0);
    CHECK(aec && aec->iface.version == SPA_VERSION_AUDIO_AEC);
    CHECK(aec->name && strcmp(aec->name, "gtcrn") == 0);
    CHECK(aec->info == NULL);
    const struct spa_audio_aec_methods *methods = aec->iface.cb.funcs;
    CHECK(methods && methods->version == SPA_VERSION_AUDIO_AEC_METHODS);
    CHECK(methods->init && methods->init2 && methods->run);
    CHECK(methods->activate && methods->deactivate);
    CHECK(!methods->add_listener && !methods->set_props && !methods->enum_props);
    CHECK(!methods->get_params && !methods->set_params);
    /* This contract check does not load a model or start an audio thread. */
    CHECK(spa_audio_aec_run(aec, NULL, NULL, NULL, 0) == -EINVAL);
    CHECK(aec->latency && strcmp(aec->latency, "768/48000") == 0);
    struct spa_audio_info_raw bad = SPA_AUDIO_INFO_RAW_INIT(
        .format = SPA_AUDIO_FORMAT_F32P, .rate = 44100, .channels = 1);
    CHECK(spa_audio_aec_init(aec, NULL, &bad) == -EINVAL);
    struct spa_audio_info_raw good = SPA_AUDIO_INFO_RAW_INIT(
        .format = SPA_AUDIO_FORMAT_F32P, .rate = 48000, .channels = 1);
    /* Each init2 argument must be checked, not only the first one. A stereo
     * stream is not an error there: init2 rewrites it to mono. */
    CHECK(spa_audio_aec_init2(aec, NULL, &bad, &good, &good) == -EINVAL);
    CHECK(spa_audio_aec_init2(aec, NULL, &good, &bad, &good) == -EINVAL);
    CHECK(spa_audio_aec_init2(aec, NULL, &good, &good, &bad) == -EINVAL);
    void *unsupported = aec;
    CHECK(handle->get_interface(handle, "not-an-interface", &unsupported) == -ENOTSUP);
    CHECK(unsupported == NULL);
    CHECK(handle->clear(handle) == 0);
    /* The same numbers are compile-time assertions in spa-aec-gtcrn/src/lib.rs. */
    if (sizeof(void *) == 8) {
        CHECK(sizeof(struct spa_handle) == 24);
        CHECK(sizeof(struct spa_handle_factory) == 48);
        CHECK(sizeof(struct spa_audio_aec) == 56);
        CHECK(sizeof(struct spa_audio_aec_methods) == 88);
        CHECK(sizeof(struct spa_audio_info_raw) == 272);
        CHECK(offsetof(struct spa_audio_aec_methods, init) == 16);
        CHECK(offsetof(struct spa_audio_aec_methods, run) == 24);
        CHECK(offsetof(struct spa_audio_aec_methods, init2) == 80);
    }
    free(handle);
    dlclose(lib);
    puts("PASS: C header / Rust SPA interface contract");
    return 0;
}
