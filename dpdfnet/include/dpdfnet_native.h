/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Diagnostic/embedding ABI; preprocessing and model execution live entirely in Rust.
 * A handle must never be used concurrently. Buffers are caller-owned and f32.
 */
#ifndef DPDFNET_NATIVE_H
#define DPDFNET_NATIVE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void *dpdfnet_native_create(const char *model_directory_utf8); /* NULL on load error */
void dpdfnet_native_destroy(void *handle);
void dpdfnet_native_reset(void *handle);
/* Returns 0 on success, -1 for null argument, -2 for latched processing fault.
 * Input/output each cover samples floats; disjoint OR exact same address.
 * No partial overlap. 0 dB = aligned dry; 100 dB = fully enhanced.
 * Internal state is always processed; changing this control never skips inference.
 */
int32_t dpdfnet_native_process(void *handle, const float *input, float *output,
                              size_t samples, float attenuation_db);
/* 481 complex bins as [real,imag] => 962 floats. Unnormalized real-FFT convention.
 * Do not mix this diagnostic interface with PCM processing without reset().
 */
int32_t dpdfnet_native_spectrum(void *handle, const float *input, float *output);
/* Same thread only. NULL output is a length query. IDs are documented in docs/ARCHITECTURE.md. */
size_t dpdfnet_native_trace(const void *handle, size_t id, float *output, size_t capacity);
size_t dpdfnet_native_latency_samples(void);
#ifdef __cplusplus
}
#endif
#endif
