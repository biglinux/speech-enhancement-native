/* SPDX-License-Identifier: MIT OR Apache-2.0
 * C API for embedding and reference comparison.
 * A handle must not be used from two threads at once. Buffers are caller-owned and f32.
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
 * Input/output each cover samples floats; disjoint or the same address.
 * No partial overlap. 0 dB = aligned dry; 100 dB = fully enhanced.
 */
int32_t dpdfnet_native_process(void *handle, const float *input, float *output,
                              size_t samples, float attenuation_db);
/* 481 complex bins as [real,imag] => 962 floats. Unnormalized real-FFT convention.
 * Do not mix this diagnostic interface with PCM processing without reset().
 * Returns 0 on success, -1 for null argument, -2 when the result is not finite.
 * An internal failure also returns -2, zeroes the output and latches the fault
 * that dpdfnet_native_process reports until reset().
 */
int32_t dpdfnet_native_spectrum(void *handle, const float *input, float *output);
/* Intermediate values of the last processed frame, for reference comparison.
 * Same thread only. Returns the length; NULL output is a length query, and
 * nothing is copied when capacity is smaller. Unknown IDs return 0.
 *   0 magnitude features   481     7 df1           48x64
 *   1 complex features     96x2    8 erb_dual      40x64
 *   2 erb0                 480x64  9 df_dual       48x64
 *   3 erb1                 160x64  10 embedding    512
 *   4 erb2                 80x64   11 mask         481
 *   5 erb3                 40x64   12 coefficients 96x5x2
 *   6 df0                  96x64   13 spectrum     481x2
 */
size_t dpdfnet_native_trace(const void *handle, size_t id, float *output, size_t capacity);
size_t dpdfnet_native_latency_samples(void);
#ifdef __cplusplus
}
#endif
#endif
