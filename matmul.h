//
// Interface to the functions necessary for cuda.
//

#ifndef LLAMA2_C_MATMUL_H
#define LLAMA2_C_MATMUL_H

#if defined (DEBUG)
#define DBG_PRINTF(fmt, ...) \
    printf("Debug: %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DBG_PRINTF(fmt, ...) do {} while (0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

float *cuda_allocate_device_weights(float *source, size_t size);
float *cuda_allocate_pinned_memory(size_t size);
void cuda_copy_device_weights_to_host(float *h_destination, float *d_source, size_t size);
float* cuda_get_temporary_device_weights(float *d_src, size_t dim);
void cuda_free_all_memory();


void matmul(float *h_out, float *h_x, float *h_w, int n, int d);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //LLAMA2_C_MATMUL_H
