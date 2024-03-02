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

float *allocateDeviceWeights(float *source, size_t size);
float *allocatePinnedHostMemory(size_t size);
void copyDeviceWeightsToHost(float *h_destination, float *d_source, size_t size);
float* getTemporaryDeviceValues(float *d_src, size_t dim);
void freeDeviceMemoryAndWeights();


void matmul(float *h_out, float *h_x, float *h_w, int n, int d);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //LLAMA2_C_MATMUL_H
