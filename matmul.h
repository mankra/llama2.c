//
// Interface to the functions necessary for cuda.
//

#ifndef LLAMA2_C_MATMUL_H
#define LLAMA2_C_MATMUL_H

#if 1
#define DBG_PRINTF(fmt, ...) \
    fprintf(stderr, "Debug: %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DBG_PRINTF(fmt, ...) do {} while (0)
#endif
/*
#if 0
    #define DBG_PRINTF(PRINTF_MSG) do { printf PRINTF_MSG; } while(0)
#else
    #define DBG_PRINTF(PRINTF_MSG) do { ; } while(0)
#endif
 */

#ifdef __cplusplus
extern "C" {
#endif

float *allocateDeviceWeights(float *source, size_t size);
float *allocatePinnedHostMemory(size_t size);
void freeDeviceMemoryAndWeights();

void copyDeviceWeightsToHost(float *destination, float *source, size_t size);

void matmul(float *h_out, float *h_x, float *h_w, int n, int d);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //LLAMA2_C_MATMUL_H
