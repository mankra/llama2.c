//
// Interface to the functions necessary for cuda.
//

#ifndef LLAMA2_C_MATMUL_H
#define LLAMA2_C_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

float *allocateDeviceWeights(void *data, size_t size);
float *allocatePinnedHostMemory(size_t size);
void freeDeviceMemoryAndWeights();

void matmul(float *h_out, float *h_x, float *h_w, int n, int d);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //LLAMA2_C_MATMUL_H
