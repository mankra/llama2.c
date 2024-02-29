#include "matmul.h"

#include <cstdio>
#include <vector>

static std::vector<float *> pinnedHostMemory;
static float *weights {nullptr};
static size_t weights_size {0};

#define HANDLE_CUDA_RESULT(FUNC) \
    do { \
        if (cudaError_t result = FUNC; result != cudaSuccess) \
        { \
            fprintf(stderr, "Encountered cuda error with function '%s' at line %d: %s(%d)\n", #FUNC, __LINE__, cudaGetErrorName(result), result); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static bool isInDeviceMemory(float *ptr, size_t size)
{
    // check before weights to avoid wrong memory assumption.
#if 0
    for (auto p : pinnedHostMemory)
    {
        if (ptr == p) {
            return false;
        }
    }
#endif

    if (weights <= ptr && ptr < weights + weights_size)
    {
        if (weights + weights_size < ptr + size)
        {
            fprintf(stderr, "Questioned memory is too big for allocated weights: %p/%zd - %p/%zd\n",
                weights, weights_size, ptr, size);
               exit(EXIT_FAILURE);
        }
        return true;
    }

    return false;
}

float *allocateDeviceWeights(float *source, size_t size)
{
    HANDLE_CUDA_RESULT(cudaMalloc((void**)&weights, size));
    HANDLE_CUDA_RESULT(cudaMemcpy(weights, source, size, cudaMemcpyHostToDevice));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());
    weights_size = size;
    DBG_PRINTF("Allocated weights: %p / %zd", weights, size);
    return weights;
}

float *allocatePinnedHostMemory(size_t size)
{
    float *ptr{nullptr};
    HANDLE_CUDA_RESULT(cudaMallocHost((void**)&ptr, size));
    HANDLE_CUDA_RESULT(cudaMemset(ptr, 0, size));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());
    pinnedHostMemory.push_back(ptr);

    DBG_PRINTF("Allocated pinned memory: %p / %zd", ptr, size);
    return ptr;
}

void freeDeviceMemoryAndWeights()
{
    for (auto ptr : pinnedHostMemory)
    {
        HANDLE_CUDA_RESULT(cudaFreeHost(ptr));
    }

    HANDLE_CUDA_RESULT(cudaFree(weights));
    weights = nullptr;
    weights_size = 0;
}

void copyDeviceWeightsToHost(float *destination, float *source, size_t size)
{
    HANDLE_CUDA_RESULT(cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());
}

__global__ void matrixMultiplicationKernel(float* w, float* x, float* out, int n, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum {0.0f};

    if (col < d)
    {
        for (int i = 0; i < n; i++) {
            // DBG_PRINTF("COL: %d i: %d w: %f x: %f", col, i, w[col * n + i], x[i]);
            sum += w[col * n + i] * x[i];
        }
    }

    // DBG_PRINTF("COL: %d n: %d d: %d sum: %f", col, n, d, sum);
    out[col] = sum;
}

// W (d,n) @ x (n,) -> xout (d,)
void matmul(float *h_out, float *h_x, float *h_w, int n, int d) {
    float *d_w {};
    const size_t size_w {sizeof(float) * (n * d)};
    if (!isInDeviceMemory(h_w, size_w))
    {
        DBG_PRINTF("copy w: %p / %zd", h_w, size_w);
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_w, size_w));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice));
    }
    else
    {
        DBG_PRINTF("use w: %p / %zd", h_w, size_w);
        d_w = h_w;
    }

    float *d_x {};
    const size_t size_x {sizeof(float) * (n)};
    if (!isInDeviceMemory(h_x, size_x))
    {
        DBG_PRINTF("copy x: %p / %zd", h_x, size_x);
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_x, size_x));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    }
    else
    {
        DBG_PRINTF("use x: %p / %zd", h_x, size_x);
        d_x = h_x;
    }

    dim3 threadsPerBlock{static_cast<unsigned>(d)};
    dim3 blocksPerGrid{1};
    if (d > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(d) / double(threadsPerBlock.x));
    }

    // Allocate device memory
    float *d_out{};
    const size_t size_out = sizeof(float) * (d);
    HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_out, size_out));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_out, n, d);
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());


    HANDLE_CUDA_RESULT(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());

    // Deallocate device memory
    if (d_x != h_x)
    {
        HANDLE_CUDA_RESULT(cudaFree(d_x));
    }
    if (d_w != h_w)
    {
        HANDLE_CUDA_RESULT(cudaFree(d_w));
    }
    HANDLE_CUDA_RESULT(cudaFree(d_out));
}