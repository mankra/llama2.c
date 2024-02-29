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

void printVector(const char *prefix, float* vector, size_t size)
{
    printf("%s size: %zd First floats: %f %f %f %f %f %f",
        prefix,
        size,
        vector[0],
        vector[1],
        vector[2],
        vector[3],
        vector[4],
        vector[5]
    );
}

float *allocateDeviceWeights(float *source, size_t size)
{
    HANDLE_CUDA_RESULT(cudaMalloc((void**)&weights, size));
    HANDLE_CUDA_RESULT(cudaMemcpy(weights, source, size, cudaMemcpyHostToDevice));
    weights_size = size;
    return weights;
}

float *allocatePinnedHostMemory(size_t size)
{
    float *ptr{nullptr};
    HANDLE_CUDA_RESULT(cudaMallocHost((void**)&ptr, size));
    HANDLE_CUDA_RESULT(cudaMemset(ptr, 0, size));
    pinnedHostMemory.push_back(ptr);
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
            //DBG_PRINTF(("COL: %d i: %d w: %f x: %f\n", col, i, w[col*n+i], x[i]));
            sum += w[col * n + i] * x[i];
        }
    }

    //DBG_PRINTF(("COL: %d n: %d d: %d sum: %f\n", col, n, d, sum));
    out[col] = sum;
}

// W (d,n) @ x (n,) -> xout (d,)
void matmul(float *h_out, float *h_x, float *h_w, int n, int d) {
    static bool isCudaChecked {false};

    if (isCudaChecked == false) {
        int deviceCnt;
        HANDLE_CUDA_RESULT(cudaGetDeviceCount(&deviceCnt));

        if (deviceCnt < 1) {
            fprintf(stderr, "No CUDA devices found.\n");
            exit(EXIT_FAILURE);
        }

        isCudaChecked = true;
    }

    const size_t size_w = sizeof(float) * (n * d);
    const size_t size_x = sizeof(float) * (n);
    const size_t size_out = sizeof(float) * (d);

    float *d_x{};
    float *d_w{};
    float *d_out{};

    if (isInDeviceMemory(h_w, size_w) == false)
    {
        DBG_PRINTF(("copy w: %p\n", h_w));
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_w, size_w));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice));
    }
    else
    {
        d_w = h_w;
    }

    if (isInDeviceMemory(h_x, size_x) == false)
    {
        DBG_PRINTF(("copy x: %p\n", h_x));
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_x, size_x));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    }
    else
        d_x = h_x;

    dim3 threadsPerBlock{static_cast<unsigned>(d)};
    dim3 blocksPerGrid{1};
    if (d > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(d) / double(threadsPerBlock.x));
    }

    // Allocate device memory
    HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_out, size_out));

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