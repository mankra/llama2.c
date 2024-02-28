#include "matmul.h"

#include <cstdio>
#include <vector>

static float *weights { nullptr };
static std::vector<float *> deviceMemory;

static void handleCudaResult(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "Could not get CUDA device count: %s(%d)\n", cudaGetErrorName(result), result);
        exit(1);
    }
}

float *allocateDeviceWeights(void *data, size_t size)
{
    if (weights)
    {
        handleCudaResult(cudaFree(weights));
    }

    handleCudaResult(cudaMalloc((void**)&weights, size));
    handleCudaResult(cudaMemcpy(weights, data, size, cudaMemcpyHostToDevice));

    return weights;
}

float *allocatePinnedHostMemory(size_t size)
{
    float *ptr{nullptr};
    handleCudaResult(cudaMallocHost((void**)&ptr, size));
    deviceMemory.push_back(ptr);
    return ptr;
}

void freeDeviceMemoryAndWeights()
{
    if (weights)
    {
        handleCudaResult(cudaFree(weights));
        weights = nullptr;
    }

    for (auto ptr : deviceMemory)
    {
        handleCudaResult(cudaFree(ptr));
    }
    deviceMemory.clear();
}

__global__ void matrixMultiplicationKernel(float* w, float* x, float* out, int n, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum {0.0f};

    if (col < d)
    {
        for (int i = 0; i < n; i++) {
            //printf("COL: %d i: %d w: %f x: %f\n", col, i, w[col*n+i], x[i]);
            sum += w[col * n + i] * x[i];
        }
    }

    //printf("COL: %d n: %d d: %d sum: %f\n", col, n, d, sum);
    out[col] = sum;
}

// W (d,n) @ x (n,) -> xout (d,)
void matmul(float *h_out, float *h_x, float *h_w, int n, int d) {
    static bool isCudaChecked {false};

    if (isCudaChecked == false) {
        int deviceCnt;
        handleCudaResult(cudaGetDeviceCount(&deviceCnt));

        if (deviceCnt < 1) {
            fprintf(stderr, "No CUDA devices found.\n");
            exit(1);
        }

        isCudaChecked = true;
    }

    const size_t size_w = sizeof(float) * (n * d);
    const size_t size_x = sizeof(float) * (n);
    const size_t size_out = sizeof(float) * (d);

    float *d_x{};
    float *d_w{};
    float *d_out{};

    // Allocate device memory
    handleCudaResult(cudaMalloc((void **) &d_w, size_w));
    handleCudaResult(cudaMalloc((void **) &d_x, size_x));
    handleCudaResult(cudaMalloc((void **) &d_out, size_out));

    handleCudaResult(cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice));
    handleCudaResult(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock{static_cast<unsigned>(d)};
    dim3 blocksPerGrid{1};
    if (d > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(d) / double(threadsPerBlock.x));
    }

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_out, n, d);
    handleCudaResult(cudaDeviceSynchronize());


    handleCudaResult(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));
    handleCudaResult(cudaDeviceSynchronize());

    // Deallocate device memory
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
}