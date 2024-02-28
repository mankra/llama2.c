#include "matmul.h"

#include <cstdio>
#include <vector>

struct DeviceMemory
{
    float *ptr;
    size_t size;
};

static std::vector<float *> pinnedHostMemory;
static std::vector<DeviceMemory> deviceMemory;

#define HANDLE_CUDA_RESULT(FUNC) \
    do { \
        if (cudaError_t result = FUNC; result != cudaSuccess) \
        { \
            fprintf(stderr, "Encountered cuda error with function '%s' at line %d: %s(%d)\n", #FUNC, __LINE__, cudaGetErrorName(result), result); \
            exit(1); \
        } \
    } while(0)

static bool isInDeviceMemory(float *ptr)
{
    for(const auto& dm : deviceMemory)
    {
        if (dm.ptr <= ptr && ptr < dm.ptr + dm.size)
        {
            return true;
        }
    }

    return false;
}

float *allocateDeviceMemory(float *source, size_t size)
{
    float *ptr{nullptr};
    HANDLE_CUDA_RESULT(cudaMalloc((void**)&ptr, size));
    HANDLE_CUDA_RESULT(cudaMemcpy(ptr, source, size, cudaMemcpyHostToDevice));
    deviceMemory.push_back({ptr, size});
    printf("allocated: %p\n", ptr);
    return ptr;
}

float *allocatePinnedHostMemory(size_t size)
{
    float *ptr{nullptr};
    HANDLE_CUDA_RESULT(cudaMallocHost((void**)&ptr, size));
    pinnedHostMemory.push_back(ptr);
    return ptr;
}

void freeDeviceMemoryAndWeights()
{
    for (auto ptr : pinnedHostMemory)
    {
        HANDLE_CUDA_RESULT(cudaFree(ptr));
    }

    for (auto dm : deviceMemory)
    {
        HANDLE_CUDA_RESULT(cudaFree(dm.ptr));
    }
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
        HANDLE_CUDA_RESULT(cudaGetDeviceCount(&deviceCnt));

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
    HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_out, size_out));

    if (isInDeviceMemory(h_w) == false)
    {
        printf("copy w: %p\n", h_w);
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_w, size_w));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice));
    }
    else
        d_w = h_w;

    if (isInDeviceMemory(h_x) == false)
    {
        printf("copy x: %p\n", h_x);
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

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_out, n, d);
    //HANDLE_CUDA_RESULT(cudaDeviceSynchronize());
    cudaDeviceSynchronize();


    HANDLE_CUDA_RESULT(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));
    //HANDLE_CUDA_RESULT(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

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