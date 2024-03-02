#include "matmul.h"

#include <cstdio>
#include <vector>

static std::vector<float *> h_pinnedHostMemory;
static float *d_weights {nullptr};
static size_t weights_size {0};
static float *h_temporaryDeviceDataPtr {nullptr};

// Macro for easier CUDA error checking and logging.
#define HANDLE_CUDA_RESULT(FUNC) \
    do { \
        if (cudaError_t result = FUNC; result != cudaSuccess) \
        { \
            fprintf(stderr, "CUDA error with function '%s' at line %d: %s(%d)\n", \
                #FUNC,           \
                __LINE__,        \
                cudaGetErrorName(result), \
                result           \
            ); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static bool isInDeviceMemory(void *ptr, size_t size)
{
    if ((char*)d_weights <= ptr && ptr < (char*)d_weights + weights_size)
    {
        if ((char*)d_weights + weights_size < (char*)ptr + size)
        {
            fprintf(stderr, "Questioned memory is too big for allocated weights: %p/%zd - %p/%zd\n",
                    d_weights, weights_size, ptr, size);
               exit(EXIT_FAILURE);
        }
        return true;
    }

    return false;
}

float *cuda_allocate_device_weights(float *source, size_t size)
{
    HANDLE_CUDA_RESULT(cudaMalloc((void**)&d_weights, size));
    HANDLE_CUDA_RESULT(cudaMemcpy(d_weights, source, size, cudaMemcpyHostToDevice));
    weights_size = size;
    return d_weights;
}

float *cuda_allocate_pinned_memory(size_t size)
{
    float *ptr{nullptr};
    HANDLE_CUDA_RESULT(cudaMallocHost((void**)&ptr, size));
    HANDLE_CUDA_RESULT(cudaMemset(ptr, 0, size));
    h_pinnedHostMemory.push_back(ptr);
    return ptr;
}

void cuda_copy_device_weights_to_host(float *h_destination, float *d_source, size_t size)
{
    HANDLE_CUDA_RESULT(cudaMemcpy(h_destination, d_source, size, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());
}

float* cuda_get_temporary_device_weights(float *d_src, size_t dim)
{

    if (h_temporaryDeviceDataPtr != nullptr)
    {
        free(h_temporaryDeviceDataPtr);
    }

    h_temporaryDeviceDataPtr = (float*)calloc(dim, sizeof *d_src);
    HANDLE_CUDA_RESULT(cudaMemcpy(h_temporaryDeviceDataPtr, d_src, dim * sizeof *d_src, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());

    return h_temporaryDeviceDataPtr;
}

void cuda_free_all_memory()
{
    for (auto d_ptr : h_pinnedHostMemory)
    {
        HANDLE_CUDA_RESULT(cudaFreeHost(d_ptr));
    }

    if (h_temporaryDeviceDataPtr != nullptr)
    {
        free(h_temporaryDeviceDataPtr);
        h_temporaryDeviceDataPtr = nullptr;
    }

    HANDLE_CUDA_RESULT(cudaFree(d_weights));
    d_weights = nullptr;
    weights_size = 0;
}


__global__ void matrixMultiplicationKernel(float* w, float* x, float* out, int n, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum {0.0f};

    if (col < d)
    {
        for (int i = 0; i < n; i++) {
            sum += w[col * n + i] * x[i];
        }
    }

    out[col] = sum;
}

// W (d,n) @ x (n,) -> xout (d,)
void matmul(float *h_out, float *h_x, float *h_w, int n, int d) {
    float *d_w {};
    const size_t size_w {sizeof(float) * (n * d)};

    // If the given Memory is already in the device, we do not need to copy it.
    if (!isInDeviceMemory(h_w, size_w))
    {
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_w, size_w));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice));
    }
    else
    {
        d_w = h_w;
    }

    float *d_x {};
    const size_t size_x {sizeof(float) * (n)};

    // If the given Memory is already in the device, we do not need to copy it.
    if (!isInDeviceMemory(h_x, size_x))
    {
        HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_x, size_x));
        HANDLE_CUDA_RESULT(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    }
    else
    {
        d_x = h_x;
    }

    // Calculate the threads and block size.
    dim3 threadsPerBlock{static_cast<unsigned>(d)};
    dim3 blocksPerGrid{1};
    if (d > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(d) / double(threadsPerBlock.x));
    }

    // Allocate device memory for the result.
    float *d_out{};
    const size_t size_out = sizeof(float) * (d);
    HANDLE_CUDA_RESULT(cudaMalloc((void **) &d_out, size_out));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());

    // Do the actual calculation.
    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_out, n, d);
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());

    // Get the result to host Memory and free only the here allocated memory.
    HANDLE_CUDA_RESULT(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_RESULT(cudaDeviceSynchronize());

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
