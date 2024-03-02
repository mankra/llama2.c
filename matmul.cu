#include "matmul.h"

#include <cstdio>
#include <vector>

static std::vector<float *> h_pinnedHostMemory;
static float *d_weights {nullptr};
static size_t weights_size {0};
static float *h_temporaryDeviceDataPtr {nullptr};

#define HANDLE_CUDA_RESULT(FUNC) \
    do { \
        if (cudaError_t result = FUNC; result != cudaSuccess) \
        { \
            fprintf(stderr, "Encountered cuda error with function '%s' at line %d: %s(%d)\n", #FUNC, __LINE__, cudaGetErrorName(result), result); \
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
    DBG_PRINTF("Allocated weights: %p / %zd", d_weights, size);
    return d_weights;
}

float *cuda_allocate_pinned_memory(size_t size)
{
    float *ptr{nullptr};
    HANDLE_CUDA_RESULT(cudaMallocHost((void**)&ptr, size));
    HANDLE_CUDA_RESULT(cudaMemset(ptr, 0, size));
    h_pinnedHostMemory.push_back(ptr);

    DBG_PRINTF("Allocated pinned memory: %p / %zd", ptr, size);
    return ptr;
}

void copyDeviceWeightsToHost(void *h_destination, float *d_source, size_t size)
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

    if (col == 0) {
        DBG_PRINTF("dev_w[0] %f dev_w[n*d] %f dev_x[0]: %f dev_x[n-1] %f", w[0], w[n*d], x[0], x[n - 1]);
    }
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
    if (!isInDeviceMemory(h_w, size_w))
    {
        DBG_PRINTF("copy w: %p / %zd h_w[0] %f", h_w, size_w, h_w[0]);
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
        DBG_PRINTF("copy x: %p / %zd h_x[0] %f", h_x, size_x, h_x[0]);
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
