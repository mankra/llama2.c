#include <stdio.h>

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

extern "C" {

// W (d,n) @ x (n,) -> xout (d,)
void matmul(float *h_out, float *h_x, float *h_w, int n, int d) {
    static bool isCudaChecked {false};

    if (isCudaChecked == false) {
        int deviceCnt;
        cudaError_t ret = cudaGetDeviceCount(&deviceCnt);
        if (ret != 0)
        {
            fprintf(stderr, "Could not get CUDA device count: %s(%d)\n", cudaGetErrorName(ret), ret);
            exit(1);
        }

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
    cudaMalloc((void **) &d_w, size_w);
    cudaMalloc((void **) &d_x, size_x);
    cudaMalloc((void **) &d_out, size_out);

    cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock{static_cast<unsigned>(d)};
    dim3 blocksPerGrid{1};
    if (d > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(d) / double(threadsPerBlock.x));
    }

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_out, n, d);
    //cudaDeviceSynchronize();


    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();

    // Deallocate device memory
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
}

} // extern "C"