#include <stdio.h>
#include <stdlib.h>

extern void matmul(float* xout, float* x, float* w, int n, int d);

void matmul_cpu(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void printMatrix(const char *pfx, float *matrix, size_t rows_, size_t columns_)
{
    printf("%s\n", pfx);
    for(size_t i = 0; i < rows_; i++)
    {
        for(size_t j = 0; j < columns_; j++)
        {
            printf("%f ", matrix[i * columns_ + j]);
        }
        printf("\n");
    }
    printf("\n");
}


#define D 3
#define N 2

#define W_SIZE (D * N)
#define X_SIZE (N * 1)
#define OUT_SIZE (D * 1)

int main(int argc, char *argv[])
{
    float *w = (float *)malloc(W_SIZE);
    float *x = (float *)malloc(X_SIZE);

    w[0] = 1.0;
    w[1] = 4.0;
    w[2] = 2.0;
    w[3] = 5.0;
    w[4] = 3.0;
    w[5] = 6.0;

    x[0] = 1.0;
    x[1] = 2.0;

    printMatrix("W:", w, D, N);
    printMatrix("X:", x, N, 1);


    float *outCuda = (float *)malloc(OUT_SIZE);
    matmul(outCuda, x, w, N, D);

    printMatrix("GPU:", outCuda, D, 1);

    float *out = (float *)malloc(OUT_SIZE);
    matmul_cpu(out, x, w, N, D);

    printMatrix("CPU:", out, D, 1);

    for (size_t i = 0; i < D; i++)
    {
        if (out[i] != outCuda[i])
        {
            printf("Error: Not equal at index %zd: CPU: %f GPU: %f\n", i, out[i], outCuda[i]);
        }
    }

    return EXIT_SUCCESS;
}

