#include <stdio.h>
#include <cuda_runtime.h>

// CPU-only function (runs on host)
void addOnCPU(const int *a, const int *b, int *c, int n)
{
    printf("[CPU] addOnCPU() runs on CPU, single-thread loop %d times\n", n);
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// GPU-only kernel: __global__ + <<<>>> means device code
__global__ void addOnGPU(const int *a, const int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        if (i == 0)
        {
            printf("[GPU] addOnGPU() runs on GPU, %d threads in parallel\n",
                   (n + blockDim.x - 1) / blockDim.x * blockDim.x);
        }
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 8;
    int h_a[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_b[N] = {10, 20, 30, 40, 50, 60, 70, 80};
    int h_c_cpu[N] = {0};
    int h_c_gpu[N] = {0};

    printf("========== 1. CPU (Host) ==========\n");
    printf("[CPU] main() runs on CPU\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[CPU] Found GPU: %s (queried from CPU)\n", prop.name);
    printf("[CPU] This printf is still on CPU\n\n");

    addOnCPU(h_a, h_b, h_c_cpu, N);

    printf("[CPU] Result: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_c_cpu[i]);
    printf("\n\n");

    printf("========== 2. GPU (Device) ==========\n");
    printf("[CPU] This line is CPU; launching GPU kernel next...\n\n");

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    addOnGPU<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c_gpu, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n========== 3. Back to CPU ==========\n");
    printf("[CPU] GPU kernel done; this runs on CPU again\n");

    printf("[CPU] Result: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_c_gpu[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("\n========== Done ==========\n");
    return 0;
}
