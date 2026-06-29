#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from GPU! Thread %u (Block %u, Thread %u)\n",
           (unsigned)id,
           (unsigned)blockIdx.x,
           (unsigned)threadIdx.x);
}

int main()
{
    printf("========== CPU ==========\n");

    helloGPU<<<4,8>>>();

    cudaDeviceSynchronize();

    printf("========== END ==========\n");

    return 0;
}