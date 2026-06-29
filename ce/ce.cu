#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 10000000;

// GPU Kernel
__global__ void add(int* a, int* b, int* c)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N)
    {
        c[id] = a[id] + b[id];
    }
}

int main()
{
    // ===========================
    // CPU Memory
    // ===========================
    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];
    int* cpu_c = new int[N];

    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i;
    }

    // ===========================
    // CPU Test
    // ===========================
    auto cpu_start = high_resolution_clock::now();

    for (int i = 0; i < N; i++)
    {
        cpu_c[i] = h_a[i] + h_b[i];
    }

    auto cpu_end = high_resolution_clock::now();

    double cpu_time =
        duration<double, milli>(cpu_end - cpu_start).count();

    // ===========================
    // GPU Memory
    // ===========================
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // ===========================
    // CUDA Event（GPU真实计算时间）
    // ===========================
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto total_start = high_resolution_clock::now();

    cudaEventRecord(start);

    add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    auto total_end = high_resolution_clock::now();

    float gpu_kernel_time = 0.0f;

    cudaEventElapsedTime(&gpu_kernel_time, start, stop);

    double gpu_total_time =
        duration<double, milli>(total_end - total_start).count();

    cudaMemcpy(h_c,
               d_c,
               N * sizeof(int),
               cudaMemcpyDeviceToHost);

    // ===========================
    // Check Result
    // ===========================
    bool ok = true;

    for (int i = 0; i < N; i++)
    {
        if (cpu_c[i] != h_c[i])
        {
            ok = false;
            break;
        }
    }

    // ===========================
    // Print
    // ===========================
    cout << "===============================" << endl;

    cout << "CPU Time        : "
         << cpu_time
         << " ms" << endl;

    cout << "GPU Kernel Time : "
         << gpu_kernel_time
         << " ms" << endl;

    cout << "GPU Total Time  : "
         << gpu_total_time
         << " ms" << endl;

    cout << "Kernel Speedup  : "
         << cpu_time / gpu_kernel_time
         << " x" << endl;

    cout << "Total Speedup   : "
         << cpu_time / gpu_total_time
         << " x" << endl;

    cout << "Result          : "
         << (ok ? "Correct" : "Wrong")
         << endl;

    cout << "===============================" << endl;

    // ===========================
    // Free Memory
    // ===========================
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] cpu_c;

    return 0;
}