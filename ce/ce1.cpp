#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 100000000;

int main()
{
    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    auto start = high_resolution_clock::now();

    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }

    auto end = high_resolution_clock::now();

    cout << "CPU耗时："
         << duration_cast<milliseconds>(end - start).count()
         << " ms" << endl;

    delete[] a;
    delete[] b;
    delete[] c;
}