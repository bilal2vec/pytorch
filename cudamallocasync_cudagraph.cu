#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define N 8 

__global__ void init(float* dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = 0.0f;
}

__global__ void add(float* dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = dst[i] + 1.0f;
}

int main(void) {
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamCreate(&stream);

    float *src_h, *src_d;

    src_h = (float*)malloc(sizeof(float) * N); 
    for (size_t i = 0; i < N; i++) {
        src_h[i] = 0.0f;
    }

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    //cudaMalloc((void**)&src_d, sizeof(float) * N);
    cudaMallocAsync((void**)&src_d, sizeof(float) * N, stream);
    init<<<1, N, 0, stream>>>(src_d);
    add<<<1, N, 0, stream>>>(src_d);

    cudaStreamEndCapture(stream, &graph);
    cudaStreamSynchronize(stream);
    cudaGraphInstantiate(&instance, graph);

    //cudaGraphLaunch(instance, stream);
    cudaMemcpyAsync(src_h, src_d, sizeof(float) * N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < N; i++) {
        std::cout << src_h[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}