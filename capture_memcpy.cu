#include "stdio.h"
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024 * 1024

__global__ void init(float* dst, float* src) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = src[idx];
}

__global__ void add(float* dst, float* src) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] += src[idx];
}

int main(void) {
  float *h_src, *h_dst, *d_src, *d_dst;
  std::vector<cudaGraphNode_t> _node_list;

  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  cudaStreamCreate(&stream);

  h_src = (float*)malloc(sizeof(float) * N);
  h_dst = (float*)malloc(sizeof(float) * N);

  cudaMalloc((void**)&d_src, sizeof(float) * N);
  cudaMalloc((void**)&d_dst, sizeof(float) * N);

  for (size_t i = 0; i < N; i++) {
    h_src[i] = 1;
    h_dst[i] = 0.0;
  }

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  
  cudaMemcpyAsync(d_src, h_src, sizeof(float) * N, cudaMemcpyHostToDevice, stream);
  
  init<<<1024, 1024, 0, stream>>>(d_dst, d_src);
  add<<<1024, 1024, 0, stream>>>(d_dst, d_dst);
  add<<<1024, 1024, 0, stream>>>(d_dst, d_dst);
  add<<<1024, 1024, 0, stream>>>(d_dst, d_dst);
  
  cudaStreamEndCapture(stream, &graph);
  cudaStreamSynchronize(stream);
  cudaGraphInstantiate(&instance, graph);

  cudaGraphLaunch(instance, stream);
  cudaMemcpyAsync(h_dst, d_dst, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  printf("%f\n", h_dst[0]);
  printf("%f\n", h_dst[1024]);

  return 0;
}
