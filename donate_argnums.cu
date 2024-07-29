#include "stdio.h"

#include <vector>
#include <set>

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
  float *h_src_1, *h_src_2, *h_dst, *d_src_1, *d_src_2, *d_dst;
  std::vector<cudaGraphNode_t> _node_list;

  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  cudaStreamCreate(&stream);

  h_src_1 = (float*)malloc(sizeof(float) * N);
  h_src_2 = (float*)malloc(sizeof(float) * N);
  h_dst = (float*)malloc(sizeof(float) * N);

  cudaMalloc((void**)&d_src_1, sizeof(float) * N);
  cudaMalloc((void**)&d_src_2, sizeof(float) * N);
  cudaMalloc((void**)&d_dst, sizeof(float) * N);

  for (size_t i = 0; i < N; i++) {
    h_src_1[i] = 1;
    h_src_2[i] = 2;
    h_dst[i] = 0.0;
  }

  cudaMemcpyAsync(d_src_1, h_src_1, sizeof(float) * N, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_src_2, h_src_2, sizeof(float) * N, cudaMemcpyHostToDevice, stream);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  
  init<<<1024, 1024, 0, stream>>>(d_dst, d_src_1);
  add<<<1024, 1024, 0, stream>>>(d_dst, d_dst);

  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph);

  cudaGraphLaunch(instance, stream);
  
  cudaMemcpyAsync(h_dst, d_dst, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  printf("%f\n", h_dst[0]);
  printf("%f\n", h_dst[1024]);

  size_t n_nodes;
  std::set<float*> dynamic_input_ptrs;

  dynamic_input_ptrs.insert(d_src_1); 


  cudaGraphGetNodes(graph, nullptr, &n_nodes);

  std::vector<cudaGraphNode_t> nodes(n_nodes);
  cudaGraphGetNodes(graph, &nodes[0], &n_nodes);

  for (auto node : nodes) {
    cudaKernelNodeParams p;
    cudaGraphKernelNodeGetParams(node, &p);
    float** dst = ((float***)p.kernelParams)[0];
    float** src = ((float***)p.kernelParams)[1];

    printf("%p %p\n", *dst, *src);
  }

  printf("d_dst: %p d_src_1: %p\n", d_dst, d_src_1);

  return 0;
}
