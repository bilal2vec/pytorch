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
 
  cudaGraph_t graph_so_far;
  cudaStreamCaptureStatus capture_status; 
  cudaKernelNodeParams params;
  cudaGraphNode_t init_kernel_node;
  const cudaGraphNode_t* deps;
  size_t dep_count;

  params.func = reinterpret_cast<void*>(init);
  params.blockDim = {static_cast<unsigned int>(1024), 1, 1};
  params.gridDim = {static_cast<unsigned int>(1024), 1, 1};
  params.sharedMemBytes = 0;
  void* args_1[] = {&d_dst, &d_src_1};
  params.kernelParams = args_1;
  params.extra = nullptr;

  cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &graph_so_far, &deps, &dep_count);

  cudaGraphAddKernelNode(&init_kernel_node, graph_so_far, deps, dep_count, &params);
  _node_list.push_back(init_kernel_node);
  cudaStreamUpdateCaptureDependencies(stream, &init_kernel_node, 1, 1);

//  add<<<1024, 1024, 0, stream>>>(d_dst, d_src_1);
  add<<<1024, 1024, 0, stream>>>(d_dst, d_dst);

  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph);

  cudaGraphLaunch(instance, stream);
  cudaMemcpyAsync(h_dst, d_dst, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  printf("%f\n", h_dst[0]);
  printf("%f\n", h_dst[1024]);

  params.kernelParams[1] = &d_src_2;

  cudaGraphExecKernelNodeSetParams(instance, _node_list[0], &params);

  cudaGraphLaunch(instance, stream);
  cudaMemcpyAsync(h_dst, d_dst, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  printf("%f\n", h_dst[0]);
  printf("%f\n", h_dst[1024]);

  return 0;
}
