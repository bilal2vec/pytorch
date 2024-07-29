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
  float *h_src, *h_dst, *d_src_1, *d_src_2, *d_dst_1, *d_dst_2;
 
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  h_src = (float*)malloc(sizeof(float) * N);
  h_dst = (float*)malloc(sizeof(float) * N);

  cudaMalloc((void**)&d_src_1, sizeof(float) * N);
  cudaMalloc((void**)&d_src_2, sizeof(float) * N);
  cudaMalloc((void**)&d_dst_1, sizeof(float) * N);
  cudaMalloc((void**)&d_dst_2, sizeof(float) * N);

  for (size_t i = 0; i < N; i++) {
    h_src[i] = 1;
    h_dst[i] = 0.0;
  }
 
  cudaMemcpyAsync(d_src_1, h_src, sizeof(float) * N, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(d_src_2, d_src_1, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream);
  cudaStreamSynchronize(stream);
  
  init<<<1024, 1024, 0, stream>>>(d_dst_1, d_src_2);
  add<<<1024, 1024, 0, stream>>>(d_dst_1, d_dst_1);
  add<<<1024, 1024, 0, stream>>>(d_dst_1, d_dst_1);
  add<<<1024, 1024, 0, stream>>>(d_dst_1, d_dst_1);

  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(d_dst_2, d_dst_1, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(h_dst, d_dst_2, sizeof(float) * N, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
 
  printf("%f\n", h_dst[0]);
  printf("%f\n", h_dst[1024]);

  return 0;

}
