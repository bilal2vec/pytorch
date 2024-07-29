#include "stdio.h"
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

int main(void) {
  printf("sizeof(cudaGraphNode_t): %d\n", sizeof(cudaGraphNode_t));

  return 0;
}
