//
// Created by unsky on 27/06/18.
//

#ifndef SITA_STUFF_COMMON_H
#define SITA_STUFF_COMMON_H

#include <vector>
#include <glog/logging.h>

namespace sita {
#define  cudaMemcpyHostToDevice cpu2gpu
#define  cudaMemcpyDeviceToHost gpu2cpu
#define  cudaMemcpyDeviceToDevice gpu2gpu


const int CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  }

#define CUDA_CHECK(condition) \
/* Code block avoids redefinition of cudaError_t error */ \
do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)


#define CUBLAS_CHECK(condition) \
do { \
  cublasStatus_t status = condition; \
  CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
    << cublasGetErrorString(status); \
} while (0)

#define CURAND_CHECK(condition) \
do { \
  curandStatus_t status = condition; \
  CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
    << curandGetErrorString(status); \
} while (0)

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

//for template
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>; \
  template class classname<int>


}//namespace
#endif //SITA_STUFF_COMMON_H
