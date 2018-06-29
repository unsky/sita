//
// Created by unsky on 27/06/18.
//

#ifndef SITA_STUFF_COMMON_H
#define SITA_STUFF_COMMON_H

#include <vector>
#include <glog/logging.h>
#include <cuda_runtime.h>
namespace sita {
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

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

}//namespace
#endif //SITA_STUFF_COMMON_H
