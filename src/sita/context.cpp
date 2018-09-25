#include "sita/context.h"

namespace sita{

float CudnnDataType<float>::oneval = 1.0;
float CudnnDataType<float>::zeroval = 0.0;
const void* CudnnDataType<float>::one =
    static_cast<void *>(&CudnnDataType<float>::oneval);
const void* CudnnDataType<float>::zero =
    static_cast<void *>(&CudnnDataType<float>::zeroval);

double CudnnDataType<double>::oneval = 1.0;
double CudnnDataType<double>::zeroval = 0.0;
const void* CudnnDataType<double>::one =
    static_cast<void *>(&CudnnDataType<double>::oneval);
const void* CudnnDataType<double>::zero =
    static_cast<void *>(&CudnnDataType<double>::zeroval);

}