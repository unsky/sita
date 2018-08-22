//
// Created by unsky on 07/08/18.
//
#ifndef SITA_CONTEXT_H
#define SITA_CONTEXT_H
#include <cuda_runtime.h>
#include <cudnn.h>
#include <glog/logging.h>
#include <cstdlib>
#include "macros.h"

namespace sita{


template <typename Dtype> class dataType;
template<> class dataType<float>  {
public:
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
    static float oneval = 1.0;
    static float zeroval = 0.0;
    static const void *one = static_cast<void *>(&dataType<float>::oneval);
    static const void *zero = static_cast<void *>(&dataType<float>::zeroval);
};
template<> class dataType<double> {
public:
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    static double oneval = 1.0;
    static double zeroval = 0.0;
    static const void *one =  static_cast<void *>(&dataType<double>::oneval);
    static const void *zero = static_cast<void *>(&dataType<double>::zeroval);
};

    class Context{
public:
    Context() {}
    ~Context() {}
    //cpu control
    inline static void * cpu_malloc(void * ptr, size_t size){
        ptr = malloc(size);
        CHECK(ptr) << "malloc cpu mem fail";
        return  ptr;
    }
    inline static void cpu_memset(void * ptr, size_t size, int val = 0){

        memset(ptr, val, size);
    }
    inline static void cpu_memcpy(void * dst_ptr, const void * src_ptr, size_t size){
        memcpy(dst_ptr, src_ptr, size);
    }
    inline static void cpu_free(void * data){
        free(data);
    }
    //gpu control
    inline static void * gpu_malloc(void * ptr, size_t size){
        CUDA_CHECK(cudaMalloc(&ptr, size)) ;
        return ptr;
    }

    inline static void gpu_memset(void * ptr, size_t size, int val = 0){
        CUDA_CHECK(cudaMemset(ptr, val, size));
    }
    inline static void gpu_memcpy(void * dst_ptr, const void *src_ptr, size_t size, cudaMemcpyKind kind){
        CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, size, kind));
    }
    inline static void gpu_free(void * data){
        CUDA_CHECK(cudaFree(data));
    }


    //cudnn control

    inline const  static char* cudnn_get_error_string(cudnnStatus_t status) {
        switch (status) {
            case CUDNN_STATUS_SUCCESS:
                return "CUDNN_STATUS_SUCCESS";
            case CUDNN_STATUS_NOT_INITIALIZED:
                return "CUDNN_STATUS_NOT_INITIALIZED";
            case CUDNN_STATUS_ALLOC_FAILED:
                return "CUDNN_STATUS_ALLOC_FAILED";
            case CUDNN_STATUS_BAD_PARAM:
                return "CUDNN_STATUS_BAD_PARAM";
            case CUDNN_STATUS_INTERNAL_ERROR:
                return "CUDNN_STATUS_INTERNAL_ERROR";
            case CUDNN_STATUS_INVALID_VALUE:
                return "CUDNN_STATUS_INVALID_VALUE";
            case CUDNN_STATUS_ARCH_MISMATCH:
                return "CUDNN_STATUS_ARCH_MISMATCH";
            case CUDNN_STATUS_MAPPING_ERROR:
                return "CUDNN_STATUS_MAPPING_ERROR";
            case CUDNN_STATUS_EXECUTION_FAILED:
                return "CUDNN_STATUS_EXECUTION_FAILED";
            case CUDNN_STATUS_NOT_SUPPORTED:
                return "CUDNN_STATUS_NOT_SUPPORTED";
            case CUDNN_STATUS_LICENSE_ERROR:
                return "CUDNN_STATUS_LICENSE_ERROR";
        #if CUDNN_VERSION_MIN(6, 0, 0)
                    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
              return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
        #endif
        }
        return "Unknown cudnn status";
    }


    template <typename Dtype>
    inline static void create_tensor4d_descriptor(cudnnTensorDescriptor_t* desc) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
    }

    template <typename Dtype>
    inline static void set_tensor4d_descriptor(cudnnTensorDescriptor_t* desc,
                                int n, int c, int h, int w,
                                int stride_n, int stride_c, int stride_h, int stride_w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
                                                 n, c, h, w, stride_n, stride_c, stride_h, stride_w));
    }

    template <typename Dtype>
    inline static void set_tensor4d_descriptor(cudnnTensorDescriptor_t* desc,
                                int n, int c, int h, int w) {
        const int stride_w = 1;
        const int stride_h = w * stride_w;
        const int stride_c = h * stride_h;
        const int stride_n = c * stride_c;
        set_tensor4d_descriptor<Dtype>(desc, n, c, h, w,
                               stride_n, stride_c, stride_h, stride_w);
    }

    template <typename Dtype>
    inline  static void create_filter_descriptor(cudnnFilterDescriptor_t* desc,
                                 int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));

        #if CUDNN_VERSION_MIN(5, 0, 0)
                CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
              CUDNN_TENSOR_NCHW, n, c, h, w));
        #else
                CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc, dataType<Dtype>::type,
                                                          CUDNN_TENSOR_NCHW, n, c, h, w));
        #endif
    }

    template <typename Dtype>
    inline static void create_convolution_descriptor(cudnnConvolutionDescriptor_t* conv) {
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
    }

    template <typename Dtype>
    inline static void set_convolution_descriptor(cudnnConvolutionDescriptor_t* conv,
                                   cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
                                   int pad_h, int pad_w, int stride_h, int stride_w) {
        #if CUDNN_VERSION_MIN(6, 0, 0)
                CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
              pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION,
              dataType<Dtype>::type));
        #else
                CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
                            pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
        #endif
    }

//    template <typename Dtype>
//    inline static void create_pooling_desc(cudnnPoolingDescriptor_t* pool_desc,
//                                  PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
//                                  int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
//        switch (poolmethod) {
//            case PoolingParameter_PoolMethod_MAX:
//                *mode = CUDNN_POOLING_MAX;
//                break;
//            case PoolingParameter_PoolMethod_AVE:
//                *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
//                break;
//            default:
//                LOG(FATAL) << "Unknown pooling method.";
//        }
//        CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
//        #if CUDNN_VERSION_MIN(5, 0, 0)
//                CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode,
//                CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
//        #else
//                CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(*pool_desc, *mode,
//                                                           CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
//        #endif
//    }

    template <typename Dtype>
    inline static void create_activation_descriptor(cudnnActivationDescriptor_t* activ_desc,
                                           cudnnActivationMode_t mode) {
        CUDNN_CHECK(cudnnCreateActivationDescriptor(activ_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(*activ_desc, mode,
                                                 CUDNN_PROPAGATE_NAN, Dtype(0)));
    }
};//contxt

}//namespace
#endif
