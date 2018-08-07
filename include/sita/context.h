//
// Created by unsky on 07/08/18.
//
#ifndef SITA_CONTEXT_H
#define SITA_CONTEXT_H
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cstdlib>

namespace sita{
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
};//contxt

}//namespace
#endif
