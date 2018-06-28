//
// Created by unsky on 27/06/18.
//
#include <cuda_runtime.h>
#include "sita/stuff/memcontrol.h"

namespace sita{

    MemControl::MemControl(): _size(0), _head_at(UNINIT){};
    MemControl::MemControl(size_t size):_size(size), _head_at(UNINIT){};
    MemControl::~MemControl(){
        if (_ptr_cpu){
            free(_ptr_cpu);
        }
        if(_ptr_gpu){
            cudaFree(_ptr_gpu);
        }
    }

    const void* MemControl::cpu_data(){
        push_data_to_cpu();
        return (const void*)_ptr_cpu;
    }

    const void* MemControl::gpu_data(){
        push_data_to_gpu();
        return (const void*)_ptr_gpu;
    }

    void * MemControl::mutable_cpu_data()
    {
        push_data_to_cpu();
        return _ptr_cpu;
    }

    void * MemControl::mutable_gpu_data()
    {
        push_data_to_gpu();
        return _ptr_gpu;
    }

    void MemControl::push_data_to_cpu(){
        switch(_head_at){
            case UNINIT:
                _ptr_cpu = malloc(_size);
                CHECK(_ptr_cpu)<<"malloc cpu mem fail";
                memset(_ptr_cpu, 0, _size);
                _head_at = CPU;
                break;
            case GPU:
                if (_ptr_cpu == NULL) {
                    _ptr_cpu = malloc(_size);
                    CHECK(_ptr_cpu)<<"malloc cpu mem fail";
                }
                CUDA_CHECK(cudaMemcpy(_ptr_cpu, _ptr_gpu, _size, cudaMemcpyDeviceToHost));
                _head_at = SYNCED;
                break;
            case CPU:
            default:
                break;
        }
    }

    void MemControl::push_data_to_gpu(){
        switch(_head_at){
            case UNINIT:
                CUDA_CHECK(cudaMalloc(&_ptr_gpu, _size));
                CUDA_CHECK(cudaMemset(_ptr_gpu, 0, _size));
                _head_at = GPU;
                break;
            case CPU:
                if(_ptr_gpu == NULL){
                    CUDA_CHECK(cudaMalloc(&_ptr_gpu, _size));
                }
                CUDA_CHECK(cudaMemcpy(_ptr_gpu, _ptr_cpu, _size, cudaMemcpyHostToDevice));
                _head_at = SYNCED;
                break;
            case GPU:
            default:
                break;
        }
    }


}//namespace;
