//
// Created by unsky on 02/07/18.
//
#include "sita/stuff/workspace.h"
namespace  sita {

void WorkSpace::device_query(){
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaError = cudaGetDeviceProperties(&deviceProp, i);
        LOG(INFO) << "===============================================================";
        LOG(INFO) << "设备 " << i  << " 的主要属性： ";
        LOG(INFO) << "设备显卡型号： " << deviceProp.name;
        LOG(INFO) << "设备全局内存总量（以MB为单位）： " << deviceProp.totalGlobalMem / 1024 / 1024;
        LOG(INFO) << "设备上一个线程块（Block）中可用的最大共享内存（以KB为单位）： " << deviceProp.sharedMemPerBlock / 1024;
        LOG(INFO) << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock;
        LOG(INFO) << "设备上一个线程块（Block）可包含的最大线程数量： " << deviceProp.maxThreadsPerBlock;
        LOG(INFO) << "设备的计算功能集（Compute Capability）的版本号： " << deviceProp.major << "." << deviceProp.minor;
        LOG(INFO) << "设备上多处理器的数量： " << deviceProp.multiProcessorCount;
    }
    LOG(INFO) << "===============================================================";
}
void WorkSpace::set_device(int gpu_id){
    _gpu_id = gpu_id;
    CUDA_CHECK(cudaSetDevice(_gpu_id));
    LOG(INFO) << "正在使用GPU:" << _gpu_id << "进行计算...";
}

template <typename Dtype>
std::pair<int, Tensor<Dtype> * > GlobalWorkSpace<Dtype>::fetch_temp_tensor() {
    if (_temp_tensor.size() == 0) {
        Tensor<Dtype> temp_tensor;
        _temp_tensor.push_back(temp_tensor);
        _temp_tensor_control.push_back(std::make_pair(&(_temp_tensor[0]), true));
        return std::make_pair(0, &(_temp_tensor[0]));
    } else {
        int released_id = -1;
        for (int i = 0; i < _temp_tensor.size(); i++) {
            if (_temp_tensor_control[i].second == false) {
                released_id = i;
            }
        }
        if (released_id == -1) {
            Tensor<Dtype> temp_tensor;
            _temp_tensor.push_back(temp_tensor);
            _temp_tensor_control.push_back(std::make_pair(&(_temp_tensor[_temp_tensor.size() - 1]), true));
            return std::make_pair(0, &(_temp_tensor[_temp_tensor.size()-1]));

        } else {
            _temp_tensor_control[released_id].second = true;
            return std::make_pair(released_id, &_temp_tensor[released_id]);
        }
    }
}
template <typename Dtype>
void GlobalWorkSpace<Dtype>::release_temp_tensor(int released_id) {
    _temp_tensor_control[released_id].second = false;
}

template <typename Dtype>
float GlobalWorkSpace<Dtype>::temp_tensor_memory_size(){
    int memory_size = 0;
    for(int i = 0; i < _temp_tensor.size(); i++){
        memory_size += (_temp_tensor[i].count() * sizeof(Dtype));
    }
    return memory_size/(1024 * 1024 * 8);
}

INSTANTIATE_CLASS(GlobalWorkSpace);
}
