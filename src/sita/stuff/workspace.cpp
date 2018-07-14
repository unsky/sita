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

// flow tensor
//std::map<std::string, std::pair<Tensor<Dtype>, int> > _flow_tensor;
template <typename Dtype>
void GlobalWorkSpace<Dtype>::init_input(std::string name){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            has_flow_tensor = true;
        }
    }
    if (has_flow_tensor == false){
        Tensor<Dtype> t;
        _flow_tensor[name] = std::make_pair(t, 0);
    }
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::init_output(std::string name){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            has_flow_tensor = true;
        }
    }
    if (has_flow_tensor == false){
        Tensor<Dtype> t;
        _flow_tensor[name] = std::make_pair(t, 0);
    }
}

template <typename Dtype>
Tensor<Dtype>* GlobalWorkSpace<Dtype>::forward_fetch_input(std::string name, bool has_param = true, bool is_data_op = false){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            if(has_param == true || is_data_op == true)
                it->second.second++;
            return &(it->second.first);
        }
    }
    LOG(FATAL) << "no this input in flow tensors, do you have init it?" << flow_tensor_list();
}

template <typename Dtype>
Tensor<Dtype>* GlobalWorkSpace<Dtype>::forward_fetch_output(std::string name, bool is_loss_op = false){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            if(is_loss_op == true)
                it->second.second++;
            return &(it->second.first);
        }
    }
    LOG(FATAL) << "no this onput in flow tensors, do you have init it?" << flow_tensor_list();
}

template <typename Dtype>
Tensor<Dtype>* GlobalWorkSpace<Dtype>::backward_fetch_input(std::string name, bool is_data_op = false){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            if(is_data_op == false)
                it->second.second--;
            return &(it->second.first);
        }
    }
    LOG(FATAL) << "no this input in flow tensors, do you have init it?" << flow_tensor_list();
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::try_release_flow_tensor(){
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->second.second == 0)
            it->second.first.clear();
    }
}

template <typename Dtype>
std::string GlobalWorkSpace<Dtype>::flow_tensor_list(){
    std::string list = "we have flow tensors: ";
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        list = list + it->first;
        list = list + " ";
    }
    return list;
}

// for params
// std::vector<std::pair<std::string, std::pair<std::string, std::map<std::string, Tensor<Dtype> > > > > _params;
template <typename Dtype>
void GlobalWorkSpace<Dtype>::init_param(std::string op_name, std::string param_name, std::vector<int> shape){
    bool has_param = false;
    for(int i = 0; i < _params.size(); i++){
        if(_params[i].first == op_name){
            for(auto it = _params[i].second.second.begin(); it != _params[i].second.second.end(); it++){
                if(it->first == param_name)
                   has_param = true;
            }
            if(has_param == false){
                Tensor<Dtype> t(shape);
                _params[i].second.second[param_name] = t;
            }
            return;
        }
    }
}
template <typename Dtype>
Tensor<Dtype> *GlobalWorkSpace<Dtype>::fecth_param(std::string op_name, std::string param_name){
    for(int i = 0; i < _params.size(); i++){
        if(_params[i].first == op_name){
            for(auto it = _params[i].second.second.begin(); it != _params[i].second.second.end(); it++){
                if(it->first == param_name)
                   return &(it->second);
            }
            LOG(FATAL) << "no this param!!";
        }
    }
    LOG(FATAL) << "no this param!!";
}


template <typename Dtype>
void GlobalWorkSpace<Dtype>::global_init(){   
    _ops.clear();
    for(int i = 0; i < _graph->graph_sym()->op_size(); i++){
        GlobalWorkSpace<Dtype> *gws = this;
        OperatorDef opdef = _graph->graph_sym()->op(i);
        boost::shared_ptr<Operator<Dtype> > op = OperatorRegistry<Dtype>::CreateOperator(opdef,gws);
        op->init();
        _ops.push_back(op);
    }
}


template <typename Dtype>
void GlobalWorkSpace<Dtype>::forward(){
    for(int i = 0; i < _ops.size(); i++){
        _ops[i]->forward();
    }
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::backward(){
    for(int i = 0; i < _ops.size(); i++){
        _ops[i]->backward();
    }
}
template <typename Dtype>
void GlobalWorkSpace<Dtype>::train(){
    forward();
    backward();
}


INSTANTIATE_CLASS(GlobalWorkSpace);
}
