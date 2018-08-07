//
// Created by unsky on 02/07/18.
//
#include "sita/workspace.h"
namespace  sita {

void WorkSpace::device_query(){
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaError = cudaGetDeviceProperties(&deviceProp, i);
        LOG(INFO) << "device " << i  << " properties ";
        LOG(INFO) << "device name: " << deviceProp.name;
        LOG(INFO) << "Total memory: " << deviceProp.totalGlobalMem / 1024 / 1024;
        LOG(INFO) << "The max Threads: " << deviceProp.maxThreadsPerBlock;
        LOG(INFO) << "The Compute Capability version: " << deviceProp.major << "." << deviceProp.minor;
        LOG(INFO) << "The number of multi-processor in this device: " << deviceProp.multiProcessorCount;
    }
}
void WorkSpace::set_device(int gpu_id){
    _gpu_id = gpu_id;
    CUDA_CHECK(cudaSetDevice(_gpu_id));
    LOG(INFO) << "using GPU:" << _gpu_id;
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
//std::map<std::string, Tensor<Dtype> > _flow_tensor;
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
        _flow_tensor[name] = t;
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
        _flow_tensor[name] = t;
    }
}

template <typename Dtype>
Tensor<Dtype>* GlobalWorkSpace<Dtype>::fetch_input(std::string name){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            return &(it->second);
        }
    }
    LOG(FATAL) << "no this input in flow tensors, do you have init it?" << flow_tensor_list();
}

template <typename Dtype>
Tensor<Dtype>* GlobalWorkSpace<Dtype>::fetch_output(std::string name){
    bool has_flow_tensor = false;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it ++){
        if(it->first == name){
            return &(it->second);
        }
    }
    LOG(FATAL) << "no this onput in flow tensors, do you have init it?" << flow_tensor_list();
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

template <typename Dtype>
void GlobalWorkSpace<Dtype>::init_param(std::string op_name, std::string op_type, std::string param_name, std::vector<int> shape, Filler filler){
    bool has_param = false;
    bool has_op_name = false;
    for(auto i = _params.begin(); i != _params.end(); i++){
        if(i->first == op_name){
            has_op_name = true;
            for(auto it = i->second.params.begin(); it != i->second.params.end(); it++){
                if(it->first == param_name)
                   has_param = true;
            }
        }
    }
    if(has_op_name == false){
        OperatorParam<Dtype> p;
        p.type = op_type;
        Tensor<Dtype> t(shape);
        p.params[param_name] = t;
        p.fillers[param_name] = filler;
        _params[op_name] = p;

    }else if(has_op_name && has_param == false){
        Tensor<Dtype> t(shape);
        _params[op_name].params[param_name] = t;
        _params[op_name].fillers[param_name] = filler;

    }
}

template <typename Dtype>
Tensor<Dtype> *GlobalWorkSpace<Dtype>::fetch_param(std::string op_name, std::string param_name){
    for(auto i = _params.begin(); i != _params.end(); i++){
        if(i->first == op_name){
            for(auto it = i->second.params.begin(); it != i->second.params.end(); it++){
                if(it->first == param_name)
                   return &(it->second);
            }
            LOG(FATAL) << "no this param!!" << param_list();
        }
    }
    LOG(FATAL) << "no this param!!" << param_list();
}

template <typename Dtype>
std::string GlobalWorkSpace<Dtype>::param_list(){
    std::string str = "we have params:";
    for(auto i = _params.begin(); i != _params.end(); i++){
        str =  str + "\n" +  i->first + ": \n";
        for(auto it = i->second.params.begin(); it != i->second.params.end(); it++){
                str = str + it->first + " ";
        }
    }
    return str;
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::global_init(){   
    _ops.clear();
    for(int i = 0; i < _graph->graph_sym()->operatordef_size(); i++){
        GlobalWorkSpace<Dtype> *gws = this;
        OperatorParameter opdef = _graph->graph_sym()->operatordef(i);
        boost::shared_ptr<Operator<Dtype> > op = OperatorRegistry<Dtype>::CreateOperator(opdef, gws);
        op->setup();
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
