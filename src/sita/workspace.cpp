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
TempTensor<Dtype>  GlobalWorkSpace<Dtype>::new_tensor() {
    TempTensor<Dtype> t;
    if (_temp_tensor_num == 0) {
        Tensor<Dtype> temp_tensor;
        _temp_tensor[0] = temp_tensor;
        _temp_tensor_control[0] = std::make_pair(&(_temp_tensor[0]), true);
        _temp_tensor_num++;
        t.key = 0;
        t.tensor = &(_temp_tensor[0]);
        return t;
    } else {
        int released_id = -1;
        for (int i = 0; i < _temp_tensor_num; i++) {
            if (_temp_tensor_control[i].second == false) {
                released_id = i;
            }
        }
        if (released_id == -1) {
            Tensor<Dtype> temp_tensor;
            _temp_tensor[_temp_tensor_num] = temp_tensor;
            _temp_tensor_control[_temp_tensor_num]= std::make_pair(&(_temp_tensor[_temp_tensor_num]), true);
            t.key = 0;
            t.tensor = &(_temp_tensor[_temp_tensor_num]);
            _temp_tensor_num++;
            return t;
        } else {
            _temp_tensor_control[released_id].second = true;
            t.key = released_id;
            t.tensor =  &_temp_tensor[released_id];
            return t;
        }
    }
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::free_tensor(TempTensor<Dtype> t) {
    _temp_tensor_control[t.key].second = false;
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::temp_memory(){
    float memory_size = 0;
    for(int i = 0; i < _temp_tensor_num; i++){
        memory_size += (_temp_tensor[i].count() * sizeof(Dtype));
    }
    LOG(INFO) << "the fact of temp tensor being used: "<<"[number: " << _temp_tensor_num << " " << "memory size: " <<
              std::to_string(memory_size * 2/(1024 * 8))<<" KB].";

    return;
}

// flow tensor
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
    LOG(FATAL) << "no this output in flow tensors, do you have init it?" << flow_tensor_list();
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
void GlobalWorkSpace<Dtype>::flow_memory(){
    float memory_size = 0;
    for(auto it = _flow_tensor.begin(); it != _flow_tensor.end(); it++){
        memory_size += (it->second.count() * sizeof(Dtype));
    }
    LOG(INFO) << "the fact of flow tensor being used: "<<"[number: " << _flow_tensor.size() << " " << "memory size: " <<
              std::to_string(memory_size*2/(1024 * 8))<<" KB].";

    return;
}

template <typename Dtype>
void GlobalWorkSpace<Dtype>::init_param(std::string op_name, std::string op_type, std::string param_name,
    std::vector<int> shape,  ParamConfig p_config, bool is_shared){
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
    if(is_shared){
        if (has_param)
            return;
        if(has_op_name == false){
            OperatorParam <Dtype> p;
            p.type = op_type;
            Tensor <Dtype> t(shape);
            p.params[param_name] = t;
            p.param_configs[param_name] = p_config;
            p.is_inited[param_name] = false;
            _params[op_name] = p;
        }else if(has_op_name && has_param == false){
            Tensor <Dtype> t(shape);
            _params[op_name].params[param_name] = t;
            _params[op_name].param_configs[param_name] = p_config;
            _params[op_name].is_inited[param_name] = false;
        }
    }else {
        if (has_op_name == false){
            OperatorParam <Dtype> p;
            p.type = op_type;
            Tensor <Dtype> t(shape);
            p.params[param_name] = t;
            p.param_configs[param_name] = p_config;
            p.is_inited[param_name] = false;
            _params[op_name] = p;
        }else if (has_op_name && has_param == false){
            Tensor <Dtype> t(shape);
            _params[op_name].params[param_name] = t;
            _params[op_name].param_configs[param_name] = p_config;
            _params[op_name].is_inited[param_name] = false;
        }
    }
}

template <typename Dtype>
Tensor<Dtype> *GlobalWorkSpace<Dtype>::fetch_param(std::string op_name, std::string param_name, bool is_shared){
    if(is_shared) {
        for(auto i = _params.begin(); i != _params.end(); i++){
            for(auto it = i->second.params.begin(); it != i->second.params.end(); it++){
                if(it->first == param_name)
                    return &(it->second);
            }
        }
        LOG(FATAL) << "no this param!!" << param_list();
    }else{
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
void GlobalWorkSpace<Dtype>::param_memory(){
    float memory_size = 0;
    int num = 0;
    for(auto it = _params.begin(); it != _params.end(); it++){
        for(auto i = it->second.params.begin(); i != it->second.params.end(); i++){
            memory_size += (i->second.count() * sizeof(Dtype));
            num++;
        }
    }
    LOG(INFO) << "the fact of param tensor being used: "<<"[number: " << num << " " << "memory size: " <<
              std::to_string(memory_size*2/(1024 * 8))<<" KB].";
    return;
}


template <typename Dtype>
void GlobalWorkSpace<Dtype>::global_init(Graph * graph, DataProvider<Dtype> * data_provider){

    _data_provider = data_provider;
    Batch<Dtype> * batch= data_provider->fetch_batch();
    //init the output of dataprovider
    for(int i = 0; i < batch->product_size(); i ++){
        init_output(batch->product_name(i));
    }
    _graph = graph;
    graph_show();
    _ops.clear();
    //init param, input and output of ops
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
    Batch<Dtype> *batch = _data_provider->fetch_batch();

    //data and label
    for(int i = 0; i < batch->product_size();i ++){
        fetch_output(batch->product_name(i))->copy_from(batch->product(i), true);
    }
    forward();
    backward();
}


INSTANTIATE_CLASS(GlobalWorkSpace);
}
