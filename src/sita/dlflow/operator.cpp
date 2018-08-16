//
// Created by unsky on 27/06/18.
//
#include "sita/dlflow/operator.h"
namespace sita{

template<typename Dtype>
void Operator<Dtype>::setup(){
    //inputs and outputs
    _inputs.clear();
    for(int i = 0; i < _opdef.input_size(); i++){
        _gws->init_input(_opdef.input(i));
        _inputs.push_back(_opdef.input(i));
    }
    _outputs.clear();
    for(int i = 0; i < _opdef.output_size(); i++){
        _gws->init_output(_opdef.output(i));
        _outputs.push_back(_opdef.output(i));
    }
    _params.clear();

    _param_configs.clear();
    for(int i = 0; i < _opdef.param_size(); i++) {
        _param_configs.push_back(_opdef.param(i));
    }
    _is_shared = false;
    _shared_param_pairs.clear();

}

template<typename Dtype>
void Operator<Dtype>::init_param(std::string param_name, std::vector<int> shape, ParamConfig p_config){
    std::string name;
    if(p_config.has_name()){
        name = p_config.name();
        _is_shared = true;
    }else{
        name = param_name;
    }
    _gws->init_param(_opdef.name(), _opdef.type(), name, shape, p_config, _is_shared);
    _params.push_back(param_name);
    if(_is_shared){
        _shared_param_pairs.push_back(std::make_pair(param_name, p_config.name()));
    }

}

template<typename Dtype>
Tensor<Dtype> * Operator<Dtype>::fetch_input(std::string name){
    bool has_input = false;
    for(int i = 0; i < _inputs.size(); i++)
        if(_inputs[i] == name)
            has_input = true;
    if(has_input) {
        return this->_gws->fetch_input(name);
    }else{
        LOG(FATAL) << "no " << name <<" in the inputs of " << _opdef.name();
    }

}

template<typename Dtype>
Tensor<Dtype> * Operator<Dtype>::fetch_output(std::string name){
    bool has_output = false;
    for(int i = 0; i < _outputs.size(); i++)
        if(_outputs[i] == name)
            has_output = true;
    if(has_output) {
        return this->_gws->fetch_output(name);
    }else{
        LOG(FATAL) << "no " << name <<" in the outputs of " << _opdef.name();
    }
}

template<typename Dtype>
Tensor<Dtype> * Operator<Dtype>::fetch_param(std::string name){
    bool has_param = false;
    if(_is_shared){
        for(auto it = _shared_param_pairs.begin(); it != _shared_param_pairs.end(); it++){
            if(it->first == name){
                has_param = true;
                name = it->second;
            }
        }
    }
    for(int i = 0; i < _params.size(); i++)
        if(_params[i] == name)
            has_param = true;

    if(has_param) {
        return this->_gws->fetch_param(_opdef.name(), name, _is_shared);
    }else{
        LOG(FATAL) << "no " << name <<" in the params of " << _opdef.name();
    }

}
INSTANTIATE_CLASS(Operator);

}//namespace;
