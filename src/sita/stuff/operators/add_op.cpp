//---------------------------------
//write by unsky
//---------------------------------
#include "sita/stuff/operators/add_op.h"
namespace sita{

template<typename Dtype>
void AddOp<Dtype>::init(){

    //inputs and outputs
    _inputs.clear();
    for(int i = 0; i < this->_opdef.inputs.size(); i++){
        this->_gws->init_input(this->_opdef.inputs[i]);
        _inputs.push_back(this->_opdef.inputs[i]);
    }
    _outputs.clear();
    for(int i = 0; i < this->_opdef.outputs.size(); i++){
        this->_gws->init_output(this->_opdef.outputs[i]);
        _outputs.push_back(this->_opdef.outputs[i]);
    }
    LOG(INFO) << "---"<<_add_op_param.stride_w;
    // params
    if(true){
        std::vector<int> shape;
        shape.push_back(5);
        shape.push_back(6);
        shape.push_back(7);
        shape.push_back(8);

// todo:Filler
        this->_gws->init_param(this->_opdef.name, "add_weight", shape);
        this->_gws->init_param(this->_opdef.name,"add_bias", shape);
    }


}
template<typename Dtype>
void AddOp<Dtype>::forward(){
 //   Tensor<Dtype> * data = this->_gws->forward_fetch_input(this->_opdef.inputs());
//    Tensor<Dtype> * input2 = this->_gws->forward_fetch_input();
//    Tensor<Dtype> * weight = this->_gws->fetch_param(this->_opdef.name(), "add_weight");
//    Tensor<Dtype> * bias = this->_gws->fetch_param(this->_opdef.name(), "add_bias");
//    Tensor<Dtype> * output1 = this->_gws->forward_fetch_output();

};
template<typename Dtype>
void AddOp<Dtype>::backward(){
LOG(INFO)<<"NUM: ";
}
INSTANTIATE_CLASS(AddOp);
REGISTER_OPERATOR_CLASS(AddOp);
}//namespace
