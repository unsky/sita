//---------------------------------
//write by unsky
//---------------------------------
#include "sita/stuff/operators/add_op.h"
namespace sita{

template<typename Dtype>
void AddOp<Dtype>::init(){
    for(int i = 0; i < this->_opdef.inputs_size(); i++){
        this->_gws->init_input(this->_opdef.inputs(i));
    }
    for(int i = 0; i < this->_opdef.outputs_size(); i++){
        this->_gws->init_output(this->_opdef.outputs(i));
    }
    std::vector<int> shape;
    shape.push_back(5);
    shape.push_back(6);
    shape.push_back(7);
    shape.push_back(8);

    this->_gws->init_param(this->_opdef.name(), "add_weight", shape);
    this->_gws->init_param(this->_opdef.name(),"add_bias", shape);
}
template<typename Dtype>
void AddOp<Dtype>::forward(){
  /*  Tensor<Dtype> * input1 = this->_gws->forward_fetch_input(this->_opdef.inputs());
    Tensor<Dtype> * input2 = this->_gws->forward_fetch_input();
    Tensor<Dtype> * weight = this->_gws->fetch_param(this->_opdef.name(), "add_weight");
    Tensor<Dtype> * bias = this->_gws->fetch_param(this->_opdef.name(), "add_bias");
    Tensor<Dtype> * output1 = this->_gws->forward_fetch_output();
*/
    LOG(INFO)<<"AAAAAAAAAAAAAA";
};
template<typename Dtype>
void AddOp<Dtype>::backward(){
LOG(INFO)<<"NUM: ";
}
INSTANTIATE_CLASS(AddOp);
REGISTER_OPERATOR_CLASS(AddOp);
}//namespace
