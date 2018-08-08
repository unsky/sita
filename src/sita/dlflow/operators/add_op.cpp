//---------------------------------
//write by unsky
//---------------------------------
#include "sita/dlflow/operators/add_op.h"
namespace sita{

template<typename Dtype>
void AddOp<Dtype>::init(){

    // params
    std::vector<int> shape;
    shape.push_back(5);
    shape.push_back(6);
    shape.push_back(7);
    shape.push_back(8);
    this->init_param("add_weight", shape);
    this->init_param("add_bias", shape);
}

template<typename Dtype>
void AddOp<Dtype>::forward(){
      Tensor<Dtype> * data = this->fetch_input(this->_inputs[0]);
      Tensor<Dtype> * add_weight = this->fetch_param("add_weight");
     //LOG(INFO)<<_add_op_param.kernel_h();


};
template<typename Dtype>
void AddOp<Dtype>::backward(){
}
INSTANTIATE_CLASS(AddOp);
REGISTER_OPERATOR_CLASS(AddOp);
}//namespace
