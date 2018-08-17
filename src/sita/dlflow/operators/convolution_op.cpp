//
// Created by unsky on 16/08/18.
//
#include "sita/dlflow/operators/convolution_op.h"
namespace sita{

template<typename Dtype>
void ConvolutionOp<Dtype>::init(){
    // params
    std::vector<int> shape;
//    shape.push_back();
//    shape.push_back(0);
//    shape.push_back(0);
//    shape.push_back(0);
//    this->init_param("convolution_weight", shape, this->_param_configs[0]);
//    this->init_param("convolution_bias", shape, this->_param_configs[1]);
}

template<typename Dtype>
void ConvolutionOp<Dtype>::forward(){
//    Tensor<Dtype> * data = this->fetch_input(this->_inputs[0]);
//    Tensor<Dtype> * add_weight = this->fetch_param("add_weight");
    //LOG(INFO)<<_add_op_param.kernel_h();
};


INSTANTIATE_CLASS(ConvolutionOp);
REGISTER_OPERATOR_CLASS(ConvolutionOp);
}//namespace
