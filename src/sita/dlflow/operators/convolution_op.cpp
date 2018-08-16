//
// Created by unsky on 16/08/18.
//
#include "sita/dlflow/operators/convolution_op.h"
namespace sita{

template<typename Dtype>
void ConvolutionOp<Dtype>::init(){
    // params
    std::vector<int> shape;
    shape.push_back(5);
    shape.push_back(6);
    shape.push_back(7);
    shape.push_back(8);
    this->init_param("convolution_weight", shape, this->_param_configs[0]);
    this->init_param("convolution_bias", shape, this->_param_configs[1]);
}



INSTANTIATE_CLASS(ConvolutionOp);
REGISTER_OPERATOR_CLASS(ConvolutionOp);
}//namespace
