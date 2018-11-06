#include "sita/operators/pooling.h"

namespace sita{
template <typename Dtype>
Pooling<Dtype>::~Pooling() {
    if (!_handles_setup) return;
    cudnnDestroyTensorDescriptor(_input_desc);
    cudnnDestroyTensorDescriptor(_output_desc);
    cudnnDestroyPoolingDescriptor(_pooling_desc);
}

template<typename Dtype>
void Pooling<Dtype>::init(){
    Context::create_tensor4d_descriptor<Dtype>(&_input_desc);
    Context::create_tensor4d_descriptor<Dtype>(&_output_desc);

    LOG(INFO) << "Params:";
    int kernel_h;
    int kernel_w;
    if(this->_op_param.has_kernel_h() && this->_op_param.has_kernel_w()){
        kernel_h = this->_op_param.kernel_h();
        kernel_w = this->_op_param.kernel_w();
    }else{
        kernel_h = this->_op_param.kernel_size();
        kernel_w = this->_op_param.kernel_size();
    }
    
    Context::create_tensor4d_descriptor<Dtype>(&_pooling_desc,
      this->layer_param_.pooling_param().pool(), &_mode,
      this->_kernel_h, this->_kernel_w, this->_pad_h, this->_pad_w,
      this->_stride_h, this->_stride_w);
    handles_setup_ = true;
}

template<typename Dtype>
void Pooling<Dtype>::infer_shape(){

}
INSTANTIATE_CLASS(Pooling);
REGISTER_OPERATOR_CLASS(Pooling);
}