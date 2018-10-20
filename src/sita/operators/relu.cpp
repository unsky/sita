#include "sita/operators/relu.h"
namespace sita{

template <typename Dtype>
ReLU<Dtype>::~ReLU() {
  if (!_handles_setup) return;
  cudnnDestroyTensorDescriptor(_input_desc);
  cudnnDestroyTensorDescriptor(_output_desc);
  cudnnDestroyActivationDescriptor(_activ_desc);
}
template<typename Dtype>
void ReLU<Dtype>::init(){
    LOG(INFO) << "Inputs:";
    for(int i = 0; i < this->_inputs.size(); i++){
        Tensor<Dtype> *input = this->fetch_input(this->_inputs[i]);
        this->_input_shapes[this->_inputs[i]] = input->shape();
        CHECK_GT(input->count(), 0) << "check your graph, cannot infer " << this->_inputs[i] <<  " shape,in " << this->operator_name()<<"!!";
        LOG(INFO) << this->_inputs[i]<<": "<< this->fetch_input(this->_inputs[i])->shape_string();
    }

      // initialize cuDNN
    Context::create_tensor4d_descriptor<Dtype>(&_input_desc);
    Context::create_tensor4d_descriptor<Dtype>(&_output_desc);
    _handles_setup = true;
    cudnnCreateActivationDescriptor(&_activ_desc);
    cudnnSetActivationDescriptor(_activ_desc, CUDNN_ACTIVATION_RELU,
                               CUDNN_PROPAGATE_NAN, 0.0);
    LOG(INFO) << "Outputs:";
    for(int i = 0; i < this->_outputs.size(); i++){
        Tensor<Dtype> *output = this->fetch_output(this->_outputs[i]);
        this->_output_shapes[this->_outputs[i]] = this->_input_shapes[this->_inputs[i]];
        output->reshape(this->fetch_input(this->_inputs[i])->shape());
        LOG(INFO) << this->_outputs[i]<<": "<< this->fetch_output(this->_outputs[i])->shape_string();
    }
    _stream = new cudaStream_t[2];
    _handle = new cudnnHandle_t[2];
    for (int g = 0; g < 2; g++) {
        CUDA_CHECK(cudaStreamCreate(&_stream[g]));
        CUDNN_CHECK(cudnnCreate(&_handle[g]));
        CUDNN_CHECK(cudnnSetStream(_handle[g], _stream[g]));
    }
    _handles_setup = true;
    infer_shape();
}
template<typename Dtype>
void ReLU<Dtype>::infer_shape(){
    const int N = this->_output_shapes[this->_outputs[0]][0];
    const int K = this->_output_shapes[this->_outputs[0]][1];
    const int H = this->_output_shapes[this->_outputs[0]][2];;
    const int W = this->_output_shapes[this->_outputs[0]][3];;
    Context::set_tensor4d_descriptor<Dtype>(&_input_desc, N, K, H, W);
    Context::set_tensor4d_descriptor<Dtype>(&_output_desc, N, K, H, W);  

}
INSTANTIATE_CLASS(ReLU);
REGISTER_OPERATOR_CLASS(ReLU);
}