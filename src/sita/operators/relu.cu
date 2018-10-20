#include "sita/operators/relu.h"
namespace sita{
template<typename Dtype>
void ReLU<Dtype>::forward(){
    Tensor<Dtype>* input_tensor = this->fetch_input(this->_inputs[0]);
    const Dtype * input_data = input_tensor->gpu_data();
    Tensor<Dtype>* output_tensor = this->fetch_output(this->_outputs[0]);
    Dtype * output_data = output_tensor->mutable_gpu_data();

#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnActivationForward(_handle[0],
        _activ_desc,
        CudnnDataType<Dtype>::one,
        this->_input_desc, input_data,
        CudnnDataType<Dtype>::zero,
        this->_output_desc, output_data));
#else 
    CUDNN_CHECK(cudnnActivationForward_v4(_handle[0]
        _activ_desc,
        CudnnDataType<Dtype>::one,
        this->_input_desc, input_data,
        CudnnDataType<Dtype>::zero,
        this->_output_desc_, _output_data));
#endif
}
template<typename Dtype>
void ReLU<Dtype>::backward(){
    Tensor<Dtype>* input_tensor = this->fetch_input(this->_inputs[0]);
    const Dtype * input_data = input_tensor->gpu_data();
    Dtype * input_diff = input_tensor->mutable_gpu_diff();
    Tensor<Dtype>* output_tensor = this->fetch_output(this->_outputs[0]);
    const Dtype * output_data = output_tensor->gpu_data();
    const Dtype * output_diff = output_tensor->gpu_diff();

#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnActivationBackward(_handle[1],
        _activ_desc,
        CudnnDataType<Dtype>::one,
        _output_desc, output_data, _output_desc, output_diff,
        _input_desc, input_data,
        CudnnDataType<Dtype>::zero,
        _input_desc, input_diff));
#else
    CUDNN_CHECK(cudnnActivationBackward_v4(_handle[1],
        _activ_desc,
        CudnnDataType<Dtype>::one,
        _output_desc, output_data, _output_desc, output_diff,
        _input_desc, input_data,
        CudnnDataType<Dtype>::zero,
        _input_desc, input_diff));
#endif
}

INSTANTIATE_OPERATOR_GPU_FUNCS(ReLU);
}//namespace