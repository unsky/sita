#include "sita/operators/convolution.h"
namespace sita{

__global__ void sync_conv_groups() { }

template<typename Dtype>
void Convolution<Dtype>::forward(){
  Tensor<Dtype>* weight_tensor = this->fetch_param("convolution_weight");
  const Dtype * weight = weight_tensor->gpu_data();
  for (int i = 0; i < this->_inputs.size(); ++i) {
    Tensor<Dtype>* input_tensor = this->fetch_input(this->_inputs[i]);
    const Dtype * input_data = input_tensor->gpu_data();
    Tensor<Dtype>* output_tensor = this->fetch_output(this->_outputs[i]);
    Dtype * output_data = output_tensor->mutable_gpu_data();
    int input_offset =  input_tensor->count() / this->_op_param.group();
    int output_offset =  output_tensor->count() / this->_op_param.group();
    int weight_offset = weight_tensor->count() / this->_op_param.group();


    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->_op_param.group(); g++) {
      //Filters.
      CUDNN_CHECK(cudnnConvolutionForward(_handle[g],
            CudnnDataType<Dtype>::one,
            _input_descs[i], input_data + input_offset * g,
            _filter_desc, weight + weight_offset * g,
            _conv_descs[i],
            _fwd_algo[i], workspace[g], _workspace_fwd_sizes[i],
            CudnnDataType<Dtype>::zero,
            _output_descs[i], output_data + output_offset * g));

      // Bias.
      if (this->_op_param.bias_term()) {

        Tensor<Dtype> * bias_tensor = this->fetch_param("convolution_bias");
        const Dtype * bias_data = bias_tensor->gpu_data();
        int bias_offset = bias_tensor->count() / this->_op_param.group();
        CUDNN_CHECK(cudnnAddTensor(_handle[g],
              CudnnDataType<Dtype>::one,
              _bias_desc, bias_data + bias_offset * g,
              CudnnDataType<Dtype>::one,
              _output_descs[i], output_data + output_offset * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    sync_conv_groups<<<1, 1>>>();
  }
};

template<typename Dtype>
void Convolution<Dtype>::backward(){
  Tensor<Dtype>* weight_tensor = this->fetch_param("convolution_weight");

  for (int i = 0; i < this->_inputs.size(); ++i) {
    Tensor<Dtype>* input_tensor = this->fetch_input(this->_inputs[i]);
    const Dtype * input_data = input_tensor->gpu_data();
    Dtype * input_diff = input_tensor->mutable_gpu_data();
    if(!this->_gradient_block){
      Tensor<Dtype>* output_tensor = this->fetch_output(this->_outputs[i]);
      const Dtype * output_data = output_tensor->gpu_data();
      Dtype * output_diff = output_tensor->mutable_gpu_diff();
      const Dtype * weight = weight_tensor->gpu_data();
      Dtype * weight_diff = weight_tensor->mutable_gpu_diff();
      Dtype * bias_diff = NULL;
      int input_offset =  input_tensor->count() / this->_op_param.group();
      int output_offset =  output_tensor->count() / this->_op_param.group();
      int weight_offset = weight_tensor->count() / this->_op_param.group();

      for (int g = 0; g < this->_op_param.group(); g++) {
        // Gradient w.r.t. bias.
        if (this->_op_param.bias_term()) {
          Tensor<Dtype>* bias_tensor = this->fetch_param("convolution_bias");
          bias_diff  = bias_tensor->mutable_gpu_diff();
          int bias_offset = bias_tensor->count() / this->_op_param.group();  
          CUDNN_CHECK(cudnnConvolutionBackwardBias(_handle[0 * this->_op_param.group() + g],
              CudnnDataType<Dtype>::one,
              _output_descs[i], 
              output_diff + output_offset * g,
              CudnnDataType<Dtype>::one,
              _bias_desc, bias_diff + bias_offset * g));
        }

        // Gradient w.r.t. weights.
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              _handle[1*this->_op_param.group() + g],
              CudnnDataType<Dtype>::one,
              _input_descs[i], input_data + input_offset * g,
              _output_descs[i], output_diff + output_offset * g,
              _conv_descs[i],
              _bwd_filter_algo[i], workspace[1*this->_op_param.group() + g],
              _workspace_bwd_filter_sizes[i],
              CudnnDataType<Dtype>::one,
              _filter_desc, weight_diff + weight_offset * g));


          // Gradient w.r.t. bottom data.
          CUDNN_CHECK(cudnnConvolutionBackwardData(
                _handle[2*this->_op_param.group() + g],
                CudnnDataType<Dtype>::one,
                _filter_desc, weight + weight_offset * g,
                _output_descs[i], output_diff + output_offset * g,
                _conv_descs[i],
                _bwd_data_algo[i], workspace[2 * this->_op_param.group() + g],
                _workspace_bwd_data_sizes[i],
                CudnnDataType<Dtype>::zero,
                _input_descs[i], input_diff + input_offset * g));
      }
    }
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    sync_conv_groups<<<1, 1>>>();
  }
};

INSTANTIATE_OPERATOR_GPU_FUNCS(Convolution);
}//namespace
