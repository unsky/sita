#include "sita/operators/batch_norm.h"
namespace sita{
  
template <typename Dtype>
BatchNorm<Dtype>::~BatchNorm() {
    if (!_handles_setup) return;
    cudnnDestroyTensorDescriptor(_input_desc);
    cudnnDestroyTensorDescriptor(_output_desc);
    cudnnDestroyTensorDescriptor(_scale_bias_mean_var_desc);
}

template<typename Dtype>
void BatchNorm<Dtype>::init(){
  	CHECK_EQ(this->_inputs.size(), this->_outputs.size()) << "input size should equal to output size";
  	LOG(INFO) << "Inputs:";
  	//Initialize param input and outputs
  	for(int i = 0; i < this->_inputs.size(); i++){
  	    Tensor<Dtype> *input = this->fetch_input(this->_inputs[i]);
  	    this->_input_shapes[this->_inputs[i]] = input->shape();
  	    CHECK_GT(input->count(), 0) << "check your graph, cannot infer " << this->_inputs[i] <<  " shape,in " << this->operator_name()<<"!!";
  	    LOG(INFO) << this->_inputs[i]<<": "<< this->fetch_input(this->_inputs[i])->shape_string();
  	}
  	Context::create_tensor4d_descriptor<Dtype>(&_input_desc);
  	Context::create_tensor4d_descriptor<Dtype>(&_output_desc);
  	Context::create_tensor4d_descriptor<Dtype>(&_scale_bias_mean_var_desc);
  	// currently only SPATIAL mode is supported (most commonly used mode)
  	// If there's enough demand we can implement CUDNN_BATCHNORM_PER_ACTIVATION
  	// though it's not currently implemented for the CPU layer
  	_mode = CUDNN_BATCHNORM_SPATIAL;
  	int channels = this->_input_shapes[this->_inputs[0]][1];
  	_save_mean.reshape(1, channels, 1, 1);
    _save_inv_var.reshape(1, channels, 1, 1);

  	CHECK_EQ(this->_param_configs.size(), 4) << "should specify param configs for scale, bias， mean, var";
  	LOG(INFO) << "Param:";
  	std::vector<int > scale_shape;
    scale_shape.push_back(1);
    scale_shape.push_back(channels);
    scale_shape.push_back(1);
    scale_shape.push_back(1);
    this->init_param("bacth_norm_scale", scale_shape, this->_param_configs[0]);
    Tensor<Dtype> * scale = this->fetch_param("bacth_norm_scale");
    LOG(INFO) << "batch norm scale: " << scale->shape_string();

  	std::vector<int > bias_shape;
    bias_shape.push_back(1);
    bias_shape.push_back(channels);
    bias_shape.push_back(1);
    bias_shape.push_back(1);
    this->init_param("bacth_norm_bias", bias_shape, this->_param_configs[1]);
    Tensor<Dtype> * bias = this->fetch_param("bacth_norm_bias");
    LOG(INFO) <<"batch norm bias: "<< bias->shape_string();

  	std::vector<int > mean_shape;
    mean_shape.push_back(1);
    mean_shape.push_back(channels);
    mean_shape.push_back(1);
    mean_shape.push_back(1);
    this->init_param("bacth_norm_mean", bias_shape, this->_param_configs[2]);
    Tensor<Dtype> * mean = this->fetch_param("bacth_norm_mean");
    LOG(INFO) <<"batch norm mean: " << scale->shape_string();

    std::vector<int > var_shape;
    var_shape.push_back(1);
    var_shape.push_back(channels);
    var_shape.push_back(1);
    var_shape.push_back(1);
    this->init_param("bacth_norm_var", var_shape, this->_param_configs[3]);
    Tensor<Dtype> * var = this->fetch_param("bacth_norm_var");
    LOG(INFO) <<"batch norm var: " << var->shape_string();

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
void BatchNorm<Dtype>::infer_shape() {
    // set up main tensors
    Context::set_tensor4d_descriptor<Dtype>(&_input_desc, this->_input_shapes[this->_inputs[0]][0],
      this->_input_shapes[this->_inputs[0]][1], this->_input_shapes[this->_inputs[0]][2], this->_input_shapes[this->_inputs[0]][3]);
    Context::set_tensor4d_descriptor<Dtype>(&_output_desc, this->_output_shapes[this->_outputs[0]][0],
      this->_output_shapes[this->_outputs[0]][1], this->_output_shapes[this->_outputs[0]][2], this->_output_shapes[this->_outputs[0]][3]);

    if (_mode != CUDNN_BATCHNORM_SPATIAL && _mode != CUDNN_BATCHNORM_PER_ACTIVATION) {
      LOG(FATAL) << "Unknown cudnnBatchNormMode_t";
    }
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(_scale_bias_mean_var_desc,
        _input_desc, _mode));
}

INSTANTIATE_CLASS(BatchNorm);
REGISTER_OPERATOR_CLASS(BatchNorm);

}