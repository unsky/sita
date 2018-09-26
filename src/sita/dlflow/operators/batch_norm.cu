#include "sita/dlflow/operators/batch_norm.h"
namespace sita{
template<typename Dtype>
void BatchNorm<Dtype>::forward(){
  Tensor<Dtype> *input_tensor = this->fetch_input(this->_inputs[0]);
  Tensor<Dtype> *output_tensor = this->fetch_output(this->_outputs[0]);
  Tensor<Dtype> *scale_tensor = this->fetch_param("bacth_norm_scale");
  Tensor<Dtype> *bias_tensor = this->fetch_param("bacth_norm_bias");
  Tensor<Dtype> *mean_tensor = this->fetch_param("bacth_norm_mean");
  Tensor<Dtype> *var_tensor = this->fetch_param("bacth_norm_var");

  const Dtype* input_data = input_tensor->gpu_data();
  const Dtype* scale_data = scale_tensor->gpu_data();
  const Dtype* bias_data = bias_tensor->gpu_data();
  Dtype* output_data = output_tensor->mutable_gpu_data();
  Dtype* save_mean = _save_mean.mutable_gpu_data();
  Dtype* save_inv_var = _save_inv_var.mutable_gpu_data();

  double epsilon = max(this->_op_param.eps(), CUDNN_BN_MIN_EPSILON);

  if (this->_phase == "train") {
    // Call Batch normalization forward
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      _handle[0], _mode,
      CudnnDataType<Dtype>::one, CudnnDataType<Dtype>::zero,
      _input_desc, input_data,
      _input_desc, output_data,
      _scale_bias_mean_var_desc, scale_data, bias_data,
      1 - this->_op_param.moving_average_fraction(),
      mean_tensor->mutable_gpu_data(),  // mean
      var_tensor->mutable_gpu_data(),  // variance
      epsilon, save_mean, save_inv_var));
  } else if (this->_phase == "test") {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      _handle[1], _mode,
      CudnnDataType<Dtype>::one, CudnnDataType<Dtype>::zero,
      _input_desc, input_data,
      _input_desc, output_data,
      _scale_bias_mean_var_desc, scale_data, bias_data,
      mean_tensor->gpu_data(),  // mean
      mean_tensor->gpu_data(),  // variance
      epsilon));
  } else {
    LOG(FATAL) << "Unknown phase";
  }

}
template<typename Dtype>
void BatchNorm<Dtype>::backward(){
  Tensor<Dtype> *input_tensor = this->fetch_input(this->_inputs[0]);
  Tensor<Dtype> *output_tensor = this->fetch_output(this->_outputs[0]);
  const Dtype* output_data = output_tensor->gpu_data();
  const Dtype* output_diff = output_tensor->gpu_diff();
  const Dtype* input_data = input_tensor->gpu_data();
  Tensor<Dtype> *scale_tensor = this->fetch_param("bacth_norm_scale");
  Tensor<Dtype> *bias_tensor = this->fetch_param("bacth_norm_bias");
  Tensor<Dtype> *mean_tensor = this->fetch_param("bacth_norm_mean");
  Tensor<Dtype> *var_tensor = this->fetch_param("bacth_norm_var");

  const Dtype* save_mean = _save_mean.gpu_data();
  const Dtype* save_inv_var = _save_inv_var.gpu_data();

  Dtype* input_diff = input_tensor->mutable_gpu_diff();
  const Dtype* scale_data = scale_tensor->gpu_data();
  Dtype* scale_diff = scale_tensor->mutable_gpu_diff();
  Dtype* bias_diff = bias_tensor->mutable_gpu_diff();

  double epsilon = max(this->_op_param.eps(), CUDNN_BN_MIN_EPSILON);

  // call Batch Normalization Backward
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      _handle[1], _mode,
      CudnnDataType<Dtype>::one, CudnnDataType<Dtype>::zero,
#if CUDNN_VERSION >= 4005
      CudnnDataType<Dtype>::one, CudnnDataType<Dtype>::one,
#endif
      _input_desc, input_data,
      _input_desc, output_diff,
      _input_desc, input_diff,
      _scale_bias_mean_var_desc,
      scale_data, scale_diff, bias_diff,
      epsilon, save_mean, save_inv_var));
}

INSTANTIATE_OPERATOR_GPU_FUNCS(BatchNorm);
}