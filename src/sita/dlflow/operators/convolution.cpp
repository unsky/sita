//
// Created by unsky on 16/08/18.
//
#include "sita/dlflow/operators/convolution.h"
namespace sita{
template<typename Dtype>
Convolution<Dtype>::~Convolution(){
      // Check that handles have been setup before destroying.
  if (!_handles_setup) { return; }

  for (int i = 0; i < _input_descs.size(); i++) {
    cudnnDestroyTensorDescriptor(_input_descs[i]);
    cudnnDestroyTensorDescriptor(_output_descs[i]);
    cudnnDestroyConvolutionDescriptor(_conv_descs[i]);
  }
  if (this->_op_param.bias_term()) {
    cudnnDestroyTensorDescriptor(_bias_desc);
  }
  cudnnDestroyFilterDescriptor(_filter_desc);

  for (int g = 0; g < this->_op_param.group() * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(_stream[g]);
    cudnnDestroy(_handle[g]);
  }

  cudaFree(workspaceData);
  delete [] workspace;
  delete [] _stream;
  delete [] _handle;
  delete [] _fwd_algo;
  delete [] _bwd_filter_algo;
  delete [] _bwd_data_algo;
  delete [] _workspace_fwd_sizes;
  delete [] _workspace_bwd_data_sizes;
  delete [] _workspace_bwd_filter_sizes;

}

template<typename Dtype>
void Convolution<Dtype>::init(){


    CHECK_EQ(this->_inputs.size(), this->_outputs.size()) << "input size should equal to output size";
    LOG(INFO) << "Inputs:";
    //Initialize param input and outputs
    for(int i = 0; i < this->_inputs.size(); i++){
        Tensor<Dtype> *input = this->fetch_input(this->_inputs[i]);
        this->_input_shapes[this->_inputs[i]] = input->shape();
        CHECK_GT(input->count(), 0) << "check your graph, cannot infer " << this->_inputs[i] <<  " shape,in " << this->operator_name()<<"!!";
        LOG(INFO) << this->_inputs[i]<<": "<< this->fetch_input(this->_inputs[i])->shape_string();
    }

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
    std::vector<int > weight_shape;
    weight_shape.push_back(this->_op_param.num_output());
    weight_shape.push_back(this->_input_shapes[this->_inputs[0]][1]/this->_op_param.group());
    weight_shape.push_back(kernel_h);
    weight_shape.push_back(kernel_w);
    this->init_param("convolution_weight",weight_shape,this->_param_configs[0]);
    Tensor<Dtype> * weight = this->fetch_param("convolution_weight");
    LOG(INFO) << "convolution_weight:" <<weight->shape_string();
    this->_param_shapes["convolution_weight"] = weight_shape;
    if(this->_op_param.bias_term()) {
        std::vector<int> bias_shape;
        bias_shape.push_back(this->_op_param.num_output());
        this->init_param("convolution_bias", bias_shape,this->_param_configs[1]);

        Tensor<Dtype> * bias = this->fetch_param("convolution_bias");
        LOG(INFO) << "convolution_bias:" <<bias->shape_string();
        this->_param_shapes["convolution_bias"] = weight_shape;
    }
    LOG(INFO) << "Outputs:";
    for(int i = 0; i < this->_outputs.size(); i++){
      std::vector<int > output_shape;
      output_shape.push_back(this->_input_shapes[this->_inputs[i]][0]);
      output_shape.push_back(this->_op_param.num_output());

      int in_height = this->_input_shapes[this->_inputs[i]][2];
      int in_width = this->_input_shapes[this->_inputs[i]][3];

      int pad_h, pad_w, stride_h, stride_w;

      if(this->_op_param.has_pad_h() && this->_op_param.has_pad_w()){
          pad_h = int(this->_op_param.pad_h());
          pad_w = int(this->_op_param.pad_w());
      }else{
          pad_h = this->_op_param.pad();
          pad_w = this->_op_param.pad();
      }

      if(this->_op_param.has_stride_w() && this->_op_param.has_stride_h()){
          stride_h = int(this->_op_param.stride_h());
          stride_w = int(this->_op_param.stride_w());
      }else{
          stride_h = this->_op_param.stride();
          stride_w = this->_op_param.stride();
      }
      int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
      int out_width = (in_width + 2* pad_w - kernel_w) / stride_w + 1;
      output_shape.push_back(out_height);
      output_shape.push_back(out_width);

      Tensor<Dtype> *output_data = this->fetch_output(this->_outputs[i]);
      output_data->reshape(output_shape);
      this->_output_shapes[this->_outputs[i]] = output_shape;
      LOG(INFO) << this->_outputs[i]<<": "<< this->fetch_output(this->_outputs[i])->shape_string();
    }

    // Initialize CUDA streams and cuDNN.
    _stream         = new cudaStream_t[this->_op_param.group() * CUDNN_STREAMS_PER_GROUP];
    _handle         = new cudnnHandle_t[this->_op_param.group() * CUDNN_STREAMS_PER_GROUP];

    // Initialize algorithm arrays
    _fwd_algo       = new cudnnConvolutionFwdAlgo_t[this->_inputs.size()];
    _bwd_filter_algo = new cudnnConvolutionBwdFilterAlgo_t[this->_inputs.size()];
    _bwd_data_algo  = new cudnnConvolutionBwdDataAlgo_t[this->_inputs.size()];

    // initialize size arrays
    _workspace_fwd_sizes = new size_t[this->_inputs.size()];
    _workspace_bwd_filter_sizes = new size_t[this->_inputs.size()];
    _workspace_bwd_data_sizes = new size_t[this->_inputs.size()];

    // workspace data
    workspaceSizeInBytes = 0;
    workspaceData = NULL;
    workspace = new void*[this->_op_param.group() * CUDNN_STREAMS_PER_GROUP];

    for (size_t i = 0; i < this->_inputs.size(); ++i) {
        // initialize all to default algorithms
        _fwd_algo[i] = (cudnnConvolutionFwdAlgo_t)0;
        _bwd_filter_algo[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
        _bwd_data_algo[i] = (cudnnConvolutionBwdDataAlgo_t)0;
        // default algorithms don't require workspace
        _workspace_fwd_sizes[i] = 0;
        _workspace_bwd_data_sizes[i] = 0;
        _workspace_bwd_filter_sizes[i] = 0;
    }
    for (int g = 0; g < this->_op_param.group() * CUDNN_STREAMS_PER_GROUP; g++) {
        CUDA_CHECK(cudaStreamCreate(&_stream[g]));
        CUDNN_CHECK(cudnnCreate(&_handle[g]));
        CUDNN_CHECK(cudnnSetStream(_handle[g], _stream[g]));
        workspace[g] = NULL;
    }

    // Create filter descriptor.
    Context::create_filter_descriptor<Dtype>(&_filter_desc,
                                   this->_op_param.num_output()/this->_op_param.group(), weight_shape[1],
                                   kernel_h, kernel_w);

    // Create tensor descriptor(s) for data and corresponding convolution(s).
    for (int i = 0; i < this->_inputs.size(); i++) {
        cudnnTensorDescriptor_t input_desc;
        Context::create_tensor4d_descriptor<Dtype>(&input_desc);
        _input_descs.push_back(input_desc);

        cudnnTensorDescriptor_t output_desc;
        Context::create_tensor4d_descriptor<Dtype>(&output_desc);
        _output_descs.push_back(output_desc);

        cudnnConvolutionDescriptor_t conv_desc;
        Context::create_convolution_descriptor<Dtype>(&conv_desc);
        _conv_descs.push_back(conv_desc);
    }

    // Tensor descriptor for bias.
    if (this->_op_param.bias_term()) {
        Context::create_tensor4d_descriptor<Dtype>(&_bias_desc);
    }
    _handles_setup = true;

    infer_shape();
}

template<typename Dtype>
void Convolution<Dtype>::infer_shape() {
    size_t workspace_limit_bytes = 8*1024*1024;

    for (int i = 0; i < this->_inputs.size(); i++) {
      Context::set_tensor4d_descriptor<Dtype>(&_input_descs[i],
          this->_input_shapes[this->_inputs[i]][0],
          this->_input_shapes[this->_inputs[i]][1] / this->_op_param.group(), 
          this->_input_shapes[this->_inputs[i]][2], this->_input_shapes[this->_inputs[i]][3],
          this->_input_shapes[this->_inputs[i]][1] * this->_input_shapes[this->_inputs[i]][2] * this->_input_shapes[this->_inputs[i]][3],
          this->_input_shapes[this->_inputs[i]][2] * this->_input_shapes[this->_inputs[i]][3], 
          this->_input_shapes[this->_inputs[i]][3], 1);
      Context::set_tensor4d_descriptor<Dtype>(&_output_descs[i],
          this->_input_shapes[this->_inputs[i]][0],
          this->_output_shapes[this->_outputs[i]][1] / this->_op_param.group(), 
          this->_output_shapes[this->_outputs[i]][2], this->_output_shapes[this->_outputs[i]][3],
          this->_output_shapes[this->_outputs[i]][1]* this->_output_shapes[this->_outputs[i]][2] * this->_output_shapes[this->_outputs[i]][3],
          this->_output_shapes[this->_outputs[i]][2] * this->_output_shapes[this->_outputs[i]][3], 
          this->_output_shapes[this->_outputs[i]][3], 1);

      int pad_h, pad_w, stride_h, stride_w;
      if(this->_op_param.has_pad_h() && this->_op_param.has_pad_w()){
          pad_h = int(this->_op_param.pad_h());
          pad_w = int(this->_op_param.pad_w());
      }else{
          pad_h = this->_op_param.pad();
          pad_w = this->_op_param.pad();
      }

      if(this->_op_param.has_stride()){
          stride_h = int(this->_op_param.stride());
          stride_w = int(this->_op_param.stride());
      }else{
          stride_h = this->_op_param.stride_h();
          stride_w = this->_op_param.stride_w();
      }

      Context::set_convolution_descriptor<Dtype>(&_conv_descs[i], _input_descs[i],
          _filter_desc, pad_h, pad_w,
          stride_h, stride_w);

      // choose forward and backward algorithms + workspace(s)
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_handle[0],
        _input_descs[i],
        _filter_desc,
        _conv_descs[i],
        _output_descs[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,
        &_fwd_algo[i]));

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_handle[0],
        _input_descs[i],
        _filter_desc,
        _conv_descs[i],
        _output_descs[i],
        _fwd_algo[i],
        &(_workspace_fwd_sizes[i])));

      // choose backward algorithm for filter
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(_handle[0],
            _input_descs[i], _output_descs[i], _conv_descs[i], _filter_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            workspace_limit_bytes, &_bwd_filter_algo[i]) );

      // get workspace for backwards filter algorithm
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(_handle[0],
            _input_descs[i], _output_descs[i], _conv_descs[i], _filter_desc,
            _bwd_filter_algo[i], &_workspace_bwd_filter_sizes[i]));

      // choose backward algo for data
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_handle[0],
            _filter_desc, _output_descs[i], _conv_descs[i], _input_descs[i],
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &_bwd_data_algo[i]));

      // get workspace size
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_handle[0],
            _filter_desc, _output_descs[i], _conv_descs[i], _input_descs[i],
            _bwd_data_algo[i], &_workspace_bwd_data_sizes[i]) );
  }


  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < this->_inputs.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     _workspace_fwd_sizes[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     _workspace_bwd_data_sizes[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     _workspace_bwd_filter_sizes[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->_op_param.group() * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < this->_inputs.size(); i++) {
        _workspace_fwd_sizes[i] = 0;
        _workspace_bwd_filter_sizes[i] = 0;
        _workspace_bwd_data_sizes[i] = 0;
        _fwd_algo[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        _bwd_filter_algo[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        _bwd_data_algo[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->_op_param.group() * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->_op_param.group() * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->_op_param.bias_term()) {
    Context::set_tensor4d_descriptor<Dtype>(&_bias_desc,
        1, this->_output_shapes[this->_outputs[0]][1] / this->_op_param.group(), 1, 1);
  }

}

INSTANTIATE_CLASS(Convolution);
REGISTER_OPERATOR_CLASS(Convolution);
}//namespace
