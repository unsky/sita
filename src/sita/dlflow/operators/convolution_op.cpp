//
// Created by unsky on 16/08/18.
//
#include "sita/dlflow/operators/convolution_op.h"
namespace sita{

template<typename Dtype>
void ConvolutionOp<Dtype>::init(){

    //Initialize param input and outputs
    for(int i = 0; i < this->_inputs.size(); i++){
        Tensor<Dtype> *input = this->fetch_input(this->_inputs[i]);
        this->_input_shapes[this->_inputs[i]] = input->shape();
        CHECK_GT(input->count(), 0) << "check your graph, cannot infer " << this->_inputs[i] <<  " shape,in " << this->operator_name()<<"!!";
        LOG(INFO) << this->_inputs[i]<<": "<< this->fetch_input(this->_inputs[i])->shape_string();
    }

    int kernel_h;
    int kernel_w;
    if(this->_op_param.has_kernel_size()){
        kernel_h = this->_op_param.kernel_size();
        kernel_w = this->_op_param.kernel_size();
    }else{
        kernel_h = this->_op_param.kernel_h();
        kernel_w = this->_op_param.kernel_w();
    }
    std::vector<int > weight_shape;
    weight_shape.push_back(this->_op_param.num_output());
    weight_shape.push_back(this->_input_shapes[this->_inputs[0]][1]/this->_op_param.group());
    weight_shape.push_back(kernel_h);
    weight_shape.push_back(kernel_w);
    this->init_param("convolution_weight",weight_shape,this->_param_configs[0]);

    if(this->_op_param.bias_term()) {
        std::vector<int> bias_shape;
        bias_shape.push_back(this->_op_param.num_output());
        this->init_param("convolution_bias", bias_shape,this->_param_configs[1]);
    }

    Tensor<Dtype> * weight = this->fetch_param("convolution_weight");
    LOG(INFO) << "convolution_weight:" <<weight->shape_string();
    Tensor<Dtype> * bias = this->fetch_param("convolution_bias");
    LOG(INFO) << "convolution_bias:" <<bias->shape_string();

    std::vector<int > output_shape;
    output_shape.push_back(this->_input_shapes[this->_inputs[0]][0]);
    output_shape.push_back(this->_op_param.num_output());

    int in_height = this->_input_shapes[this->_inputs[0]][2];
    int in_width = this->_input_shapes[this->_inputs[0]][3];

    int pad_h, pad_w, dilation, stride_h, stride_w;

    if(this->_op_param.has_pad()){
        pad_h = int(this->_op_param.pad());
        kernel_w = int(this->_op_param.pad());
    }else{
        pad_h = this->_op_param.pad_h();
        pad_w = this->_op_param.pad_w();
    }

    if(this->_op_param.has_stride()){
        stride_h = int(this->_op_param.stride());
        stride_w = int(this->_op_param.stride());
    }else{
        stride_h = this->_op_param.stride_h();
        stride_w = this->_op_param.stride_w();
    }
    dilation =  this->_op_param.dilation();
    int out_height = (in_height + 2 * pad_h - dilation)/stride_h + 1;
    int out_width = (in_width + 2* pad_w - dilation)/stride_w +1;
    output_shape.push_back(out_height);
    output_shape.push_back(out_width);

    Tensor<Dtype> *output_data = this->fetch_output(this->_outputs[0]);
    output_data->reshape(output_shape);
    LOG(INFO) << this->_outputs[0]<<": "<< this->fetch_output(this->_outputs[0])->shape_string();


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
}

template<typename Dtype>
void ConvolutionOp<Dtype>::infer_shape() {

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
