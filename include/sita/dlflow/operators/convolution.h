//
// Created by unsky on 15/08/18.
//

#ifndef SITA_DLFLOW_CONVOLUTION_H
#define SITA_DLFLOW_CONVOLUTION_H
#include "sita/dlflow/operator.h"
#include "sita/proto/sita.h"
namespace  sita{

template<typename Dtype>
class Convolution: public Operator<Dtype>{
public:
    Convolution(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws, std::string phase):Operator<Dtype>(opdef,gws, phase){
        _op_param = opdef.convolution_param();
    }
    ~Convolution();
    void init();
    void infer_shape();
    void forward();
    void backward();

    bool inline has_param(){ return _has_param;}

protected:
    bool _has_param = true;
    ConvolutionParameter _op_param;

private:
    bool _handles_setup;
    cudnnHandle_t* _handle;
    cudaStream_t*  _stream;

    // algorithms for forward and backwards convolutions
    cudnnConvolutionFwdAlgo_t *_fwd_algo;
    cudnnConvolutionBwdFilterAlgo_t *_bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t *_bwd_data_algo;

    //data desc
    std::vector<cudnnTensorDescriptor_t> _input_descs, _output_descs;
    cudnnTensorDescriptor_t    _bias_desc;
    cudnnFilterDescriptor_t      _filter_desc;
    std::vector<cudnnConvolutionDescriptor_t> _conv_descs;

    int _bottom_offset, _top_offset, _bias_offset;

    size_t *_workspace_fwd_sizes;
    size_t *_workspace_bwd_data_sizes;
    size_t *_workspace_bwd_filter_sizes;
    size_t workspaceSizeInBytes;  // size of underlying storage
    void *workspaceData;  // underlying storage
    void **workspace;  // aliases into workspaceData


};
}
#endif //SITA_DLFLOW_CONVOLUTION_H
