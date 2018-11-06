//
// Created by unsky on 15/08/18.
//

#ifndef SITA_DLFLOW_POOLING_H
#define SITA_DLFLOW_POOLING_H
#include "sita/operator.h"
#include "sita/proto/sita.h"
namespace  sita{
template<typename Dtype>
class Pooling: public Operator<Dtype>{
public:
    Pooling(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws, std::string phase):Operator<Dtype>(opdef, gws, phase){
        _op_param = opdef.pooling_param();
        _handles_setup = false;
    }
    ~Pooling();
    void init();
    void infer_shape();
    void forward();
    void backward();

    bool inline has_param(){ return _has_param;}

protected:
    bool _has_param = false;
    PoolingParameter _op_param;

private:
  cudnnHandle_t* _handle;
  cudaStream_t*  _stream;
  // cuDNN descriptors / handles
  cudnnTensorDescriptor_t _input_desc, _output_desc;
  cudnnPoolingDescriptor_t  _pooling_desc;
  cudnnPoolingMode_t        _mode;
  bool _handles_setup;
};

}