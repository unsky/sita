//
// Created by unsky on 15/08/18.
//

#ifndef SITA_DLFLOW_RELU_H
#define SITA_DLFLOW_RELU_H
#include "sita/dlflow/operator.h"
#include "sita/proto/sita.h"
namespace  sita{

template<typename Dtype>
class ReLU: public Operator<Dtype>{
public:
    ReLU(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws, std::string phase):Operator<Dtype>(opdef, gws, phase){
        _op_param = opdef.relu_param();
    }
    ~ReLU();
    void init();
    void infer_shape();
    void forward();
    void backward();

    bool inline has_param(){ return _has_param;}

protected:
    bool _has_param = false;
    ReLUParameter _op_param;

private:
  cudnnHandle_t* _handle;
  cudaStream_t*  _stream;
  // cuDNN descriptors / handles
  cudnnTensorDescriptor_t _input_desc, _output_desc;
  cudnnActivationDescriptor_t _activ_desc;
  bool _handles_setup;
};
}
#endif //SITA_DLFLOW_BATCH_NORM_H