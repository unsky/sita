//
// Created by unsky on 15/08/18.
//

#ifndef SITA_OPERATORS_BATCH_NORM_H
#define SITA_OPERATORS_BATCH_NORM_H
#include "sita/operator.h"
#include "sita/proto/sita.h"
namespace  sita{

template<typename Dtype>
class BatchNorm: public Operator<Dtype>{
public:
    BatchNorm(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws, std::string phase):Operator<Dtype>(opdef, gws, phase){
        _op_param = opdef.batch_norm_param();
    }
    ~BatchNorm();
    void init();
    void infer_shape();
    void forward();
    void backward();

    bool inline has_param(){ return _has_param;}

protected:
    bool _has_param = true;
    BatchNormParameter _op_param;

private:
  cudnnHandle_t* _handle;
  cudaStream_t*  _stream;
  // cuDNN descriptors / handles
  cudnnTensorDescriptor_t _input_desc, _output_desc;
  cudnnTensorDescriptor_t _scale_bias_mean_var_desc;
  cudnnBatchNormMode_t _mode;

  Tensor<Dtype> _save_mean, _save_inv_var;
  bool _handles_setup;
};
}
#endif //SITA_OPERATORS_BATCH_NORM_H