//
// Created by unsky on 15/08/18.
//

#ifndef SITA_DLFLOW_CONVOLUTION_OP_H
#define SITA_DLFLOW_CONVOLUTION_OP_H
#include "sita/dlflow/operator.h"
#include "sita/proto/sita.h"
namespace  sita{

template<typename Dtype>
class ConvolutionOp: public Operator<Dtype>{
public:
    ConvolutionOp(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws):Operator<Dtype>(opdef,gws){
        _convolution_op_param = opdef.convolution_op_param();
    }
    ~ConvolutionOp(){};
    void init();
    void forward();
    void backward(){};
    bool inline has_param(){ return _has_param;}

protected:
    bool _has_param = true;
    ConvolutionOpParameter _convolution_op_param;


};
}
#endif //SITA_DLFLOW_CONVOLUTION_OP_H
