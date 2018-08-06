//---------------------------------
//write by unsky
//---------------------------------
#ifndef SITA_STUFF_OPERATORS_ADD_OP_H
#define SITA_STUFF_OPERATORS_ADD_OP_H
#include <string>
#include <vector>
#include "sita/stuff/operator.h"
#include "sita/stuff/registry.h"
namespace sita{

template<typename Dtype>
class AddOp: public Operator<Dtype>{
public:
    AddOp(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws):Operator<Dtype>(opdef,gws){
        _add_op_param = opdef.add_op_param();
    }
    ~AddOp(){};
    void init();
    void forward();
    void backward();
    bool inline has_param(){ return _has_param;}

protected:
    bool _has_param = true;
    AddOpParameter11 _add_op_param;


};
}//namespace
#endif //SITA_STUFF_OPERATORS_ADD_OP_H
