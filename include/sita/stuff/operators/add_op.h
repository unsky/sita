//---------------------------------
//write by unsky
//---------------------------------
#ifndef SITA_STUFF_OPERATORS_ADD_OP_H
#define SITA_STUFF_OPERATORS_ADD_OP_H

#include "sita/stuff/operator.h"
sita{
template<typename Dtype>
class AddOp: public Operator
{
    AddOp(const OperatorDef& opdef, GlobalWorkSpace<Dtype> *gws):Operator(opdef, gws){}
    void test();
}
}//namespace
#endif //SITA_STUFF_OPERATORS_ADD_OP_H