//---------------------------------
//write by unsky
//---------------------------------
#ifndef SITA_STUFF_OPERATORS_ADD_OP_H
#define SITA_STUFF_OPERATORS_ADD_OP_H

#include "sita/stuff/operator.h"
#include "sita/stuff/registry.h"
namespace sita{

template<typename Dtype>
class AddOp: public Operator<Dtype>{
    public:
    AddOp(const OperatorDef& opdef, GlobalWorkSpace<Dtype> *gws):Operator<Dtype>(opdef,gws){}
    ~AddOp(){};
    void init();
    void forward();
    void backward();
    private:
    int _num;
    bool is_loss_op;
    bool is_data_op;
    bool has_param;
};
}//namespace
#endif //SITA_STUFF_OPERATORS_ADD_OP_H
