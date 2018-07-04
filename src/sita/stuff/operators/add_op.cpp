//---------------------------------
//write by unsky
//---------------------------------
#include "sita/stuff/operators/add_op.h"
sita{
template<typename Dtype>
void Add<Dtype>::test(){
    LOG(INFO)<<"AAAAAAAAAAAAAA";
};
INSTANTIATE_CLASS(AddOp);
REGISTER_OPERATOR_CLASS(AddOp);
}//namespace
