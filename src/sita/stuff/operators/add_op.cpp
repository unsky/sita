//---------------------------------
//write by unsky
//---------------------------------
#include "sita/stuff/operators/add_op.h"
namespace sita{
template<typename Dtype>
void AddOp<Dtype>::forward(int num){
    _num = num;
    LOG(INFO)<<"AAAAAAAAAAAAAA";
};
template<typename Dtype>
void AddOp<Dtype>::backward(){
LOG(INFO)<<"NUM: "<<_num;
}
INSTANTIATE_CLASS(AddOp);
REGISTER_OPERATOR_CLASS(AddOp);
}//namespace
